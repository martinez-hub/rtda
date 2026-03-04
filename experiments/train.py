from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtda.data.cifar import build_train_loader
from rtda.models.model import build_model
from rtda.training.trainer import train_one_epoch
from rtda.utils.config import load_config
from rtda.utils.device import resolve_device
from rtda.utils.io import ensure_dir, write_json
from rtda.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override total number of training epochs without modifying config.",
    )
    p.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume training from")
    p.add_argument(
        "--resume-allow-config-drift",
        action="store_true",
        help="Allow resume even if current config differs from checkpoint config.",
    )
    return p.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def _restore_rng_state(state: dict) -> None:
    rng_state = state.get("rng_state")
    if not isinstance(rng_state, dict):
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.random.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _flatten_cfg(cfg, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(cfg, dict):
        for key in sorted(cfg.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_cfg(cfg[key], child_prefix))
        return out
    out[prefix] = json.dumps(cfg, sort_keys=True)
    return out


def _config_diff(old_cfg: dict, new_cfg: dict) -> list[str]:
    old_flat = _flatten_cfg(old_cfg)
    new_flat = _flatten_cfg(new_cfg)
    all_keys = sorted(set(old_flat.keys()) | set(new_flat.keys()))
    diffs: list[str] = []
    for key in all_keys:
        a = old_flat.get(key, "<MISSING>")
        b = new_flat.get(key, "<MISSING>")
        if a != b:
            diffs.append(f"{key}: {a} -> {b}")
    return diffs


def main():
    args = parse_args()
    cfg = load_config(args.config)
    target_epochs = int(cfg["train"]["epochs"]) if args.max_epochs is None else int(args.max_epochs)
    if target_epochs <= 0:
        raise ValueError("--max-epochs must be positive")

    seed = cfg["system"]["seed"] if args.seed is None else args.seed
    set_seed(seed)

    device = resolve_device(cfg["system"].get("device"))

    out_root = Path(args.output_dir or cfg["output"].get("root", "results"))
    resume_state = None
    resumed_from = None

    if args.resume:
        resume_ckpt = Path(args.resume)
        resume_state = _load_checkpoint(resume_ckpt, device)
        checkpoint_cfg = resume_state.get("config")
        if not isinstance(checkpoint_cfg, dict):
            raise ValueError("Resume checkpoint does not include a valid config dictionary.")
        if not args.resume_allow_config_drift:
            diffs = _config_diff(checkpoint_cfg, cfg)
            if diffs:
                preview = "\n".join(diffs[:20])
                tail = "\n... (additional differences omitted)" if len(diffs) > 20 else ""
                raise ValueError(
                    "Config drift detected while resuming in strict mode.\n"
                    "Use --resume-allow-config-drift to override.\n"
                    f"Differences:\n{preview}{tail}"
                )
        out_dir = ensure_dir(resume_ckpt.parent)
        run_id = out_dir.name
        resumed_from = str(resume_ckpt)
    else:
        run_id = f"{cfg['train']['method']}_seed{seed}_{int(time.time())}"
        out_dir = ensure_dir(out_root / run_id)

    model = build_model(cfg).to(device)
    momentum = float(cfg["train"].get("momentum", 0.9))
    nesterov = bool(cfg["train"].get("nesterov", True))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler_name = cfg["train"].get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=target_epochs, eta_min=float(cfg["train"].get("min_lr", 0.0))
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported train.scheduler: {scheduler_name}")

    start_epoch = 0
    history = []
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state.get("epoch", 0))
        history = list(resume_state.get("history", []))
        _restore_rng_state(resume_state)

    for epoch in range(start_epoch, target_epochs):
        loader = build_train_loader(cfg, epoch=epoch)
        metrics = train_one_epoch(model, loader, optimizer, device, cfg)
        if scheduler is not None:
            scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "loss": metrics.train_loss,
            "acc": metrics.train_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        last_ckpt_path = out_dir / "checkpoint_last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "config": cfg,
                "target_epochs": target_epochs,
                "seed": seed,
                "epoch": epoch + 1,
                "history": history,
                "rng_state": rng_state,
            },
            last_ckpt_path,
        )

    ckpt_path = out_dir / "model.pt"
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "config": cfg,
            "target_epochs": target_epochs,
            "seed": seed,
            "epoch": target_epochs,
            "history": history,
            "rng_state": rng_state,
        },
        ckpt_path,
    )

    write_json(
        out_dir / "metrics.json",
        {
            "run_id": run_id,
            "method": cfg["train"]["method"],
            "seed": seed,
            "target_epochs": target_epochs,
            "resumed_from": resumed_from,
            "resume_allow_config_drift": bool(args.resume_allow_config_drift),
            "start_epoch": start_epoch,
            "train": history,
            "checkpoint": str(ckpt_path),
            "checkpoint_last": str(out_dir / "checkpoint_last.pt"),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
