from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtda.eval.evaluator import evaluate_cifar10c, evaluate_clean, evaluate_pgd
from rtda.models.model import build_model
from rtda.utils.config import load_config
from rtda.utils.device import resolve_device
from rtda.utils.io import ensure_dir, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def _find_latest_checkpoint(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("*/model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {results_root}")
    return candidates[0]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["system"].get("device"))

    results_root = Path(cfg["output"].get("root", "results"))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint(results_root)

    state = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    model.to(device)

    clean_acc = evaluate_clean(model, cfg, device)
    adv_rows = evaluate_pgd(model, cfg, device)
    adv_by_eps = {str(x["epsilon"]): x["accuracy"] for x in adv_rows}
    adv_acc = adv_by_eps.get("1.0")
    if adv_acc is None and adv_rows:
        adv_acc = adv_rows[0]["accuracy"]
    rows, avg_c_acc = evaluate_cifar10c(model, cfg, device)

    eval_dir = ensure_dir(ckpt_path.parent)
    write_csv(eval_dir / "cifar10c_per_corruption.csv", rows)
    write_csv(eval_dir / "pgd_per_epsilon.csv", adv_rows)
    write_json(
        eval_dir / "eval_metrics.json",
        {
            "checkpoint": str(ckpt_path),
            "clean_accuracy": clean_acc,
            "pgd_adversarial_accuracy": adv_acc,
            "pgd_adversarial_accuracy_by_epsilon": adv_by_eps,
            "cifar10c_mean_accuracy": avg_c_acc,
        },
    )
    print(f"clean_accuracy={clean_acc:.4f}")
    print(f"pgd_adversarial_accuracy={adv_acc:.4f}")
    print(f"cifar10c_mean_accuracy={avg_c_acc:.4f}")


if __name__ == "__main__":
    main()
