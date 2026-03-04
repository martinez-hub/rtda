from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtda.utils.config import load_config
from rtda.utils.io import ensure_dir, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def latest_run_dir(results_root: Path) -> Path:
    run_dirs = [p for p in results_root.iterdir() if p.is_dir() and (p / "model.pt").exists()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with model.pt found under {results_root}")
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_eval_metrics(run_dir: Path) -> dict:
    path = run_dir / "eval_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval metrics: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    methods = cfg["reproduce"]["methods"]
    tolerance = float(cfg["reproduce"].get("tolerance", 2.0))
    targets = cfg["reproduce"].get("paper_targets", {})

    project_root = PROJECT_ROOT
    results_root = project_root / cfg["output"].get("root", "results")

    method_metrics = {}
    for method in methods:
        method_cfg = project_root / f"experiments/configs/{method}_cifar10.yaml"
        run([sys.executable, str(project_root / "experiments/train.py"), "--config", str(method_cfg)])
        latest = latest_run_dir(results_root)
        run(
            [
                sys.executable,
                str(project_root / "experiments/eval.py"),
                "--config",
                str(method_cfg),
                "--checkpoint",
                str(latest / "model.pt"),
            ]
        )
        method_metrics[method] = load_eval_metrics(latest)

    comparisons = {}
    for metric_name, target in targets.items():
        candidate = method_metrics.get("rtda", {}).get(metric_name)
        if candidate is None:
            comparisons[metric_name] = {"target": target, "actual": None, "pass": False}
            continue
        comparisons[metric_name] = {
            "target": target,
            "actual": candidate,
            "delta": candidate - target,
            "pass": abs(candidate - target) <= tolerance,
        }

    summary = {
        "methods": methods,
        "tolerance": tolerance,
        "paper_targets": targets,
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "status": "pass" if all(x.get("pass", False) for x in comparisons.values()) else "needs_review",
        "results_root": str(results_root),
    }
    summary_dir = ensure_dir(results_root / "summary")
    write_json(summary_dir / "reproduction_report.json", summary)
    print(f"Wrote summary: {summary_dir / 'reproduction_report.json'}")


if __name__ == "__main__":
    main()
