from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when config is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required_roots = ["dataset", "train", "augment", "system", "output"]
    for key in required_roots:
        if key not in cfg:
            raise ConfigError(f"Missing root key: {key}")

    method = cfg["train"].get("method")
    if method not in {"vanilla", "adversarial", "augmix", "rtda"}:
        raise ConfigError(
            "train.method must be one of: vanilla, adversarial, augmix, rtda"
        )

    severity = int(cfg["augment"].get("severity", 3))
    if severity < 1 or severity > 5:
        raise ConfigError("augment.severity must be in [1, 5]")

    batch_size = int(cfg["train"].get("batch_size", 0))
    if batch_size <= 0:
        raise ConfigError("train.batch_size must be positive")
    epochs = int(cfg["train"].get("epochs", 0))
    if epochs <= 0:
        raise ConfigError("train.epochs must be positive")

    max_steps = int(cfg["train"].get("max_steps", 0))
    if max_steps < 0:
        raise ConfigError("train.max_steps must be >= 0")
    scheduler = cfg["train"].get("scheduler", "cosine")
    if scheduler not in {"cosine", "none"}:
        raise ConfigError("train.scheduler must be one of: cosine, none")
    model_name = cfg.get("model", {}).get("name", "wrn50_2")
    if model_name not in {"wrn50_2", "resnet18"}:
        raise ConfigError("model.name must be one of: wrn50_2, resnet18")
    prefetch_factor = int(cfg["system"].get("prefetch_factor", 2))
    if prefetch_factor <= 0:
        raise ConfigError("system.prefetch_factor must be positive")

    if method in {"adversarial", "rtda"}:
        attack = cfg.get("attack", {})
        eps = float(attack.get("epsilon", 2.0))
        step = float(attack.get("step_size", 0.5))
        n_steps = int(attack.get("num_steps", 7))
        scale = float(attack.get("pixel_scale", 255.0))
        if eps <= 0 or step <= 0 or n_steps <= 0 or scale <= 0:
            raise ConfigError("attack.{epsilon,step_size,num_steps,pixel_scale} must be positive")

    eval_attack = cfg.get("eval", {}).get("adversarial_attack", {})
    if eval_attack:
        epsilons = eval_attack.get("epsilons", [1.0])
        if not isinstance(epsilons, list) or len(epsilons) == 0:
            raise ConfigError("eval.adversarial_attack.epsilons must be a non-empty list")
        if any(float(x) <= 0 for x in epsilons):
            raise ConfigError("eval.adversarial_attack.epsilons must all be positive")
        eval_steps = int(eval_attack.get("num_steps", 10))
        if eval_steps <= 0:
            raise ConfigError("eval.adversarial_attack.num_steps must be positive")
        eval_scale = float(eval_attack.get("pixel_scale", 1.0))
        if eval_scale <= 0:
            raise ConfigError("eval.adversarial_attack.pixel_scale must be positive")
        mode = eval_attack.get("step_size_mode", "proportional")
        if mode not in {"proportional", "fixed"}:
            raise ConfigError("eval.adversarial_attack.step_size_mode must be one of: proportional, fixed")
        if mode == "proportional":
            factor = float(eval_attack.get("step_size_factor", 2.5 / 7.0))
            if factor <= 0:
                raise ConfigError("eval.adversarial_attack.step_size_factor must be positive")
        else:
            step = float(eval_attack.get("step_size", 0.35714285714285715))
            if step <= 0:
                raise ConfigError("eval.adversarial_attack.step_size must be positive")
