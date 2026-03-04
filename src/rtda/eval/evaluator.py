from __future__ import annotations

from statistics import mean

import torch

from rtda.attacks.pgd import pgd_l2_attack
from rtda.data.cifar import CIFAR10C_CORRUPTIONS, build_cifar10c_loader, build_test_loader


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def evaluate_clean(model, cfg, device):
    loader = build_test_loader(cfg)
    return evaluate_loader(model, loader, device)


def evaluate_cifar10c(model, cfg, device):
    rows = []
    severities = cfg["eval"].get("corruption_severities", [1, 2, 3, 4, 5])
    all_acc = []

    for corruption in CIFAR10C_CORRUPTIONS:
        for severity in severities:
            loader = build_cifar10c_loader(cfg, corruption=corruption, severity=severity)
            acc = evaluate_loader(model, loader, device)
            all_acc.append(acc)
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": round(acc, 6),
                }
            )

    return rows, float(mean(all_acc)) if all_acc else 0.0


def evaluate_pgd(model, cfg, device):
    loader = build_test_loader(cfg)
    attack_cfg = cfg.get("eval", {}).get("adversarial_attack", {})

    eps_scale = float(attack_cfg.get("pixel_scale", 1.0))
    epsilons = [float(x) for x in attack_cfg.get("epsilons", [1.0])]
    num_steps = int(attack_cfg.get("num_steps", 10))
    random_start = bool(attack_cfg.get("random_start", True))
    step_size_mode = attack_cfg.get("step_size_mode", "proportional")
    step_size_factor = float(attack_cfg.get("step_size_factor", 2.5 / 7.0))
    fixed_step_size = float(attack_cfg.get("step_size", 0.35714285714285715))

    rows = []
    for eps in epsilons:
        epsilon = eps / eps_scale
        if step_size_mode == "proportional":
            step_size = (step_size_factor * eps) / eps_scale
        else:
            step_size = fixed_step_size / eps_scale

        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            adv = pgd_l2_attack(
                model=model,
                x_norm=x,
                labels=y,
                epsilon=epsilon,
                step_size=step_size,
                num_steps=num_steps,
                random_start=random_start,
            )
            with torch.no_grad():
                logits = model(adv)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        rows.append({"epsilon": eps, "accuracy": correct / max(total, 1)})
    return rows
