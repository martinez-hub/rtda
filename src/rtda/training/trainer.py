from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rtda.attacks.pgd import pgd_l2_attack
from rtda.training.losses import jsd_consistency_loss


@dataclass
class TrainMetrics:
    train_loss: float
    train_acc: float


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, cfg) -> TrainMetrics:
    model.train()
    method = cfg["train"]["method"]
    jsd_weight = float(cfg["train"].get("jsd_weight", 12.0))
    max_steps = int(cfg["train"].get("max_steps", 0))

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for step_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_steps > 0 and step_idx > max_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        if method == "rtda":
            clean, aug, labels = batch
            clean = clean.to(device)
            aug = aug.to(device)
            labels = labels.to(device)

            attack_cfg = cfg.get("attack", {})
            eps_scale = float(attack_cfg.get("pixel_scale", 255.0))
            epsilon = float(attack_cfg.get("epsilon", 2.0)) / eps_scale
            step_size = float(attack_cfg.get("step_size", 0.5)) / eps_scale
            num_steps = int(attack_cfg.get("num_steps", 7))
            random_start = bool(attack_cfg.get("random_start", True))

            was_training = model.training
            model.eval()
            adv = pgd_l2_attack(
                model=model,
                x_norm=clean,
                labels=labels,
                epsilon=epsilon,
                step_size=step_size,
                num_steps=num_steps,
                random_start=random_start,
            )
            if was_training:
                model.train()

            logits_clean = model(clean)
            logits_aug = model(aug)
            logits_adv = model(adv)

            ce = F.cross_entropy(logits_clean, labels)
            jsd = jsd_consistency_loss(logits_clean, logits_aug, logits_adv)
            loss = ce + jsd_weight * jsd
            acc = _acc(logits_clean, labels)
        elif method == "augmix":
            clean, aug1, aug2, labels = batch
            clean = clean.to(device)
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)
            labels = labels.to(device)

            logits_clean = model(clean)
            logits_aug1 = model(aug1)
            logits_aug2 = model(aug2)

            ce = F.cross_entropy(logits_clean, labels)
            jsd = jsd_consistency_loss(logits_clean, logits_aug1, logits_aug2)
            loss = ce + jsd_weight * jsd
            acc = _acc(logits_clean, labels)
        elif method == "adversarial":
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            attack_cfg = cfg.get("attack", {})
            eps_scale = float(attack_cfg.get("pixel_scale", 255.0))
            epsilon = float(attack_cfg.get("epsilon", 2.0)) / eps_scale
            step_size = float(attack_cfg.get("step_size", 0.5)) / eps_scale
            num_steps = int(attack_cfg.get("num_steps", 7))
            random_start = bool(attack_cfg.get("random_start", True))

            was_training = model.training
            model.eval()
            adv = pgd_l2_attack(
                model=model,
                x_norm=inputs,
                labels=labels,
                epsilon=epsilon,
                step_size=step_size,
                num_steps=num_steps,
                random_start=random_start,
            )
            if was_training:
                model.train()

            logits_adv = model(adv)
            loss = F.cross_entropy(logits_adv, labels)
            acc = _acc(logits_adv, labels)
        elif method == "rtda":
            # RTDA: Robust Training with Data Augmentation
            # Generate adversarial examples from clean images (not augmented)
            clean, aug, labels = batch
            clean = clean.to(device)
            aug = aug.to(device)
            labels = labels.to(device)

            attack_cfg = cfg.get("attack", {})
            eps_scale = float(attack_cfg.get("pixel_scale", 255.0))
            epsilon = float(attack_cfg.get("epsilon", 2.0)) / eps_scale
            step_size = float(attack_cfg.get("step_size", 0.5)) / eps_scale
            num_steps = int(attack_cfg.get("num_steps", 7))
            random_start = bool(attack_cfg.get("random_start", True))

            was_training = model.training
            model.eval()
            # Generate adversarial examples from clean images
            adv = pgd_l2_attack(
                model=model,
                x_norm=clean,  # Use clean images as base
                labels=labels,
                epsilon=epsilon,
                step_size=step_size,
                num_steps=num_steps,
                random_start=random_start,
            )
            if was_training:
                model.train()

            # Train with CE loss on adversarial examples
            logits_adv = model(adv)
            loss = F.cross_entropy(logits_adv, labels)
            acc = _acc(logits_adv, labels)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            acc = _acc(logits, labels)

        loss.backward()
        optimizer.step()

        b = labels.size(0)
        total_loss += loss.item() * b
        total_acc += acc * b
        n += b

    return TrainMetrics(train_loss=total_loss / max(n, 1), train_acc=total_acc / max(n, 1))
