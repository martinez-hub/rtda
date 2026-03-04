from __future__ import annotations

import torch
import torch.nn.functional as F


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_NORM_CACHE: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _mean_std(device: torch.device):
    key = (device.type, device.index if device.index is not None else -1, "float32")
    cached = _NORM_CACHE.get(key)
    if cached is not None:
        return cached
    mean = torch.tensor(_CIFAR10_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR10_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    _NORM_CACHE[key] = (mean, std)
    return mean, std


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean, std = _mean_std(x.device)
    return x * std + mean


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean, std = _mean_std(x.device)
    return (x - mean) / std


def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factor = torch.clamp(eps / norm, max=1.0)
    projected = flat * factor
    return projected.view_as(delta)


def pgd_l2_attack(
    model,
    x_norm: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    num_steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    x = _denormalize(x_norm).detach()

    if random_start:
        delta = torch.randn_like(x)
        delta = _project_l2(delta, epsilon)
    else:
        delta = torch.zeros_like(x)

    adv = (x + delta).clamp(0.0, 1.0)

    for _ in range(num_steps):
        adv = adv.detach().requires_grad_(True)
        logits = model(_normalize(adv))
        loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        grad_unit = grad / grad_norm.view(-1, 1, 1, 1)

        adv = adv + step_size * grad_unit
        delta = _project_l2(adv - x, epsilon)
        adv = (x + delta).clamp(0.0, 1.0)

    return _normalize(adv.detach())
