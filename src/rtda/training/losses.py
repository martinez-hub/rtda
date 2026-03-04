from __future__ import annotations

import torch
import torch.nn.functional as F


def jsd_consistency_loss(logits_clean, logits_aug1, logits_aug2):
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0)

    kl_clean = F.kl_div(torch.log(p_mixture), p_clean, reduction="batchmean")
    kl_aug1 = F.kl_div(torch.log(p_mixture), p_aug1, reduction="batchmean")
    kl_aug2 = F.kl_div(torch.log(p_mixture), p_aug2, reduction="batchmean")
    return (kl_clean + kl_aug1 + kl_aug2) / 3.0
