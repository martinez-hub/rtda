# RTDA: Robust Training with Data Augmentation

Official implementation of **Robust Training with Data Augmentation for Medical Imaging Classification** for CIFAR-10 and CIFAR-10-C in PyTorch.

**Paper**: [Robust Training with Data Augmentation for Medical Imaging Classification](https://arxiv.org/abs/2506.17133)
**Authors**: Josué Martínez-Martínez, Olivia Brown, Mostafa Karami, Sheida Nabavi
**Venue**: 9th International Workshop on Health Intelligence (W3PHIAI-25) @ AAAI 2025

---

## Method

RTDA combines adversarial training with data augmentation to achieve superior robustness against adversarial attacks while maintaining high clean accuracy and improved generalization under distribution shift.

**Loss function:**

`L = CE(f(x_adv), y)`

where `x_adv` is generated via L2-PGD from **clean images**.

**Key difference from standard adversarial training:**
- Standard adversarial training: Uses standard data augmentation (crop, flip)
- **RTDA**: Uses **AugMix data augmentation** + adversarial training
- The combination of AugMix's diverse augmentations with adversarial training significantly improves robustness to both adversarial attacks and natural corruptions.

---

## Installation

```bash
cd rtda
python3.11 -m venv .venv && source .venv/bin/activate
make install
```

If `python3.11` is not on your PATH, use any Python >= 3.11:

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

---

## Quick Start

### Smoke Test (CPU-friendly)

```bash
make smoke
```

Configured for CPU-only development (small batch, 1 epoch, 2 max steps).

### Train RTDA

```bash
make train
# or
make run-rtda
```

Train fewer epochs:

```bash
python experiments/train.py --config experiments/configs/rtda_cifar10.yaml --max-epochs 20
```

### Resume Training

```bash
python experiments/train.py \
  --config experiments/configs/rtda_cifar10.yaml \
  --resume results/<run_id>/checkpoint_last.pt \
  --max-epochs 120
```

Or with Make:

```bash
make resume CHECKPOINT=results/<run_id>/checkpoint_last.pt
```

### Evaluate CIFAR-10 + CIFAR-10-C

```bash
make eval
```

Evaluation reports:
- Clean CIFAR-10 test accuracy
- PGD adversarial accuracy (multiple epsilon values)
- CIFAR-10-C corruption robustness (mean accuracy across 15 corruptions × 5 severities)

---

## Comparison Baselines

The repository includes implementations of comparison methods from the paper:

```bash
make run-vanilla        # Standard training
make run-adversarial    # PGD adversarial training
make run-augmix         # AugMix (Hendrycks et al.)
make run-robustaugmix   # RobustAugMix (Martínez-Martínez & Brown, 2022)
make run-rtda           # RTDA (this paper)
```

---

## Configuration

Paper-aligned defaults in `experiments/configs/rtda_cifar10.yaml`:
- **Model**: WRN-50-2 (Wide ResNet)
- **Training**: 100 epochs, SGD + Nesterov, cosine LR schedule
- **Preprocessing**: Random horizontal flip + random crop
- **Attack**: L2-PGD with `epsilon=1.0`, `num_steps=7`, `step_size=2.5*epsilon/7`
- **AugMix**: Follows Google AugMix with per-op sampled severity levels

---

## Dataset Setup

- **CIFAR-10**: Auto-downloaded by torchvision to `dataset.data_root`
- **CIFAR-10-C**: Should exist under `dataset.cifar10c_root` with files:
  - `gaussian_noise.npy`, `shot_noise.npy`, etc. (15 corruption types)
  - `labels.npy`

Download CIFAR-10-C from: https://zenodo.org/record/2535967

---

## Reproducibility

The repository provides complete reproducibility controls:
- Global seed control for Python, NumPy, and PyTorch RNGs
- Deterministic DataLoader seeding
- Training checkpoints include model, optimizer, scheduler, config, seed, and RNG state
- Strict resume policy prevents accidental config drift

### Reproduce All Methods

```bash
make reproduce
```

Trains and evaluates all methods (vanilla, adversarial, augmix, robustaugmix, rtda) with paper settings.

---

## Docker (CPU, Mac-friendly)

```bash
make docker-build
make docker-smoke
```

---

## Outputs

Training and evaluation produce:
- `results/<run_id>/metrics.json` - Training metrics per epoch
- `results/<run_id>/checkpoint_last.pt` - Latest checkpoint
- `results/<run_id>/eval_metrics.json` - Final evaluation results
- `results/<run_id>/cifar10c_per_corruption.csv` - Per-corruption accuracy
- `results/<run_id>/pgd_per_epsilon.csv` - Adversarial accuracy by epsilon
- `results/summary/reproduction_report.json` - Summary of all methods

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{martinez2025rtda,
  title={Robust Training with Data Augmentation for Medical Imaging Classification},
  author={Mart{\\'i}nez-Mart{\\'i}nez, Josu{\\'e} and Brown, Olivia and Karami, Mostafa and Nabavi, Sheida},
  booktitle={9th International Workshop on Health Intelligence (W3PHIAI-25) at AAAI},
  year={2025},
  url={https://arxiv.org/abs/2506.17133}
}
```

---

## Related Work

This repository also includes implementations of related methods:

**RobustAugMix** (Martínez-Martínez & Brown, NeurIPS 2022):
```bibtex
@inproceedings{martinez2022robustaugmix,
  title={RobustAugMix: Joint Optimization of Natural and Adversarial Robustness},
  author={Mart{\\'i}nez-Mart{\\'i}nez, Josu{\\'e} and Brown, Olivia},
  booktitle={ML Safety Workshop at NeurIPS},
  year={2022},
  url={https://openreview.net/forum?id=8MfPfECiFET}
}
```

**AugMix** (Hendrycks et al., ICLR 2020):
```bibtex
@inproceedings{hendrycks2020augmix,
  title={AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

---

## License

MIT License - see LICENSE file for details
