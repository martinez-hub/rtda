# AugMix Verification Notes

Reference repository: https://github.com/google-research/augmix
Reference file: `augmentations.py`

This implementation is aligned to the Google AugMix operator behavior used in the reference code:

- Base ops: `autocontrast`, `equalize`, `posterize`, `rotate`, `solarize`, `shear_x`, `shear_y`, `translate_x`, `translate_y`
- All-ops extension: `color`, `contrast`, `brightness`, `sharpness`
- Per-op severity sampling: random `sample_level` in `[0.1, severity]`
- Mixture: Dirichlet width weights + Beta mixture coefficient

Code mapping in this repo:
- Operator parameterization and op lists: `src/robustaugmix/augmentations/ops.py`
- AugMix mixture policy: `src/robustaugmix/augmentations/augmix.py`
- Config control: `augment.all_ops` in `experiments/configs/*.yaml`
