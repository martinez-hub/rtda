# Mapping Notes

This repo uses a hybrid port strategy inspired by the Google AugMix reference implementation.

- AugMix op sampling and chain mixing: adapted into `src/robustaugmix/augmentations/augmix.py`
- AugMix op-set and severity behavior verification: `docs/augmix_verification.md`
- CIFAR-10/CIFAR-10-C evaluation flow: implemented in `src/robustaugmix/eval/evaluator.py`
- RobustAugMix objective `CE(clean) + lambda * JSD(clean, augmix, pgd_adv)`: implemented in `src/robustaugmix/training/trainer.py`
- L2-PGD adversarial example generation: implemented in `src/robustaugmix/attacks/pgd.py`
- Baselines for user comparisons: `vanilla`, `adversarial`, `augmix`, `robustaugmix`
- True resume training (optimizer/scheduler/RNG restoration): implemented in `experiments/train.py`
