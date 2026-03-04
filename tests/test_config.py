from rtda.utils.config import ConfigError, validate_config


def test_validate_config_accepts_valid():
    cfg = {
        "model": {"name": "wrn50_2"},
        "dataset": {"data_root": "./data", "cifar10c_root": "./data/CIFAR-10-C"},
        "train": {"method": "vanilla", "batch_size": 32, "epochs": 1, "max_steps": 0},
        "augment": {"severity": 3},
        "system": {"num_workers": 0},
        "output": {"root": "results"},
    }
    validate_config(cfg)


def test_validate_config_rejects_method():
    cfg = {
        "model": {"name": "wrn50_2"},
        "dataset": {"data_root": "./data", "cifar10c_root": "./data/CIFAR-10-C"},
        "train": {"method": "bad", "batch_size": 32, "epochs": 1, "max_steps": 0},
        "augment": {"severity": 3},
        "system": {"num_workers": 0},
        "output": {"root": "results"},
    }
    try:
        validate_config(cfg)
        assert False, "expected error"
    except ConfigError:
        assert True


def test_validate_config_rejects_invalid_attack_for_rtda():
    cfg = {
        "model": {"name": "wrn50_2"},
        "dataset": {"data_root": "./data", "cifar10c_root": "./data/CIFAR-10-C"},
        "train": {"method": "rtda", "batch_size": 32, "epochs": 1, "max_steps": 0},
        "augment": {"severity": 3},
        "attack": {"epsilon": -1.0, "step_size": 0.5, "num_steps": 7, "pixel_scale": 255.0},
        "system": {"num_workers": 0},
        "output": {"root": "results"},
    }
    try:
        validate_config(cfg)
        assert False, "expected error"
    except ConfigError:
        assert True


def test_validate_config_accepts_adversarial_with_attack():
    cfg = {
        "model": {"name": "wrn50_2"},
        "dataset": {"data_root": "./data", "cifar10c_root": "./data/CIFAR-10-C"},
        "train": {"method": "adversarial", "batch_size": 32, "epochs": 1, "max_steps": 0},
        "augment": {"severity": 3},
        "attack": {"epsilon": 1.0, "step_size": 0.3, "num_steps": 7, "pixel_scale": 1.0},
        "system": {"num_workers": 0},
        "output": {"root": "results"},
    }
    validate_config(cfg)


def test_validate_config_rejects_invalid_eval_attack():
    cfg = {
        "model": {"name": "wrn50_2"},
        "dataset": {"data_root": "./data", "cifar10c_root": "./data/CIFAR-10-C"},
        "train": {"method": "vanilla", "batch_size": 32, "epochs": 1, "max_steps": 0},
        "augment": {"severity": 3},
        "eval": {"adversarial_attack": {"epsilons": [1.0], "num_steps": 10, "step_size_mode": "bad"}},
        "system": {"num_workers": 0},
        "output": {"root": "results"},
    }
    try:
        validate_config(cfg)
        assert False, "expected error"
    except ConfigError:
        assert True
