import torch
from torch.utils.data import DataLoader, TensorDataset

from rtda.models.model import build_model
from rtda.training.trainer import train_one_epoch


def test_smoke_train_vanilla_one_batch():
    cfg = {"train": {"method": "vanilla", "jsd_weight": 12.0}}
    model = build_model(cfg)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    m = train_one_epoch(model, loader, optimizer, torch.device("cpu"), cfg)
    assert m.train_loss > 0
    assert 0.0 <= m.train_acc <= 1.0
