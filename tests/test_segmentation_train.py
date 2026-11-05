import torch

from segmentation.losses import BCEDiceLoss
from segmentation.train import collect_predictions, train_one_epoch


class ConstantModel(torch.nn.Module):
    """Tiny stand-in for the U-Net: one learnable logit broadcast to the frame."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.logit.expand(x.shape[0], 1, x.shape[2], x.shape[3])


def _batches():
    return [
        {
            "image": torch.zeros(2, 3, 32, 32),
            "mask": torch.ones(2, 1, 32, 32),
            "stem": ["S01_001", "S01_002"],
        }
    ]


def test_one_step_updates_weights_and_returns_finite_loss():
    torch.manual_seed(0)
    model = ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.logit.detach().clone()

    loss = train_one_epoch(model, _batches(), optimizer, BCEDiceLoss(), torch.device("cpu"))

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0
    assert not torch.equal(before, model.logit.detach())


def test_collect_predictions_returns_one_map_per_stem():
    model = ConstantModel()
    loss, probabilities, targets, stems = collect_predictions(
        model, _batches(), BCEDiceLoss(), torch.device("cpu")
    )
    assert torch.isfinite(torch.tensor(loss))
    assert stems == ["S01_001", "S01_002"]
    assert len(probabilities) == 2
    assert probabilities[0].shape == (32, 32)
    assert targets[0].shape == (32, 32)
    assert 0.0 <= float(probabilities[0].mean()) <= 1.0
