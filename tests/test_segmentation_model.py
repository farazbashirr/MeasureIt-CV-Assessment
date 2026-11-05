import pytest
import torch

from segmentation.losses import BCEDiceLoss
from segmentation.metrics import MetricResult
from segmentation.model import (
    ARCHITECTURE,
    UNetMobileNetV3Small,
    count_parameters,
)
from segmentation.train import (
    CHECKPOINT_SCHEMA,
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def _target(batch=2, size=32):
    torch.manual_seed(7)
    return (torch.rand(batch, 1, size, size) > 0.5).float()


def test_forward_shape_and_parameters():
    model = UNetMobileNetV3Small(pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64)
    assert torch.isfinite(logits).all()
    assert count_parameters(model) > 1_000_000
    assert ARCHITECTURE == "unet_mobilenetv3_small"


def test_eval_is_deterministic():
    model = UNetMobileNetV3Small(pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        first = model(x)
        second = model(x)
    assert torch.equal(first, second)


def test_loss_prefers_correct_prediction():
    target = _target()
    perfect = torch.where(target > 0.5, 10.0, -10.0)
    inverted = -perfect
    loss_fn = BCEDiceLoss()
    correct_loss = loss_fn(perfect, target)
    wrong_loss = loss_fn(inverted, target)
    assert correct_loss < 0.01
    assert wrong_loss > correct_loss
    assert torch.isfinite(wrong_loss)


def test_backward_produces_finite_gradients():
    model = UNetMobileNetV3Small(pretrained=False)
    target = _target(batch=1, size=64)
    loss = BCEDiceLoss()(model(torch.zeros(1, 3, 64, 64)), target)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_checkpoint_round_trip(tmp_path):
    model = UNetMobileNetV3Small(pretrained=False)
    config = TrainConfig(
        dataset="dataset",
        splits="dataset/splits.csv",
        out=str(tmp_path),
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=1,
        image_size=64,
        workers=0,
        pretrained=False,
        patience=1,
        allow_partial=False,
        resume=None,
    )
    metrics = MetricResult(
        available=True,
        reason=None,
        n_images=1,
        mean_iou=0.5,
        precision=0.5,
        recall=0.5,
        map50=0.5,
        map50_95=0.25,
    )
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model=model,
        config=config,
        epoch=3,
        metrics=metrics,
        best_iou=0.5,
    )
    assert path.is_file()

    loaded, payload = load_checkpoint(path)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["epoch"] == 3
    assert payload["image_size"] == 64
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_seed_everything_is_reproducible():
    seed_everything(123)
    first = torch.rand(5)
    seed_everything(123)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        BCEDiceLoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 7))
