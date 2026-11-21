import numpy as np
import pytest
import torch
from zipfile import ZipFile

from egophys.cli import envelope, export_submission, metrics, training_loss
from egophys.extract import da3_windows
from egophys.model import EgoPhys, LAGS, TARGETS, VARIANTS, lagged_delta


torch.set_num_threads(1)


def fixture():
    generator = torch.Generator().manual_seed(1)
    return {
        "dino_grid": torch.randn(1, 5, 7, 7, 1024, generator=generator),
        "da3_grid": torch.randn(1, 5, 7, 7, 2048, generator=generator),
        "dino_mean": torch.randn(1, 5, 1024, generator=generator),
        "dino_cls": torch.randn(1, 5, 1024, generator=generator),
        "time": torch.randn(1, 5, 2, generator=generator),
        "visual_valid": torch.tensor([[True, True, True, False, False]]),
        "geometry_valid": torch.tensor([[True, False, True, False, False]]),
        "target": torch.tensor([[12., 1.5]]),
        "mean_force_target": torch.ones(1, 5),
        "mean_force_valid": torch.tensor([[True, False, True, False, False]]),
        "contact": torch.tensor([[True, False, True, False, False]]),
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_variants_ignore_padding(variant):
    model = EgoPhys(variant).eval()
    batch = fixture()
    with torch.no_grad():
        predictions = model(batch)
        changed = dict(batch)
        for key in ("dino_cls", "dino_mean", "dino_grid", "da3_grid", "time"):
            changed[key] = batch[key].clone()
            changed[key][:, 3:] = 50.0
        predictions_changed = model(changed)
    for target in TARGETS:
        assert torch.isfinite(predictions[target]).all()
        assert (predictions[target] >= 0).all()
        torch.testing.assert_close(predictions_changed[target], predictions[target])
        assert not predictions[target + "_attention"][:, 3:].any()
        torch.testing.assert_close(predictions[target + "_attention"].sum(1), torch.ones(1))


def test_short_deltas_and_loaded_envelope():
    signal = torch.tensor([[[1.], [2.], [8.]]])
    valid = torch.tensor([[True, False, True]])
    for lag in LAGS:
        assert not lagged_delta(signal, valid, lag).any()
    torch.testing.assert_close(
        lagged_delta(signal, torch.ones_like(valid), 1),
        torch.tensor([[[0.], [1.], [6.]]])
    )
    valid = torch.tensor([[True, True, True, False]])
    contact = torch.tensor([[True, False, True, True]])
    torch.testing.assert_close(envelope(contact, valid), valid)
    for length in (1, 10, 16, 17, 155):
        size, starts, owner = da3_windows(length)
        for frame, index in enumerate(owner):
            assert starts[index] <= frame < starts[index] + size


def test_loss_and_metrics():
    batch = fixture()
    model = EgoPhys()
    loss = training_loss(model(batch), batch, torch.tensor([20.0, 3.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dense_head.weight.grad is not None
    result = metrics(np.array([[2., 1.], [1., 2.]]), np.array([[1., 1.], [2., 2.]]))
    assert result["max_force_n"] == {"mae": 1.0, "median_ae": 1.0, "rmse": 1.0, "mre": 0.75}
    assert result["work_j"]["mae"] == 0.0


def test_submission(tmp_path):
    ids = [("recording__clip", "clip")]
    export_submission(tmp_path, ids, np.array([[2., 1.]]), [ids[0][0]])
    with ZipFile(tmp_path / "submission.zip") as archive:
        assert archive.namelist() == ["predictions.json"]
        assert archive.read("predictions.json") == (tmp_path / "predictions.json").read_bytes()
    with pytest.raises(ValueError, match="requested IDs"):
        export_submission(tmp_path, ids, np.array([[2., 1.]]), ["different"])
