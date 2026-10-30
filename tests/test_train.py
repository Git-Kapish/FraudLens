import pytest

from fraud import train
from fraud.config import load_config


def test_every_configured_deployable_model_has_a_calibration_recipe():
    train.check_deployable()  # the shipped config must pass


def test_deployable_model_without_recipe_fails_before_training(monkeypatch):
    training = load_config()["training"]
    monkeypatch.setitem(training, "deployable_models", [*training["deployable_models"], "M2_rf"])
    with pytest.raises(ValueError, match="M2_rf"):
        train.check_deployable()
    with pytest.raises(ValueError, match="M2_rf"):
        train.main()  # main checks first, so this fails in seconds, not after training
