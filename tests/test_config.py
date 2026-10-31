from pathlib import Path

import pytest

from sentinel.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_defaults():
    s = Settings(_env_file=None)
    assert s.plan_mode == "deterministic"
    assert s.autonomy_level == 2
    assert s.dry_run is True
    assert s.z_threshold == 3.5
    assert s.sustained_z_threshold == 5.5
    assert s.baseline_weeks == 8
    assert s.min_consecutive_days == 2
    assert s.seed == 42


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("Z_THRESHOLD", "4.5")
    monkeypatch.setenv("PLAN_MODE", "llm")
    s = Settings(_env_file=None)
    assert s.z_threshold == 4.5
    assert s.plan_mode == "llm"


def test_invalid_values_are_rejected(monkeypatch):
    monkeypatch.setenv("AUTONOMY_LEVEL", "7")
    with pytest.raises(ValueError, match="autonomy_level"):
        Settings(_env_file=None)


def test_env_example_matches_the_settings_defaults():
    """`.env.example` é a documentação das opções; ele não pode divergir do código."""
    from_example = Settings(_env_file=ROOT / ".env.example")
    assert from_example == Settings(_env_file=None)
