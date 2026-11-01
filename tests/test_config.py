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


def test_secrets_never_show_up_in_repr_or_dumps(monkeypatch):
    password = "pw" + "7z7"
    token = "123456789:" + "AbCdEfGhIj" * 4
    monkeypatch.setenv("MEMORY_DSN", f"postgresql://postgres:{password}@localhost:5432/sentinel")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    s = Settings(_env_file=None)
    for text in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        assert password not in text and token not in text
    assert s.memory_dsn.get_secret_value().endswith("/sentinel")
    assert {password, token} <= set(s.secret_values())


def test_secret_values_with_no_secrets_configured_is_empty():
    assert Settings(_env_file=None).secret_values() == []
