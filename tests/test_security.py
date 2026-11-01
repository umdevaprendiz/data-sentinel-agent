import logging
import sqlite3
from pathlib import Path

import pytest

from sentinel import cli, security
from sentinel.config import Settings
from sentinel.security import (
    REDACTED,
    RedactingFormatter,
    UserError,
    clean_text,
    configure_logging,
    describe_error,
    find_secrets,
    redact,
    register_secrets,
)

# Os valores são montados em tempo de execução: o teste não pode conter segredo literal.
PASSWORD = "hunter" + "2x"
DSN = "postgresql://postgres:" + PASSWORD + "@localhost:5432/sentinel"
BOT_TOKEN = "123456789:" + "AbCdEfGhIj" * 4


@pytest.fixture(autouse=True)
def clean_registry():
    security._SECRETS.clear()
    yield
    security._SECRETS.clear()


# --- redact -----------------------------------------------------------------------------------


def test_redact_hides_the_password_of_a_connection_string():
    text = redact(f"connection to {DSN} failed")
    assert PASSWORD not in text
    assert f"postgres:{REDACTED}@localhost:5432/sentinel" in text


def test_redact_hides_telegram_tokens_even_inside_a_url():
    text = redact(f"POST https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
    assert BOT_TOKEN not in text and BOT_TOKEN.split(":")[1] not in text
    assert "api.telegram.org" in text


@pytest.mark.parametrize(
    "text",
    ["password=" + PASSWORD, "senha: " + PASSWORD, "API_KEY = " + PASSWORD, "token=" + PASSWORD],
)
def test_redact_hides_values_after_secret_looking_keys(text):
    assert PASSWORD not in redact(text)


def test_redact_hides_known_provider_keys_and_private_keys():
    github = "ghp_" + "a1B2c3D4e5" * 3
    cloud = "AKIA" + "ABCDEFGHIJKLMNOP"
    pem = "-----BEGIN " + "PRIVATE KEY-----\nabc\n-----END " + "PRIVATE KEY-----"
    text = redact(f"{github} {cloud} {pem}")
    assert github not in text and cloud not in text and "abc" not in text


def test_redact_hides_registered_secrets_wherever_they_appear():
    register_secrets("valor-sem-formato-conhecido")
    assert "valor-sem-formato-conhecido" not in redact("falhou com valor-sem-formato-conhecido!")


def test_redact_ignores_registered_secrets_that_are_too_short_to_be_safe():
    register_secrets("ab", "")
    assert redact("abacate") == "abacate"


def test_redact_hides_the_home_directory():
    home = str(Path.home())
    assert home not in redact(f"arquivo em {home}\\projeto\\x.db")
    assert "~" in redact(f"arquivo em {home}")


def test_redact_leaves_harmless_text_alone():
    text = "A taxa caiu -14,4% no android 3.2 (z = -8,2)."
    assert redact(text) == text


# --- clean_text -------------------------------------------------------------------------------


def test_clean_text_removes_terminal_control_characters_and_bidi_tricks():
    hostile = "ok\x1b[2J\x07\nlinha\u202edois\u200b"
    cleaned = clean_text(hostile)
    assert "\x1b" not in cleaned and "\x07" not in cleaned and "\n" not in cleaned
    assert "\u202e" not in cleaned and "\u200b" not in cleaned
    assert cleaned.startswith("ok") and "linha" in cleaned


def test_clean_text_limits_the_length_and_collapses_spaces():
    assert clean_text("a   b\t\tc") == "a b c"
    long = clean_text("x" * 500)
    assert len(long) == security.MAX_TEXT and long.endswith("…")


# --- find_secrets (usado pelo teste de higiene) -----------------------------------------------


def test_find_secrets_reports_kind_and_line_but_never_the_value():
    text = f"ok\nlinha {DSN}\nfim\n{BOT_TOKEN}"
    found = find_secrets(text)
    assert ("connection_string_password", 2) in found
    assert ("telegram_token", 4) in found
    assert all(PASSWORD not in repr(item) for item in found)


def test_find_secrets_accepts_documented_placeholders():
    example = "postgresql://USUARIO:" + "SENHA" + "@localhost:5432/sentinel"
    assert find_secrets(example) == []
    for template in ("{password}", "<senha>", "$SENHA", "%(pw)s"):
        assert find_secrets("postgresql://postgres:" + template + "@localhost/db") == []
    assert find_secrets("postgresql://postgres@localhost/sentinel") == []


def test_find_secrets_flags_personal_paths():
    assert find_secrets("C:\\Users\\" + "Alguem" + "\\Desktop\\x") == [("personal_path", 1)]


# --- logs -------------------------------------------------------------------------------------


def test_the_log_formatter_redacts_messages_and_tracebacks():
    formatter = RedactingFormatter("%(levelname)s %(message)s")
    try:
        raise RuntimeError(f"não conectou em {DSN}")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "falha com %s", (BOT_TOKEN,), sys.exc_info()
        )
    output = formatter.format(record)
    assert PASSWORD not in output and BOT_TOKEN not in output
    assert "Traceback" in output and REDACTED in output


def test_configure_logging_is_idempotent_and_silences_http_clients():
    root = logging.getLogger()
    before = len(root.handlers)
    configure_logging()
    configure_logging()
    assert len(root.handlers) == before + 1
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    for handler in [h for h in root.handlers if getattr(h, "_sentinel", False)]:
        assert isinstance(handler.formatter, RedactingFormatter)
        root.removeHandler(handler)


def test_configure_logging_registers_the_secrets_of_the_settings():
    settings = Settings(_env_file=None, memory_dsn=DSN, telegram_bot_token=BOT_TOKEN)
    configure_logging(settings)
    assert PASSWORD not in redact("senha da vez: " + PASSWORD)
    assert BOT_TOKEN not in redact(BOT_TOKEN)
    for handler in [h for h in logging.getLogger().handlers if getattr(h, "_sentinel", False)]:
        logging.getLogger().removeHandler(handler)


# --- mensagens de erro ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            sqlite3.OperationalError("unable to open database file C:/segredo/x.db"),
            "banco de dados",
        ),
        (ValueError(f"dsn inválido {DSN}"), "inesperado"),
        (OSError("acesso negado a /ho" + "me/fulano/.ssh"), "arquivo"),
        (KeyError("tokenzinho"), "inesperado"),
    ],
)
def test_unexpected_errors_are_described_without_their_message(error, expected):
    text = describe_error(error)
    assert expected in text
    assert type(error).__name__ in text
    assert PASSWORD not in text and "segredo" not in text and "fulano" not in text
    assert "tokenzinho" not in text


def test_user_errors_show_their_own_message_but_redacted():
    text = describe_error(UserError(f"Não conectei em {DSN}"))
    assert "Não conectei" in text and PASSWORD not in text


def test_psycopg_and_httpx_errors_are_recognised_by_family():
    import httpx
    import psycopg

    assert "Postgres" in describe_error(psycopg.OperationalError(f"falhou {DSN}"))
    assert "rede" in describe_error(httpx.ConnectError(f"recusou {DSN}"))


# --- ponto de entrada da CLI ------------------------------------------------------------------


def run_failing_cli(monkeypatch, error: BaseException):
    def failing():
        raise error

    monkeypatch.setattr(cli, "app", failing)
    with pytest.raises(SystemExit) as stop:
        cli.run()
    return stop.value.code


def test_cli_hides_traceback_and_message_of_unexpected_errors(monkeypatch, capsys):
    monkeypatch.delenv("SENTINEL_DEBUG", raising=False)
    secret_local = "variavel_secreta_" + PASSWORD  # noqa: F841
    code = run_failing_cli(monkeypatch, RuntimeError(f"quebrou em {DSN}"))
    err = capsys.readouterr().err
    assert code == 1
    assert "Erro inesperado (RuntimeError)" in err
    assert "SENTINEL_DEBUG" in err
    assert "Traceback" not in err and PASSWORD not in err and "variavel_secreta" not in err


def test_cli_debug_mode_shows_a_redacted_traceback_without_local_variables(monkeypatch, capsys):
    monkeypatch.setenv("SENTINEL_DEBUG", "1")
    leaked_local = "valor_local_" + PASSWORD  # noqa: F841
    code = run_failing_cli(monkeypatch, RuntimeError(f"quebrou em {DSN}"))
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" in err and "RuntimeError" in err
    assert PASSWORD not in err and "valor_local_" not in err


def test_cli_keyboard_interrupt_and_system_exit_are_not_errors(monkeypatch, capsys):
    assert run_failing_cli(monkeypatch, KeyboardInterrupt()) == 130
    assert "Interrompido" in capsys.readouterr().err
    assert run_failing_cli(monkeypatch, SystemExit(0)) == 0


def test_typer_is_not_allowed_to_print_local_variables():
    assert cli.app.pretty_exceptions_show_locals is False
    assert cli.app.pretty_exceptions_enable is False


def test_invalid_settings_name_the_field_but_never_the_value(monkeypatch):
    wrong = "valor" + "-que-nao-pode-vazar"
    monkeypatch.setenv("AUTONOMY_LEVEL", wrong)
    with pytest.raises(ValueError) as raised:
        Settings(_env_file=None)
    text = describe_error(raised.value)
    assert "autonomy_level" in text
    assert wrong not in text


def test_a_missing_database_message_does_not_expose_the_home_directory():
    from typer.testing import CliRunner

    missing = Path.home() / "pasta_que_nao_existe" / "x.db"
    result = CliRunner().invoke(cli.app, ["detect", "--db-path", str(missing)])
    assert result.exit_code != 0
    assert str(Path.home()) not in result.output
    flat = " ".join(result.output.replace("│", " ").split())
    assert "não existe" in flat


def test_a_corrupt_database_is_reported_as_a_database_error(tmp_path):
    from sentinel.data.db import connect_readonly
    from sentinel.detection.detector import load_daily_series

    broken = tmp_path / "quebrado.db"
    broken.write_text("isto não é um banco de dados", encoding="utf-8")
    con = connect_readonly(broken)
    with pytest.raises(Exception) as raised:
        load_daily_series(con)
    con.close()
    assert "banco de dados" in describe_error(raised.value)
