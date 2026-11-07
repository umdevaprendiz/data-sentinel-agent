"""Proteção contra vazamento: redação de segredos, logs seguros e erros sem dados.

Regra do projeto: nenhum segredo (senha de banco, token, chave) nem caminho pessoal pode aparecer
em log, traceback ou mensagem de erro. Tudo que sai para o usuário passa por aqui.
"""

import logging
import re
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.config import Settings

REDACTED = "[REDIGIDO]"
MAX_TEXT = 200
MIN_SECRET_LENGTH = 4  # segredos menores que isso viram ruído (trocariam trechos de palavras)

_SECRETS: set[str] = set()

_DSN_PASSWORD = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]*):([^\s@/]+)@")
_TELEGRAM_TOKEN = re.compile(r"(?:bot)?\d{6,12}:[A-Za-z0-9_-]{30,}")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_CLOUD_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL
)
_KEYED_VALUE = re.compile(
    r"(?i)\b(password|passwd|pwd|senha|token|secret|api[_-]?key)\b(\s*[=:]\s*)[^\s,;]+"
)
_PERSONAL_PATH = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\/\s]+|/(?:Users|home)/[^/\s]+")

# Senhas de exemplo que aparecem na documentação e não são segredo.
_PLACEHOLDER_PASSWORDS = {
    "senha",
    "sua_senha",
    "sua_nova_senha",
    "password",
    "pass",
    "xxx",
    "***",
    "redigido",
    "[redigido]",
}


def register_secrets(*values: str) -> None:
    """Valores literais que nunca podem sair (além dos padrões conhecidos)."""
    _SECRETS.update(v for v in values if len(v) >= MIN_SECRET_LENGTH)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Troca segredos conhecidos, padrões de credencial e a pasta pessoal por marcadores."""
    for value in sorted({*_SECRETS, *secrets}, key=len, reverse=True):
        if len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, REDACTED)
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _DSN_PASSWORD.sub(lambda m: f"{m.group(1)}:{REDACTED}@", text)
    for pattern in (_TELEGRAM_TOKEN, _GITHUB_TOKEN, _CLOUD_KEY, _API_KEY):
        text = pattern.sub(REDACTED, text)
    text = _KEYED_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    home = str(Path.home())
    if home:
        text = text.replace(home, "~").replace(home.replace("\\", "/"), "~")
    return text


def redact_obj(value: Any) -> Any:
    """`redact` em cada texto de uma estrutura JSON (chaves e valores), sem quebrar o JSON.

    Redigir o texto já serializado pode engolir uma aspa (`token=abc"`) e corromper o arquivo.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {redact(str(k)): redact_obj(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_obj(v) for v in value]
    return value


def clean_text(value: str, limit: int = MAX_TEXT) -> str:
    """Texto vindo do banco, seguro para prompt e terminal: sem controle, sem bidi, curto."""
    out = []
    for char in value:
        category = unicodedata.category(char)
        if category in ("Cc", "Zl", "Zp"):
            out.append(" ")
        elif category != "Cf":
            out.append(char)
    text = " ".join("".join(out).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def find_secrets(text: str) -> list[tuple[str, int]]:
    """(tipo, linha) de cada segredo ou dado pessoal no texto. Nunca devolve o valor."""
    found: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in _DSN_PASSWORD.finditer(line):
            password = match.group(2)
            is_template = password.startswith(("{", "<", "$", "%"))  # f-string, <senha>, $SENHA
            if not is_template and password.lower() not in _PLACEHOLDER_PASSWORDS:
                found.append(("connection_string_password", number))
        checks = {
            "telegram_token": _TELEGRAM_TOKEN,
            "github_token": _GITHUB_TOKEN,
            "cloud_key": _CLOUD_KEY,
            "api_key": _API_KEY,
            "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            "personal_path": _PERSONAL_PATH,
        }
        found += [(kind, number) for kind, pattern in checks.items() if pattern.search(line)]
    return found


# --- Logs -------------------------------------------------------------------------------------


class RedactingFormatter(logging.Formatter):
    """Aplica `redact` na mensagem e no traceback já formatados."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(settings: "Settings | None" = None, level: int = logging.WARNING) -> None:
    """Logs só em stderr, redigidos; clientes HTTP não registram URLs (o bot usa token nelas)."""
    if settings is not None:
        register_secrets(*settings.secret_values())
    root = logging.getLogger()
    if not any(getattr(h, "_sentinel", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._sentinel = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


# --- Erros para o usuário ---------------------------------------------------------------------


class UserError(Exception):
    """Erro esperado, com mensagem escrita por nós (sem dados): pode ser mostrada."""


def _family(error: BaseException) -> str:
    modules = {cls.__module__.split(".")[0] for cls in type(error).__mro__}
    if "psycopg" in modules:
        return "falha ao acessar o Postgres da memória"
    if "sqlite3" in modules or type(error).__name__ == "DatabaseError":  # inclui o do pandas
        return "falha ao ler o banco de dados"
    if "httpx" in modules:
        return "falha de rede com o serviço local"
    if modules & {"pydantic", "pydantic_core"}:
        return "configuração ou dado inválido"
    if isinstance(error, OSError):
        return "falha ao acessar um arquivo"
    return "erro inesperado"


def describe_error(error: BaseException) -> str:
    """Descrição sem a mensagem original: só a família e o tipo da exceção."""
    if isinstance(error, UserError):
        return redact(str(error))
    family = _family(error)
    text = f"{family[0].upper()}{family[1:]} ({type(error).__name__})"
    fields = _invalid_fields(error)
    return f"{text}; campos: {', '.join(fields)}" if fields else text


def _invalid_fields(error: BaseException) -> list[str]:
    """Nomes dos campos inválidos de um erro do Pydantic. Nunca os valores recebidos."""
    errors = getattr(error, "errors", None)
    if not callable(errors) or "pydantic" not in type(error).__module__:
        return []
    try:
        names = {".".join(str(part) for part in item["loc"]) for item in errors()}
    except Exception:  # um erro ao descrever o erro não pode piorar a situação
        return []
    return sorted(n for n in names if n)[:5]
