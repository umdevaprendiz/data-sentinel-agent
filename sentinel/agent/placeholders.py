"""Placeholders `{f003.observed}`: o LLM aponta o campo, o código escreve o número.

O caminho começa pelo id do `Finding` e segue pelo `result` (`{f004.inside.change_pct}`,
`{f005.changes.0.description}`). Para ler os parâmetros da consulta, use o prefixo `params`
(`{f001.params.window_start}`). Só valores simples (número, texto, booleano) são aceitos.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from sentinel.schemas import Finding
from sentinel.security import clean_text

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_TOKEN = re.compile(r"^(f\d{3,})\.([A-Za-z_]\w*|\d+)(\.([A-Za-z_]\w*|\d+))*$")
METRICS = ("sessions", "conversions", "conversion_rate", "avg_ticket", "revenue")
MAX_LIST_ITEMS = 3  # no prompt, listas longas mostram só os primeiros itens


class PlaceholderError(ValueError):
    """Placeholder malformado ou que não aponta para um valor existente."""


def tokens(text: str) -> list[str]:
    """Conteúdo de cada par de chaves do texto, na ordem em que aparecem."""
    return PLACEHOLDER.findall(text)


def strip_placeholders(text: str) -> str:
    return PLACEHOLDER.sub(" ", text)


def finding_ids_in(text: str) -> list[str]:
    """Ids de `Finding` citados por placeholders bem formados."""
    return [t.split(".", 1)[0] for t in tokens(text) if _TOKEN.match(t)]


def _lookup(node: Any, parts: list[str], token: str) -> Any:
    for part in parts:
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise PlaceholderError(f"{{{token}}}: campo inexistente ({part!r})")
    return node


def _metric_of(parts: list[str], result: Mapping[str, Any]) -> str | None:
    for part in parts:
        if part in METRICS:
            return part
    metric = result.get("metric")
    return metric if metric in METRICS else None


def resolve(token: str, findings: Mapping[str, Finding]) -> str:
    """Valor formatado de um placeholder. Levanta `PlaceholderError` se não existir."""
    if not _TOKEN.match(token):
        raise PlaceholderError(f"{{{token}}}: placeholder malformado")
    finding_id, path = token.split(".", 1)
    finding = findings.get(finding_id)
    if finding is None:
        raise PlaceholderError(f"{{{token}}}: Finding {finding_id} não existe")
    parts = path.split(".")
    if parts[0] == "params":
        root: Any = finding.params
        parts = parts[1:]
        metric = finding.params.get("metric")
    else:
        root = finding.result
        metric = _metric_of(parts, finding.result)
    value = _lookup(root, parts, token)
    if value is None or isinstance(value, Mapping | list):
        raise PlaceholderError(f"{{{token}}}: não é um valor simples")
    return format_value(parts[-1] if parts else "", value, metric)


def fill(text: str, findings: Iterable[Finding]) -> str:
    """Troca cada placeholder pelo valor do `Finding`."""
    by_id = {f.id: f for f in findings}
    return PLACEHOLDER.sub(lambda m: resolve(m.group(1), by_id), text)


# --- Formatação (pt-BR) ---------------------------------------------------------------------


def _decimal(value: float, digits: int, sign: bool = False) -> str:
    text = f"{value:{'+' if sign else ''},.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def format_value(name: str, value: Any, metric: str | None = None) -> str:
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, str):
        return clean_text(value)  # texto do banco: sem controle de terminal, curto
    if isinstance(value, int):
        return _decimal(value, 0)
    if name.endswith("change_pct"):
        return _decimal(value * 100, 1, sign=True) + "%"
    if name.endswith("_share") or name == "threshold":
        return _decimal(value * 100, 0) + "%"
    if name == "z":
        return _decimal(value, 1)
    if metric == "conversion_rate" and name in ("observed", "expected"):
        return _decimal(value * 100, 2) + "%"
    if metric == "conversion_rate" and name == "change":
        return _decimal(value * 100, 2, sign=True) + " p.p."
    if abs(value) >= 100:
        return _decimal(value, 0)
    if abs(value) >= 1:
        return _decimal(value, 2)
    return _decimal(value, 4)


# --- Resumo para o prompt -------------------------------------------------------------------


def _walk(node: Any, path: str, metric: str | None, out: dict[str, str]) -> None:
    if isinstance(node, Mapping):
        for key, child in node.items():
            _walk(
                child, f"{path}.{key}" if path else str(key), key if key in METRICS else metric, out
            )
    elif isinstance(node, list):
        for index, child in enumerate(node[:MAX_LIST_ITEMS]):
            _walk(child, f"{path}.{index}", metric, out)
    elif node is not None:
        out[path] = format_value(path.rsplit(".", 1)[-1], node, metric)


def flatten(finding: Finding) -> dict[str, str]:
    """Caminho → valor formatado de cada campo que um placeholder pode citar."""
    out: dict[str, str] = {}
    _walk(finding.result, "", _metric_of([], finding.result), out)
    return out
