"""Menu fechado de ferramentas de investigação (seção 8 do CLAUDE.md).

Cada ferramenta é somente leitura, valida os parâmetros e devolve um `Finding`. O agente (código
ou LLM) só enxerga os dados por aqui. Pedido inválido não derruba ninguém: vira `Finding` de erro.
Métricas e dimensões válidas vêm do perfil descoberto nos dados (decisão 21).
"""

import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sentinel.config import Settings, get_settings
from sentinel.data.profile import (
    ProfileError,
    SchemaProfile,
    load_events,
    profile_from_frame,
    propose_dimension,
    propose_ratio,
)
from sentinel.investigation.cube import Cube
from sentinel.investigation.drilldown import (
    DrillParams,
    SliceTable,
    Window,
    build_table,
    dims_table,
    evaluate_hypothesis,
)
from sentinel.llm.embeddings import EmbeddingError
from sentinel.memory.incidents import IncidentStore, MemoryStoreError, signature_text
from sentinel.schemas import NAME_PATTERN, Finding
from sentinel.security import clean_text


class ToolError(ValueError):
    """Pedido que a ferramenta recusa (janela fora dos dados, escopo inválido…)."""


def _num(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else round(value, digits)


# --- Parâmetros -----------------------------------------------------------------------------

Name = str  # métrica ou dimensão: o formato é livre aqui e o perfil confere em `check`


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def check(self, profile: SchemaProfile) -> None:
        """Confere métricas e dimensões contra o perfil dos dados (o erro cita o campo)."""
        for field in type(self).model_fields:
            value = getattr(self, field)
            if value is None:
                continue
            try:
                if field == "metric":
                    profile.check_metric(value)
                elif field == "dimension":
                    profile.check_dimension(value)
                elif field == "dimensions" and isinstance(value, list):
                    for name in value:
                        profile.check_dimension(name)
                elif field in ("dimensions", "together_with", "scope"):
                    profile.check_dimensions(dict(value))
            except ProfileError as error:
                raise ToolError(f"{field}: {error}") from None


class WindowParams(_Params):
    window_start: date
    window_end: date
    baseline_weeks: int | None = Field(default=None, ge=4, le=12)


class MetricWindowParams(WindowParams):
    metric: Name


class SplitByParams(MetricWindowParams):
    dimension: Name


class TopContributorsParams(MetricWindowParams):
    dimensions: list[Name] = Field(min_length=1, max_length=2)
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("dimensions")
    @classmethod
    def _distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("dimensões repetidas")
        return value


class TestHypothesisParams(MetricWindowParams):
    __test__ = False  # nome não é uma classe de teste do pytest

    dimensions: dict[Name, str] = Field(min_length=1, max_length=2)
    together_with: dict[Name, str] | None = Field(default=None, min_length=1, max_length=2)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.together_with == self.dimensions:
            raise ValueError("together_with igual a dimensions")
        return self


class DecomposeParams(WindowParams):
    metric: Name | None = None  # medida aditiva a decompor; padrão: a mais derivada (receita)


class ChangesParams(WindowParams):
    scope: dict[Name, str] | None = None
    lookback_days: int = Field(default=7, ge=0, le=30)


class DefineMetricParams(_Params):
    name: str = Field(pattern=NAME_PATTERN)
    numerator: Name
    denominator: Name


class AddDimensionParams(_Params):
    column: str = Field(pattern=NAME_PATTERN)


class RunSqlParams(_Params):
    query: str = Field(min_length=1, max_length=2000)


class SimilarIncidentsParams(_Params):
    metric: Name
    direction: Literal["drop", "spike"]
    dimensions: dict[Name, str] | None = None
    k: int | None = Field(default=None, ge=1, le=10)


# --- Contexto -------------------------------------------------------------------------------


class ToolContext:
    """Conexão somente leitura, perfil e cubo em memória e numeração dos `Finding`s."""

    def __init__(
        self,
        con: sqlite3.Connection,
        settings: Settings | None = None,
        profile: SchemaProfile | None = None,
        memory: IncidentStore | None = None,
    ) -> None:
        self.con = con
        self.settings = settings or get_settings()
        self.findings: list[Finding] = []
        self.memory = memory  # para `find_similar_incidents`; None = sem memória
        # Chamado a cada Finding com a duração em ms (o grafo grava no audit).
        self.listener: Callable[[Finding, float], None] | None = None
        self._started = 0.0
        self._frame: pd.DataFrame | None = None
        self._profile = profile
        self._cube: Cube | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = load_events(self.con)
        return self._frame

    @property
    def profile(self) -> SchemaProfile:
        if self._profile is None:
            self._profile = profile_from_frame(self.frame)
        return self._profile

    @property
    def cube(self) -> Cube:
        if self._cube is None:
            self._cube = Cube.from_frame(self.frame, self.profile)
        return self._cube

    def update_profile(self, profile: SchemaProfile) -> None:
        """Novo perfil (métrica ou dimensão validada): o cubo acompanha."""
        dims_changed = self._profile is None or profile.dimensions != self._profile.dimensions
        self._profile = profile
        if self._cube is not None:
            if dims_changed:
                self._cube = Cube.from_frame(self.frame, profile)
            else:
                self._cube.profile = profile

    def drill_params(self, baseline_weeks: int | None = None) -> DrillParams:
        params = DrillParams.from_settings(self.settings)
        if baseline_weeks is None:
            return params
        return DrillParams(**{**params.__dict__, "baseline_weeks": baseline_weeks})

    def window(self, params: WindowParams) -> Window:
        if params.window_end < params.window_start:
            raise ToolError("window_end antes de window_start")
        try:
            self.cube.pos(params.window_start)
            self.cube.pos(params.window_end)
        except ValueError as error:
            raise ToolError(str(error)) from error
        return Window(params.window_start, params.window_end)

    def finding(self, tool: str, params: dict, result: dict) -> Finding:
        finding = Finding(
            id=f"f{len(self.findings) + 1:03d}", tool=tool, params=params, result=result
        )
        self.findings.append(finding)
        if self.listener is not None:
            self.listener(finding, (time.perf_counter() - self._started) * 1000)
        return finding

    def run(self, tool: str, raw_params: dict) -> Finding:
        """Executa uma ferramenta do menu. Erros viram `Finding` com a chave `error`."""
        self._started = time.perf_counter()
        spec = TOOLS.get(tool)
        if spec is None:
            name = clean_text(str(tool), 40)
            return self.finding(
                name, _safe(raw_params), {"error": f"ferramenta desconhecida: {name}"}
            )
        try:
            params = spec.params.model_validate(raw_params)
            params.check(self.profile)
            result = spec.run(self, params)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
            )
            message = clean_text(f"parâmetros inválidos: {problems}", 400)
            return self.finding(tool, _safe(raw_params), {"error": message})
        except ValueError as error:  # inclui ToolError e ProfileError
            return self.finding(tool, _safe(raw_params), {"error": clean_text(str(error), 400)})
        except (EmbeddingError, MemoryStoreError):
            return self.finding(tool, _safe(raw_params), {"error": "memória indisponível"})
        return self.finding(tool, params.model_dump(mode="json"), result)


def _safe(raw: Any) -> dict:
    """Parâmetros recusados, guardados de forma que o Finding continue sendo JSON."""
    if not isinstance(raw, dict):
        return {"raw": clean_text(str(raw), 200)}
    try:
        json.dumps(raw)
    except (TypeError, ValueError):
        return {"raw": clean_text(str(raw), 200)}
    return raw


# --- Ferramentas ----------------------------------------------------------------------------


def _slice_row(table: SliceTable, index: int, with_dimensions: dict[str, str]) -> dict:
    return {
        "dimensions": with_dimensions,
        "observed": _num(table.obs[index]),
        "expected": _num(table.exp[index]),
        "change_pct": _num(table.change_pct[index]),
        "impact": _num(table.impact[index]),
        "explained_share": _num(table.ep[index]),
        "z": _num(table.z[index]),
        "eligible": bool(table.eligible[index]),
    }


def _total_table(ctx: ToolContext, metric: str, params: WindowParams) -> SliceTable:
    window, drill = ctx.window(params), ctx.drill_params(params.baseline_weeks)
    cube = ctx.cube
    return build_table(
        cube, metric, np.zeros(cube.n_rows, dtype=np.int64), [("total",)], window, drill
    )


def compare_periods(ctx: ToolContext, p: MetricWindowParams) -> dict:
    table = _total_table(ctx, p.metric, p)
    return {
        "metric": p.metric,
        "observed": _num(table.total_obs),
        "expected": _num(table.total_exp),
        "change": _num(table.total_change),
        "change_pct": _num(table.change_pct[0]),
        "z": _num(table.z[0]),
        "direction": "drop" if table.total_change < 0 else "spike",
    }


def split_by(ctx: ToolContext, p: SplitByParams) -> dict:
    window, drill = ctx.window(p), ctx.drill_params(p.baseline_weeks)
    table = dims_table(ctx.cube, p.metric, (p.dimension,), window, drill)
    order = sorted(
        (i for i in range(len(table.labels)) if table.nonempty[i]),
        key=lambda i: -abs(table.impact[i]),
    )
    return {
        "metric": p.metric,
        "dimension": p.dimension,
        "total_change": _num(table.total_change),
        "slices": [_slice_row(table, i, {p.dimension: table.labels[i][0]}) for i in order],
    }


def top_contributors(ctx: ToolContext, p: TopContributorsParams) -> dict:
    window, drill = ctx.window(p), ctx.drill_params(p.baseline_weeks)
    table = dims_table(ctx.cube, p.metric, tuple(p.dimensions), window, drill)
    ranked = sorted(
        (i for i in range(len(table.labels)) if table.nonempty[i]),
        key=lambda i: -np.nan_to_num(table.ep[i], nan=-np.inf),
    )[: p.k]
    return {
        "metric": p.metric,
        "dimensions": list(p.dimensions),
        "total_change": _num(table.total_change),
        "slices": [
            _slice_row(table, i, dict(zip(p.dimensions, table.labels[i], strict=True)))
            for i in ranked
        ],
    }


def test_hypothesis(ctx: ToolContext, p: TestHypothesisParams) -> dict:
    window, drill = ctx.window(p), ctx.drill_params(p.baseline_weeks)
    together = None if p.together_with is None else dict(p.together_with)
    evaluation = evaluate_hypothesis(
        ctx.cube, p.metric, window, drill, dict(p.dimensions), together
    )

    def stats(s) -> dict:
        return {
            "observed": _num(s.observed),
            "expected": _num(s.expected),
            "change_pct": _num(s.change_pct),
            "z": _num(s.z),
        }

    joint = (
        {
            "together_with": together,
            "members": [
                {"dimensions": m.dimensions, "explained_share": _num(m.explained_share)}
                for m in evaluation.members
            ],
        }
        if together is not None
        else {}
    )
    return {
        "metric": p.metric,
        "dimensions": dict(p.dimensions),
        **joint,
        "explained_share": _num(evaluation.explained_share),
        "threshold": evaluation.threshold,
        "others_stable": evaluation.others_stable,
        "eligible": evaluation.eligible,
        "total_significant": evaluation.total_significant,
        "accepted": evaluation.accepted,
        "inside": stats(evaluation.inside),
        "outside": stats(evaluation.outside),
        "refinements": [
            {"dimensions": c.dimensions, "explained_share": _num(c.explained_share)}
            for c in evaluation.refinements
        ],
    }


def decompose_metric(ctx: ToolContext, p: DecomposeParams) -> dict:
    """Qual fator se moveu numa medida derivada (receita = sessões × taxa × ticket, em log)."""
    profile = ctx.profile
    target = p.metric or profile.deepest_measure
    if target not in profile.measures:
        raise ToolError(f"metric: {target} não é uma medida aditiva")
    factors_names = profile.chain(target)
    if len(factors_names) < 2:
        raise ToolError(f"metric: {target} não se decompõe em fatores (não há razões até ela)")
    tables = {m: _total_table(ctx, m, p) for m in [target, *factors_names]}
    total = tables[target]
    total_log = float(np.log(total.total_obs / total.total_exp))
    factors = {}
    for name in factors_names:
        t = tables[name]
        ratio = t.total_obs / t.total_exp
        log_change = float(np.log(ratio))
        factors[name] = {
            "observed": _num(t.total_obs),
            "expected": _num(t.total_exp),
            "change_pct": _num(ratio - 1),
            "log_share": _num(log_change / total_log) if abs(total_log) > 1e-9 else None,
        }
    return {
        "target": target,
        target: {
            "observed": _num(total.total_obs),
            "expected": _num(total.total_exp),
            "change_pct": _num(total.total_obs / total.total_exp - 1),
        },
        "factors": factors,
        "dominant_factor": max(factors, key=lambda f: abs(factors[f]["log_share"] or 0.0)),
    }


def _scope_overlaps(change_scope: dict, scope: dict) -> bool:
    """A mudança fala da mesma fatia: escopo global, ou dimensões em comum com os mesmos valores.

    Escopos sem nenhuma dimensão em comum (ex.: região × produto) não contam: poderiam se cruzar,
    mas nada indica que a mudança tenha relação com a hipótese.
    """
    if not change_scope:
        return True
    shared = change_scope.keys() & scope.keys()
    return bool(shared) and all(scope[k] == change_scope[k] for k in shared)


def _scope_of(raw: str) -> dict[str, str]:
    """Escopo gravado em `changes` (JSON). Texto do banco: limpo e com tamanho limitado."""
    try:
        scope = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(scope, dict):
        return {}
    return {clean_text(str(k), 40): clean_text(str(v), 80) for k, v in scope.items()}


def check_recent_changes(ctx: ToolContext, p: ChangesParams) -> dict:
    window = ctx.window(p)
    first = window.start - timedelta(days=p.lookback_days)
    rows = ctx.con.execute(
        "SELECT ts, type, description, scope FROM changes WHERE ts >= ? AND ts <= ? ORDER BY ts",
        (first.isoformat(), window.end.isoformat()),
    ).fetchall()
    changes = []
    for ts, kind, description, scope_json in rows:
        scope = _scope_of(scope_json)
        if p.scope is not None and not _scope_overlaps(scope, dict(p.scope)):
            continue
        changes.append(
            {
                "ts": ts,
                "type": kind,
                "description": description,
                "scope": scope,
                "days_from_window_start": (date.fromisoformat(ts) - window.start).days,
            }
        )
    return {
        "lookback_days": p.lookback_days,
        "scope": p.scope,
        "count": len(changes),
        "changes": changes,
    }


def check_data_quality(ctx: ToolContext, p: WindowParams) -> dict:
    window, drill = ctx.window(p), ctx.drill_params(p.baseline_weeks)
    cube = ctx.cube
    first, last = cube.pos(window.start), cube.pos(window.end)
    volume = cube.profile.volume

    in_window = (cube.day_idx >= first) & (cube.day_idx <= last)
    rows_per_day = np.bincount(cube.day_idx, minlength=cube.n_days)
    typical = int(np.median(rows_per_day[rows_per_day > 0]))
    zero_slices = []
    for name in cube.dims:
        for metric in cube.profile.count_measures:
            table = dims_table(cube, metric, (name,), window, drill)
            for i, (value,) in enumerate(table.labels):
                if table.nonempty[i] and table.obs[i] == 0 and table.exp[i] > 0:
                    zero_slices.append({"dimension": name, "value": value, "metric": metric})
    return {
        "missing_days": [
            d.isoformat() for d in cube.missing_days if window.start <= d <= window.end
        ],
        "days_with_fewer_rows": [
            (cube.start + timedelta(days=int(d))).isoformat()
            for d in range(first, last + 1)
            if 0 < rows_per_day[d] < typical
        ],
        "duplicate_rows": cube.duplicate_rows,
        "null_values": cube.null_values,
        "zero_session_rows": int((cube.values[volume][in_window] == 0).sum()),
        "zeroed_slices": zero_slices,
    }


# --- Perfil: o LLM propõe, o código valida (decisão 21) ---------------------------------------


def define_metric(ctx: ToolContext, p: DefineMetricParams) -> dict:
    """Nova razão entre duas medidas. Só entra no perfil se o código validar."""
    try:
        profile = propose_ratio(ctx.profile, ctx.frame, p.name, p.numerator, p.denominator)
    except ProfileError as error:
        raise ToolError(str(error)) from None
    ctx.update_profile(profile)
    ratio = profile.ratios[p.name]
    return {
        "metric": p.name,
        "numerator": ratio.numerator,
        "denominator": ratio.denominator,
        "is_rate": ratio.is_rate,
        "accepted": True,
        "metrics": list(profile.metrics),
    }


def add_dimension(ctx: ToolContext, p: AddDimensionParams) -> dict:
    """Coluna de texto que ficou de fora só pela cardinalidade passa a ser dimensão."""
    try:
        profile = propose_dimension(ctx.profile, ctx.frame, p.column)
    except ProfileError as error:
        raise ToolError(str(error)) from None
    ctx.update_profile(profile)
    return {
        "dimension": p.column,
        "values": len(profile.dimensions[p.column]),
        "accepted": True,
    }


# --- SQL somente leitura (etapa 6) -------------------------------------------------------------

SQL_TABLES = frozenset({"events", "changes"})
_SQL_STRINGS = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_SQL_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_FORBIDDEN = re.compile(
    r"\b(pragma|attach|detach|insert|update|delete|drop|create|alter|replace|vacuum|reindex|"
    r"analyze|begin|commit|rollback|savepoint|release|load_extension)\b",
    re.IGNORECASE,
)
_SQL_DENIED_FUNCTIONS = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
)
MAX_SQL_CELL = 120


def _sql_authorizer(
    action: int, arg1: str | None, arg2: str | None, database: str | None, *_: object
) -> int:
    """Só leitura das tabelas do menu e funções inofensivas; qualquer outra ação é negada.

    Leitura de CTE ou subconsulta chega sem banco (`None`); tabela de verdade chega com `main`.
    """
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if database is None:
            return sqlite3.SQLITE_OK
        allowed = database == "main" and arg1 in SQL_TABLES
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        name = (arg2 or "").lower()
        return sqlite3.SQLITE_DENY if name in _SQL_DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    if action == getattr(sqlite3, "SQLITE_RECURSIVE", 33):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def check_sql(query: str) -> str:
    """Uma instrução, só `SELECT`/`WITH`, sem `PRAGMA`/`ATTACH` nem escrita."""
    text = _SQL_COMMENTS.sub(" ", query).strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    bare = _SQL_STRINGS.sub("''", text)
    if ";" in bare:
        raise ToolError("query: só uma instrução por consulta")
    words = bare.split(None, 1)
    if not words or words[0].lower() not in ("select", "with"):
        raise ToolError("query: só SELECT ou WITH")
    found = _SQL_FORBIDDEN.search(bare)
    if found:
        raise ToolError(f"query: palavra proibida ({found.group(0).lower()})")
    return text


def _cell(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value, MAX_SQL_CELL)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, float):
        return _num(value)
    return value


def run_sql(ctx: ToolContext, p: RunSqlParams) -> dict:
    query = check_sql(p.query)
    con = ctx.con
    max_rows = ctx.settings.sql_max_rows
    deadline = time.monotonic() + ctx.settings.sql_timeout_s
    con.set_authorizer(_sql_authorizer)
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        cursor = con.execute(query)
        columns = [clean_text(str(d[0]), 60) for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as error:
        message = str(error).lower()
        if "interrupt" in message:
            raise ToolError(f"query: tempo esgotado ({ctx.settings.sql_timeout_s:g} s)") from None
        if "not authorized" in message or "prohibited" in message:
            raise ToolError(
                "query: consulta não autorizada (só leitura de events e changes)"
            ) from None
        raise ToolError(f"query: SQL inválido ({clean_text(str(error), 160)})") from None
    finally:
        con.set_authorizer(None)
        con.set_progress_handler(None, 0)
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {
        "columns": columns,
        "rows": [[_cell(v) for v in row] for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


# --- Memória (etapa 6) --------------------------------------------------------------------------


def find_similar_incidents(ctx: ToolContext, p: SimilarIncidentsParams) -> dict:
    """Incidentes passados parecidos e o que foi decidido (sem o texto dos relatórios)."""
    if ctx.memory is None:
        return {"count": 0, "incidents": [], "available": False}
    causes = [dict(p.dimensions)] if p.dimensions else []
    text = signature_text(p.metric, p.direction, causes)
    hits = ctx.memory.similar(
        text, p.k or ctx.settings.memory_top_k, ctx.settings.memory_min_similarity
    )
    return {
        "count": len(hits),
        "available": True,
        "incidents": [
            {
                "incident_id": hit.incident.incident_id,
                "signature": hit.incident.signature,
                "status": hit.incident.status,
                "state": hit.incident.state,
                "similarity": _num(hit.similarity, 3),
            }
            for hit in hits
        ],
    }


# --- Menu -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: type[_Params]
    run: Callable[[ToolContext, Any], dict]


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "compare_periods",
            "Valor observado, esperado e variação de uma métrica na janela.",
            MetricWindowParams,
            compare_periods,
        ),
        ToolSpec(
            "split_by",
            "Quebra a variação de uma métrica pelos valores de uma dimensão.",
            SplitByParams,
            split_by,
        ),
        ToolSpec(
            "top_contributors",
            "Fatias (1 ou 2 dimensões) que mais explicam a variação de uma métrica.",
            TopContributorsParams,
            top_contributors,
        ),
        ToolSpec(
            "test_hypothesis",
            "Mede quanto uma hipótese (dimensões = valores) explica e se as outras fatias estão "
            "estáveis. Com together_with, testa duas causas independentes de uma vez.",
            TestHypothesisParams,
            test_hypothesis,
        ),
        ToolSpec(
            "decompose_metric",
            "Qual fator se moveu: sessões, taxa de conversão ou ticket médio.",
            DecomposeParams,
            decompose_metric,
        ),
        ToolSpec(
            "check_recent_changes",
            "Mudanças registradas (deploy, preço, campanha, estoque) perto da janela.",
            ChangesParams,
            check_recent_changes,
        ),
        ToolSpec(
            "check_data_quality",
            "Dados faltando, duplicados ou zerados na janela.",
            WindowParams,
            check_data_quality,
        ),
        ToolSpec(
            "find_similar_incidents",
            "Incidentes passados parecidos (métrica, direção, fatia) e o que foi decidido.",
            SimilarIncidentsParams,
            find_similar_incidents,
        ),
        ToolSpec(
            "run_sql",
            "Um SELECT somente leitura sobre as tabelas events e changes (linhas limitadas).",
            RunSqlParams,
            run_sql,
        ),
        ToolSpec(
            "define_metric",
            "Propõe uma métrica nova: razão numerator / denominator entre duas medidas.",
            DefineMetricParams,
            define_metric,
        ),
        ToolSpec(
            "add_dimension",
            "Propõe uma coluna de texto que ficou de fora do perfil como dimensão.",
            AddDimensionParams,
            add_dimension,
        ),
    )
}


def _with_enums(schema: dict, profile: SchemaProfile) -> dict:
    """JSON Schema dos parâmetros com as métricas e dimensões do perfil como `enum`."""
    properties = schema.get("properties", {})
    metrics, dims = list(profile.metrics), list(profile.dimensions)

    def enum_of(node: dict, values: list[str]) -> None:
        for option in node.get("anyOf", [node]):
            if option.get("type") == "string":
                option["enum"] = values
            elif option.get("type") == "array":
                option.setdefault("items", {})["enum"] = values
            elif option.get("type") == "object":
                option["propertyNames"] = {"enum": dims}

    for name, node in properties.items():
        if name == "metric":
            enum_of(node, metrics)
        elif name in ("dimension", "dimensions", "together_with", "scope"):
            enum_of(node, dims)
    return schema


def tool_catalog(profile: SchemaProfile | None = None) -> list[dict]:
    """Nome, descrição e JSON Schema de cada ferramenta (o que o LLM vê no modo `llm`)."""
    catalog = []
    for spec in TOOLS.values():
        schema = spec.params.model_json_schema()
        if profile is not None:
            schema = _with_enums(schema, profile)
        catalog.append({"name": spec.name, "description": spec.description, "parameters": schema})
    return catalog
