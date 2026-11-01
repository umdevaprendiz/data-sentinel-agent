"""Menu fechado de ferramentas de investigação (seção 8 do CLAUDE.md).

Cada ferramenta é somente leitura, valida os parâmetros e devolve um `Finding`. O agente (código
ou LLM) só enxerga os dados por aqui. Pedido inválido não derruba ninguém: vira `Finding` de erro.
"""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sentinel.config import Settings, get_settings
from sentinel.investigation.cube import Cube
from sentinel.investigation.drilldown import (
    DrillParams,
    SliceTable,
    Window,
    build_table,
    dims_table,
    evaluate_hypothesis,
)
from sentinel.schemas import DIMENSION_VALUES, Dimension, Finding, Metric


class ToolError(ValueError):
    """Pedido que a ferramenta recusa (janela fora dos dados, escopo inválido…)."""


def _num(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else round(value, digits)


def _check_dimensions(dimensions: dict[str, str]) -> dict[str, str]:
    for name, value in dimensions.items():
        if name not in DIMENSION_VALUES:
            raise ValueError(f"dimensão desconhecida: {name}")
        if value not in DIMENSION_VALUES[name]:
            raise ValueError(f"valor inválido para {name}: {value}")
    return dimensions


# --- Parâmetros -----------------------------------------------------------------------------


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WindowParams(_Params):
    window_start: date
    window_end: date
    baseline_weeks: int | None = Field(default=None, ge=4, le=12)


class MetricWindowParams(WindowParams):
    metric: Metric


class SplitByParams(MetricWindowParams):
    dimension: Dimension


class TopContributorsParams(MetricWindowParams):
    dimensions: list[Dimension] = Field(min_length=1, max_length=2)
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("dimensions")
    @classmethod
    def _distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("dimensões repetidas")
        return value


class TestHypothesisParams(MetricWindowParams):
    __test__ = False  # nome não é uma classe de teste do pytest

    dimensions: dict[Dimension, str] = Field(min_length=1, max_length=2)
    together_with: dict[Dimension, str] | None = Field(default=None, min_length=1, max_length=2)

    @field_validator("dimensions", "together_with")
    @classmethod
    def _values(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        return None if value is None else _check_dimensions(value)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.together_with == self.dimensions:
            raise ValueError("together_with igual a dimensions")
        return self


class ChangesParams(WindowParams):
    scope: dict[Dimension, str] | None = None
    lookback_days: int = Field(default=7, ge=0, le=30)

    @field_validator("scope")
    @classmethod
    def _values(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        return None if value is None else _check_dimensions(value)


# --- Contexto -------------------------------------------------------------------------------


class ToolContext:
    """Conexão somente leitura, cubo em memória e numeração dos `Finding`s."""

    def __init__(self, con: sqlite3.Connection, settings: Settings | None = None) -> None:
        self.con = con
        self.settings = settings or get_settings()
        self.findings: list[Finding] = []
        self._cube: Cube | None = None

    @property
    def cube(self) -> Cube:
        if self._cube is None:
            self._cube = Cube.from_connection(self.con)
        return self._cube

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
        return finding

    def run(self, tool: str, raw_params: dict) -> Finding:
        """Executa uma ferramenta do menu. Erros viram `Finding` com a chave `error`."""
        spec = TOOLS.get(tool)
        if spec is None:
            return self.finding(tool, raw_params, {"error": f"ferramenta desconhecida: {tool}"})
        try:
            params = spec.params.model_validate(raw_params)
            result = spec.run(self, params)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
            )
            return self.finding(tool, raw_params, {"error": f"parâmetros inválidos: {problems}"})
        except ValueError as error:  # inclui ToolError
            return self.finding(tool, raw_params, {"error": str(error)})
        return self.finding(tool, params.model_dump(mode="json"), result)


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


def _total_table(ctx: ToolContext, metric: Metric, params: WindowParams) -> SliceTable:
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


def decompose_metric(ctx: ToolContext, p: WindowParams) -> dict:
    """Qual fator se moveu em revenue = sessions × conversion_rate × avg_ticket (em log)."""
    tables = {
        m: _total_table(ctx, m, p) for m in ("revenue", "sessions", "conversion_rate", "avg_ticket")
    }
    total_log = float(np.log(tables["revenue"].total_obs / tables["revenue"].total_exp))
    factors = {}
    for name in ("sessions", "conversion_rate", "avg_ticket"):
        t = tables[name]
        ratio = t.total_obs / t.total_exp
        log_change = float(np.log(ratio))
        factors[name] = {
            "observed": _num(t.total_obs),
            "expected": _num(t.total_exp),
            "change_pct": _num(ratio - 1),
            "log_share": _num(log_change / total_log) if abs(total_log) > 1e-9 else None,
        }
    revenue = tables["revenue"]
    return {
        "revenue": {
            "observed": _num(revenue.total_obs),
            "expected": _num(revenue.total_exp),
            "change_pct": _num(revenue.total_obs / revenue.total_exp - 1),
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


def check_recent_changes(ctx: ToolContext, p: ChangesParams) -> dict:
    window = ctx.window(p)
    first = window.start - timedelta(days=p.lookback_days)
    rows = ctx.con.execute(
        "SELECT ts, type, description, scope FROM changes WHERE ts >= ? AND ts <= ? ORDER BY ts",
        (first.isoformat(), window.end.isoformat()),
    ).fetchall()
    changes = []
    for ts, kind, description, scope_json in rows:
        scope = json.loads(scope_json)
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

    in_window = (cube.day_idx >= first) & (cube.day_idx <= last)
    rows_per_day = np.bincount(cube.day_idx, minlength=cube.n_days)
    typical = int(np.median(rows_per_day[rows_per_day > 0]))
    zero_slices = []
    for name in DIMENSION_VALUES:
        for metric in ("sessions", "conversions"):
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
        "zero_session_rows": int((cube.values["sessions"][in_window] == 0).sum()),
        "zeroed_slices": zero_slices,
    }


# --- Menu -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: type[BaseModel]
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
            WindowParams,
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
    )
}


def tool_catalog() -> list[dict]:
    """Nome, descrição e JSON Schema de cada ferramenta (o que o LLM verá no modo `llm`)."""
    return [
        {"name": s.name, "description": s.description, "parameters": s.params.model_json_schema()}
        for s in TOOLS.values()
    ]
