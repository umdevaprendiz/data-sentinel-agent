"""Drill-down: poder de explicação por fatia e busca de causas (seção 9.2 do CLAUDE.md).

Código puro, sem LLM. Para uma métrica e uma janela, compara cada fatia dos dados com o
esperado dela (mediana dos mesmos dias da semana nas semanas anteriores) e mede quanto da
variação total cada fatia explica.

- Métrica aditiva (`sessions`, `conversions`, `revenue`): EP(v) = (obs_v − esp_v) / (obs − esp).
- Razão (`conversion_rate`, `avg_ticket`): o impacto soma o efeito da taxa e o efeito de
  mistura, de modo que a soma dos impactos das fatias é exatamente a variação da razão.
"""

import itertools
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from sentinel.config import Settings
from sentinel.detection.detector import MIN_HISTORY_DAYS, robust_scale
from sentinel.investigation.cube import Cube
from sentinel.schemas import DIMENSION_VALUES, Metric

# Razões: (numerador, denominador).
RATIO_METRICS: dict[str, tuple[str, str]] = {
    "conversion_rate": ("conversions", "sessions"),
    "avg_ticket": ("revenue", "conversions"),
}
OUTSIDE = "__fora__"


@dataclass(frozen=True)
class DrillParams:
    baseline_weeks: int = 8
    min_slice_sessions: int = 200
    ep_single: float = 0.6
    ep_pair: float = 0.8
    ep_multi_each: float = 0.3
    ep_multi_total: float = 0.8
    z_stable: float = 3.5
    ep_material: float = 0.1
    refine_keep: float = 0.9
    min_total_z: float = 3.5

    @classmethod
    def from_settings(cls, settings: Settings) -> "DrillParams":
        return cls(
            baseline_weeks=settings.baseline_weeks,
            min_slice_sessions=settings.min_slice_sessions,
            ep_single=settings.ep_single,
            ep_pair=settings.ep_pair,
            ep_multi_each=settings.ep_multi_each,
            ep_multi_total=settings.ep_multi_total,
            z_stable=settings.z_stable,
            ep_material=settings.ep_material,
            refine_keep=settings.refine_keep,
            min_total_z=settings.min_total_z,
        )


@dataclass(frozen=True)
class Window:
    start: date
    end: date


@dataclass(frozen=True)
class Candidate:
    dimensions: dict[str, str]
    explained_share: float  # 0..1


@dataclass
class SliceTable:
    metric: str
    labels: list[tuple[str, ...]]
    obs: np.ndarray
    exp: np.ndarray
    change_pct: np.ndarray
    impact: np.ndarray
    ep: np.ndarray
    z: np.ndarray  # z da média da janela (≈ N(0, 1) numa fatia sem mudança, qualquer duração)
    eligible: np.ndarray  # sessões esperadas por dia >= MIN_SLICE_SESSIONS
    nonempty: np.ndarray
    total_obs: float
    total_exp: float
    total_change: float

    @property
    def n_nonempty(self) -> int:
        return int(self.nonempty.sum())

    def others_stable(self, index: int, params: "DrillParams") -> bool:
        """As outras fatias não vazias estão no normal.

        Uma fatia só conta como instável se saiu do ruído (|z| >= z_stable) E explica uma parte
        relevante da variação total (|EP| >= ep_material). Z ou EP desconhecidos contam contra.
        """
        for j in range(len(self.labels)):
            if j == index or not self.nonempty[j]:
                continue
            significant = not abs(self.z[j]) < params.z_stable
            material = not abs(self.ep[j]) < params.ep_material
            if significant and material:
                return False
        return True


# --- Tabelas de fatias ----------------------------------------------------------------------


def _expected(array: np.ndarray, weeks: int) -> np.ndarray:
    """Mediana dos mesmos dias da semana nas `weeks` semanas anteriores (NaN sem histórico)."""
    lags = np.full((weeks, *array.shape), np.nan)
    for k in range(1, weeks + 1):
        shift = 7 * k
        if shift < array.shape[0]:
            lags[k - 1, shift:] = array[: array.shape[0] - shift]
    return np.median(lags, axis=0)


def _safe_div(num: np.ndarray, den: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full(np.broadcast(num, den).shape, fill, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def build_table(
    cube: Cube,
    metric: Metric,
    keys: np.ndarray,
    labels: list[tuple[str, ...]],
    window: Window,
    params: DrillParams,
) -> SliceTable:
    first, last = cube.pos(window.start), cube.pos(window.end)
    if last < first:
        raise ValueError("janela invertida")
    history_start = 7 * params.baseline_weeks
    if first - history_start < MIN_HISTORY_DAYS:
        raise ValueError(
            f"histórico insuficiente: a janela precisa começar depois de "
            f"{history_start + MIN_HISTORY_DAYS} dias de dados"
        )

    n_slices = len(labels)
    daily = cube.daily(keys, n_slices)
    expected = {name: _expected(array, params.baseline_weeks) for name, array in daily.items()}
    win = slice(first, last + 1)
    obs_sum = {name: array[win].sum(axis=0) for name, array in daily.items()}
    exp_sum = {name: array[win].sum(axis=0) for name, array in expected.items()}

    if metric in RATIO_METRICS:
        num_name, den_name = RATIO_METRICS[metric]
        n_obs, d_obs = obs_sum[num_name], obs_sum[den_name]
        n_exp, d_exp = exp_sum[num_name], exp_sum[den_name]
        if d_obs.sum() <= 0 or d_exp.sum() <= 0:
            raise ValueError("denominador zerado na janela")
        r_exp = _safe_div(n_exp, d_exp, fill=0.0)
        r_obs = np.where(d_obs > 0, _safe_div(n_obs, d_obs, fill=0.0), r_exp)
        total_obs, total_exp = n_obs.sum() / d_obs.sum(), n_exp.sum() / d_exp.sum()
        w_obs, w_exp = d_obs / d_obs.sum(), d_exp / d_exp.sum()
        rate_effect = (r_obs - r_exp) * w_obs
        mix_effect = (w_obs - w_exp) * (r_exp - total_exp)
        impact = rate_effect + mix_effect
        obs, exp = r_obs, r_exp
        series = _safe_div(daily[num_name], daily[den_name])
        series_exp = _expected(series, params.baseline_weeks)
    else:
        obs, exp = obs_sum[metric], exp_sum[metric]
        total_obs, total_exp = float(obs.sum()), float(exp.sum())
        impact = obs - exp
        series, series_exp = daily[metric], expected[metric]

    total_change = float(total_obs - total_exp)
    if abs(total_change) > 1e-12 * max(1.0, abs(total_exp)):
        ep = impact / total_change
    else:
        ep = np.full(n_slices, np.nan)
    change_pct = _safe_div(obs - exp, np.abs(exp))
    change_pct = np.where(exp > 0, change_pct, np.nan)

    relative = np.where(
        series_exp > 0, series / np.where(series_exp > 0, series_exp, 1) - 1, np.nan
    )
    z = np.full(n_slices, np.nan)
    for s in range(n_slices):
        history = relative[history_start:first, s]
        history = history[~np.isnan(history)]
        if len(history) < MIN_HISTORY_DAYS:
            continue
        daily_z = np.nan_to_num(relative[win, s] / robust_scale(history))
        z[s] = daily_z.mean() * np.sqrt(last - first + 1)  # z da média da janela

    days = last - first + 1
    sessions_per_day = exp_sum["sessions"] / days
    return SliceTable(
        metric=metric,
        labels=labels,
        obs=obs,
        exp=exp,
        change_pct=change_pct,
        impact=impact,
        ep=ep,
        z=z,
        eligible=sessions_per_day >= params.min_slice_sessions,
        nonempty=exp_sum["sessions"] > 0,
        total_obs=float(total_obs),
        total_exp=float(total_exp),
        total_change=total_change,
    )


def dims_table(
    cube: Cube, metric: Metric, dims: tuple[str, ...], window: Window, params: DrillParams
) -> SliceTable:
    keys = np.zeros(cube.n_rows, dtype=np.int64)
    for name in dims:
        keys = keys * len(DIMENSION_VALUES[name]) + cube.codes[name]
    labels = list(itertools.product(*(DIMENSION_VALUES[name] for name in dims)))
    return build_table(cube, metric, keys, labels, window, params)


def membership_table(
    cube: Cube, metric: Metric, inside: np.ndarray, window: Window, params: DrillParams
) -> SliceTable:
    """Duas fatias: dentro (índice 0) e fora (índice 1) da hipótese."""
    keys = np.where(inside, 0, 1).astype(np.int64)
    return build_table(cube, metric, keys, [("dentro",), (OUTSIDE,)], window, params)


# --- Busca de causas ------------------------------------------------------------------------


def _share(ep: float) -> float:
    return float(min(max(ep, 0.0), 1.0))


def single_dim_candidates(
    cube: Cube,
    metric: Metric,
    window: Window,
    params: DrillParams,
    min_ep: float | None = None,
    require_stable: bool = True,
) -> list[Candidate]:
    """Fatias de uma dimensão com EP alto e, se pedido, as demais fatias da dimensão estáveis."""
    threshold = params.ep_single if min_ep is None else min_ep
    found: list[Candidate] = []
    for name in DIMENSION_VALUES:
        table = dims_table(cube, metric, (name,), window, params)
        if table.n_nonempty < 2:
            continue
        for i, (value,) in enumerate(table.labels):
            if not (table.eligible[i] and table.ep[i] >= threshold):
                continue
            if require_stable and not table.others_stable(i, params):
                continue
            found.append(Candidate({name: value}, _share(table.ep[i])))
    return sorted(found, key=lambda c: -c.explained_share)


def refinement_candidates(
    cube: Cube,
    metric: Metric,
    window: Window,
    params: DrillParams,
    dimensions: dict[str, str],
    explained_share: float,
) -> list[Candidate]:
    """Sub-fatias da hipótese (uma dimensão a mais) que ainda explicam quase o mesmo."""
    inside = cube.mask(dimensions)
    found: list[Candidate] = []
    for extra, values in DIMENSION_VALUES.items():
        if extra in dimensions:
            continue
        keys = np.where(inside, cube.codes[extra], len(values)).astype(np.int64)
        labels = [(v,) for v in values] + [(OUTSIDE,)]
        table = build_table(cube, metric, keys, labels, window, params)
        for i, value in enumerate(values):
            if not (table.eligible[i] and table.ep[i] >= params.refine_keep * explained_share):
                continue
            if table.others_stable(i, params):
                found.append(Candidate({**dimensions, extra: value}, _share(table.ep[i])))
    return sorted(found, key=lambda c: -c.explained_share)


def _refine(
    cube: Cube, metric: Metric, window: Window, params: DrillParams, c: Candidate
) -> Candidate:
    if len(c.dimensions) >= 2:
        return c
    refined = refinement_candidates(cube, metric, window, params, c.dimensions, c.explained_share)
    return refined[0] if refined else c


def _best_pair(cube: Cube, metric: Metric, window: Window, params: DrillParams) -> Candidate | None:
    best: Candidate | None = None
    for first, second in itertools.combinations(DIMENSION_VALUES, 2):
        table = dims_table(cube, metric, (first, second), window, params)
        for i, (a, b) in enumerate(table.labels):
            if table.eligible[i] and table.ep[i] >= params.ep_pair:
                share = _share(table.ep[i])
                if best is None or share > best.explained_share:
                    best = Candidate({first: a, second: b}, share)
    return best


def _two_causes(cube: Cube, metric: Metric, window: Window, params: DrillParams) -> list[Candidate]:
    """Duas causas independentes: tira a primeira dos dados e procura a segunda no que sobra."""
    firsts = single_dim_candidates(
        cube, metric, window, params, min_ep=params.ep_multi_each, require_stable=False
    )
    for first in firsts:
        rest = cube.subset(~cube.mask(first.dimensions))
        if not total_is_significant(rest, metric, window, params):
            continue  # o que sobra é ruído: não há segunda causa para explicar
        seconds = single_dim_candidates(rest, metric, window, params)
        for second in seconds:
            second = _refine(rest, metric, window, params, second)
            union = cube.mask(first.dimensions) | cube.mask(second.dimensions)
            joint = membership_table(cube, metric, union, window, params)
            alone = membership_table(cube, metric, cube.mask(second.dimensions), window, params)
            if (
                joint.ep[0] >= params.ep_multi_total
                and joint.others_stable(0, params)
                and alone.ep[0] >= params.ep_multi_each
            ):
                first = _refine(cube, metric, window, params, first)
                return [first, Candidate(second.dimensions, _share(alone.ep[0]))]
    return []


def total_z(cube: Cube, metric: Metric, window: Window, params: DrillParams) -> float:
    """z médio da métrica agregada na janela (NaN se não der para estimar)."""
    keys = np.zeros(cube.n_rows, dtype=np.int64)
    return float(build_table(cube, metric, keys, [("total",)], window, params).z[0])


def total_is_significant(cube: Cube, metric: Metric, window: Window, params: DrillParams) -> bool:
    """O poder de explicação só faz sentido se a variação total saiu do ruído."""
    return abs(total_z(cube, metric, window, params)) >= params.min_total_z


def find_causes(
    cube: Cube, metric: Metric, window: Window, params: DrillParams | None = None
) -> list[Candidate]:
    """Causas da variação de `metric` na janela. Lista vazia = inconclusivo."""
    params = params or DrillParams()
    if not total_is_significant(cube, metric, window, params):
        return []

    singles = single_dim_candidates(cube, metric, window, params)
    if singles:
        refined = [_refine(cube, metric, window, params, c) for c in singles]
        return [max(refined, key=lambda c: (c.explained_share, len(c.dimensions)))]

    pair = _best_pair(cube, metric, window, params)
    if pair is not None:
        return [pair]

    return _two_causes(cube, metric, window, params)


# --- Avaliação de uma hipótese dada ---------------------------------------------------------


@dataclass(frozen=True)
class SliceStats:
    observed: float
    expected: float
    change_pct: float | None
    z: float | None


@dataclass(frozen=True)
class HypothesisEval:
    dimensions: dict[str, str]
    explained_share: float | None
    threshold: float
    inside: SliceStats
    outside: SliceStats
    others_stable: bool
    eligible: bool
    total_significant: bool
    accepted: bool
    refinements: list[Candidate]
    together_with: dict[str, str] | None = None  # 2ª causa, numa hipótese de duas causas
    members: list[Candidate] = field(default_factory=list)  # quanto cada causa explica sozinha


def _stats(table: SliceTable, index: int) -> SliceStats:
    def clean(value: float) -> float | None:
        return None if np.isnan(value) else float(value)

    return SliceStats(
        observed=float(table.obs[index]),
        expected=float(table.exp[index]),
        change_pct=clean(table.change_pct[index]),
        z=clean(table.z[index]),
    )


def _evaluate_two_causes(
    cube: Cube,
    metric: Metric,
    window: Window,
    params: DrillParams,
    first: dict[str, str],
    second: dict[str, str],
) -> HypothesisEval:
    """Duas causas independentes: juntas explicam o suficiente e cada uma pesa o bastante."""
    union = membership_table(cube, metric, cube.mask(first) | cube.mask(second), window, params)
    alone = [membership_table(cube, metric, cube.mask(d), window, params) for d in (first, second)]
    ep = None if np.isnan(union.ep[0]) else float(union.ep[0])
    eligible = bool(union.eligible[0] and union.nonempty[0] and union.nonempty[1])
    stable = union.others_stable(0, params)
    significant = total_is_significant(cube, metric, window, params)
    each = [float(np.nan_to_num(t.ep[0], nan=0.0)) for t in alone]
    accepted = (
        significant
        and eligible
        and ep is not None
        and ep >= params.ep_multi_total
        and stable
        and all(share >= params.ep_multi_each for share in each)
    )
    return HypothesisEval(
        dimensions=first,
        explained_share=ep,
        threshold=params.ep_multi_total,
        inside=_stats(union, 0),
        outside=_stats(union, 1),
        others_stable=stable,
        eligible=eligible,
        total_significant=significant,
        accepted=accepted,
        refinements=[],
        together_with=second,
        members=[
            Candidate(d, _share(share)) for d, share in zip((first, second), each, strict=True)
        ],
    )


def evaluate_hypothesis(
    cube: Cube,
    metric: Metric,
    window: Window,
    params: DrillParams,
    dimensions: dict[str, str],
    together_with: dict[str, str] | None = None,
) -> HypothesisEval:
    """Avalia uma causa; com `together_with`, avalia a hipótese de duas causas independentes."""
    if together_with is not None:
        return _evaluate_two_causes(cube, metric, window, params, dimensions, together_with)
    inside = cube.mask(dimensions)
    table = membership_table(cube, metric, inside, window, params)
    threshold = params.ep_single if len(dimensions) == 1 else params.ep_pair

    if len(dimensions) == 1:
        ((name, value),) = dimensions.items()
        by_dim = dims_table(cube, metric, (name,), window, params)
        index = DIMENSION_VALUES[name].index(value)
        stable = by_dim.others_stable(index, params)
    else:
        stable = table.others_stable(0, params)

    ep = None if np.isnan(table.ep[0]) else float(table.ep[0])
    proper = bool(table.nonempty[0] and table.nonempty[1])
    eligible = bool(table.eligible[0]) and proper
    significant = total_is_significant(cube, metric, window, params)
    accepted = significant and eligible and ep is not None and ep >= threshold and stable
    refinements = (
        refinement_candidates(cube, metric, window, params, dimensions, _share(ep))
        if len(dimensions) == 1 and ep is not None and eligible
        else []
    )
    return HypothesisEval(
        dimensions=dimensions,
        explained_share=ep,
        threshold=threshold,
        inside=_stats(table, 0),
        outside=_stats(table, 1),
        others_stable=stable,
        eligible=eligible,
        total_significant=significant,
        accepted=accepted,
        refinements=refinements,
    )
