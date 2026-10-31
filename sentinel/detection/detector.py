"""Detector de anomalias (seção 9.1 do CLAUDE.md). Código puro, sem LLM.

Para cada métrica diária:
- esperado = mediana dos mesmos dias da semana nas `baseline_weeks` semanas anteriores;
- resíduo relativo = observado / esperado − 1;
- escala = desvio-padrão robusto dos resíduos históricos (só dias anteriores à janela);
- z = resíduo relativo / escala;
- anomalia = `min_consecutive_days` dias seguidos, do mesmo sinal, com |z| >= `z_threshold`,
  ou um deslocamento sustentado na janela inteira.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from sentinel.config import Settings
from sentinel.schemas import Anomaly, Metric

# Ordem de prioridade para escolher a métrica-raiz:
# revenue = sessions × conversion_rate × avg_ticket; as conversions e a revenue são derivadas.
METRIC_PRIORITY: tuple[Metric, ...] = (
    "sessions",
    "conversion_rate",
    "avg_ticket",
    "conversions",
    "revenue",
)

MIN_HISTORY_DAYS = 14
MIN_SCALE = 1e-4
ONSET_LOOKBACK_DAYS = 7
MAD_TO_SIGMA = 1.4826
CLIP_SCALES = 3.0


@dataclass(frozen=True)
class DetectorParams:
    z_threshold: float = 3.5
    sustained_z_threshold: float = 5.5
    min_consecutive_days: int = 2
    baseline_weeks: int = 8
    window_days: int = 7

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectorParams":
        return cls(
            z_threshold=settings.z_threshold,
            sustained_z_threshold=settings.sustained_z_threshold,
            min_consecutive_days=settings.min_consecutive_days,
            baseline_weeks=settings.baseline_weeks,
            window_days=settings.window_days,
        )


# --- Séries diárias -------------------------------------------------------------------------


def _with_derived(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.astype(float)
    daily["conversion_rate"] = daily["conversions"] / daily["sessions"].where(daily["sessions"] > 0)
    daily["avg_ticket"] = daily["revenue"] / daily["conversions"].where(daily["conversions"] > 0)
    return daily[list(METRIC_PRIORITY)]


def daily_series(events: pd.DataFrame) -> pd.DataFrame:
    """Série diária agregada (índice = dia, colunas = métricas) a partir da tabela `events`."""
    daily = events.groupby("day")[["sessions", "conversions", "revenue"]].sum()
    daily.index = pd.to_datetime(daily.index)
    return _with_derived(daily)


def load_daily_series(con: sqlite3.Connection) -> pd.DataFrame:
    """O mesmo que `daily_series`, lendo do banco (somente leitura)."""
    daily = pd.read_sql_query(
        "SELECT day, SUM(sessions) AS sessions, SUM(conversions) AS conversions, "
        "SUM(revenue) AS revenue FROM events GROUP BY day ORDER BY day",
        con,
        index_col="day",
    )
    daily.index = pd.to_datetime(daily.index)
    return _with_derived(daily)


# --- Núcleo ---------------------------------------------------------------------------------


def expected_series(values: pd.Series, weeks: int) -> pd.Series:
    """Mediana dos mesmos dias da semana nas `weeks` semanas anteriores (NaN sem histórico)."""
    lags = pd.concat([values.shift(7 * k) for k in range(1, weeks + 1)], axis=1)
    return lags.median(axis=1, skipna=False)


def robust_scale(history: np.ndarray) -> float:
    """MAD × 1,4826, refinado: desvio-padrão dos resíduos limitados a ±3 escalas MAD.

    Só com o MAD, a escala estimada em ~28 dias oscila muito (eficiência de ~37%) e cria
    falsos alarmes raros. O limite de ±3 mantém a robustez a pontos fora da curva.
    """
    center = float(np.median(history))
    mad_scale = float(np.median(np.abs(history - center))) * MAD_TO_SIGMA
    if mad_scale < MIN_SCALE:
        return MIN_SCALE
    clipped = np.clip(history, center - CLIP_SCALES * mad_scale, center + CLIP_SCALES * mad_scale)
    return max(float(clipped.std(ddof=1)), MIN_SCALE)


def _is_sustained_shift(window_z: np.ndarray, params: DetectorParams) -> bool:
    """Deslocamento pequeno, mas persistente na janela inteira (ex.: deriva lenta).

    Dias isolados ficam abaixo do limiar, mas a média da janela, em desvios-padrão da média,
    não fica. Quase todos os dias precisam ter o mesmo sinal.
    """
    days = len(window_z)
    if days < 2 * params.min_consecutive_days:
        return False
    mean_z = float(window_z.mean())
    same_sign = int((np.sign(window_z) == np.sign(mean_z)).sum())
    return abs(mean_z) * np.sqrt(days) >= params.sustained_z_threshold and same_sign >= days - 2


def _runs(flags: np.ndarray, signs: np.ndarray) -> list[tuple[int, int]]:
    """Trechos consecutivos de dias sinalizados com o mesmo sinal (posições inclusivas)."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flagged in enumerate(flags):
        extends = start is not None and signs[i] == signs[i - 1]
        if flagged and extends:
            continue
        if start is not None:
            runs.append((start, i - 1))
            start = None
        if flagged:
            start = i
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _refine_onset(z: np.ndarray, run_start: int, run_end: int, first_valid: int) -> int:
    """Changepoint simples: ajusta dois níveis constantes à série de z e escolhe o melhor corte."""
    lo = max(first_valid, run_start - ONSET_LOOKBACK_DAYS)
    segment = np.nan_to_num(z[lo : run_end + 1])
    best_split, best_sse = run_start, np.inf
    for split in range(lo + 1, run_start + 1):
        k = split - lo
        left, right = segment[:k], segment[k:]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if sse < best_sse:
            best_split, best_sse = split, sse
    return best_split


def _detect_metric(
    name: Metric, values: pd.Series, params: DetectorParams, end_pos: int
) -> Anomaly | None:
    expected = expected_series(values, params.baseline_weeks)
    first_valid = 7 * params.baseline_weeks
    window_start = end_pos - params.window_days + 1
    if window_start - first_valid < MIN_HISTORY_DAYS:
        raise ValueError(
            f"histórico insuficiente: são necessários {first_valid + MIN_HISTORY_DAYS} dias "
            f"antes do fim da janela de {params.window_days} dias"
        )

    relative = (values / expected - 1).to_numpy()
    history = relative[first_valid:window_start]
    history = history[~np.isnan(history)]
    if len(history) < MIN_HISTORY_DAYS:
        return None
    z = relative / robust_scale(history)

    window_z = np.nan_to_num(z[window_start : end_pos + 1])
    flags = np.abs(window_z) >= params.z_threshold
    runs = [
        r for r in _runs(flags, np.sign(window_z)) if r[1] - r[0] + 1 >= params.min_consecutive_days
    ]
    if runs:
        first, last = max(runs, key=lambda r: (r[1] - r[0], r[0]))  # o trecho mais longo
        first, last = first + window_start, last + window_start
    elif _is_sustained_shift(window_z, params):
        first, last = window_start, end_pos
    else:
        return None
    onset = _refine_onset(z, first, last, first_valid)

    index = values.index
    observed = float(values.iloc[first : last + 1].mean())
    expected_mean = float(expected.iloc[first : last + 1].mean())
    z_mean = float(np.mean(z[first : last + 1]))
    return Anomaly(
        metric=name,
        window_start=index[first].date(),
        window_end=index[last].date(),
        onset=index[onset].date(),
        observed=observed,
        expected=expected_mean,
        z_score=z_mean,
        direction="drop" if z_mean < 0 else "spike",
    )


def detect(
    series: pd.DataFrame, params: DetectorParams | None = None, as_of: date | None = None
) -> list[Anomaly]:
    """Anomalias na janela que termina em `as_of` (padrão: último dia), por ordem de prioridade."""
    params = params or DetectorParams()
    if (
        not series.index.is_monotonic_increasing
        or not (series.index.diff()[1:] == pd.Timedelta(days=1)).all()
    ):
        raise ValueError("a série precisa ter um ponto por dia, sem buracos")
    end_pos = len(series) - 1 if as_of is None else int(series.index.get_loc(pd.Timestamp(as_of)))

    found = [
        anomaly
        for name in METRIC_PRIORITY
        if (anomaly := _detect_metric(name, series[name], params, end_pos)) is not None
    ]
    return found


def select_root(anomalies: list[Anomaly]) -> Anomaly | None:
    """A métrica-raiz é a primeira da árvore de prioridade que disparou."""
    if not anomalies:
        return None
    return min(anomalies, key=lambda a: METRIC_PRIORITY.index(a.metric))
