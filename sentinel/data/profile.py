"""Perfil dos dados descoberto no banco (decisão 21): o que o agente pode vigiar.

O agente não traz uma lista fixa de métricas e dimensões. A cada execução ele olha a tabela
`events` e descobre:

- a **coluna de tempo** (texto em formato de data ISO);
- as **dimensões** (texto de baixa cardinalidade);
- as **medidas aditivas** (colunas numéricas, somáveis por dia e por fatia);
- as **razões** entre duas medidas: as de nome conhecido (`conversion_rate`, `avg_ticket`) quando
  as colunas existem, e as **taxas** descobertas (numerador <= denominador em todas as linhas).

O LLM pode propor outra razão ou uma dimensão de cardinalidade maior; o código valida e só então
ela entra no perfil (`with_ratio`, `with_dimension`). O perfil tem uma impressão digital (hash):
quando a estrutura muda, o agente registra no audit e recalcula, sem mudança de código.
"""

import hashlib
import json
import re
import sqlite3
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.schemas import DIMENSION_VALUES, NAME_PATTERN
from sentinel.security import clean_text

TABLE = "events"
NAME = re.compile(NAME_PATTERN)
MAX_DIMENSION_VALUES = 50  # acima disso a coluna de texto não vira dimensão sozinha
HARD_MAX_DIMENSION_VALUES = 200  # limite para uma dimensão proposta pelo LLM
MAX_VALUE_LENGTH = 80
TIME_NAMES = ("day", "date", "dt", "ts")

# Razões com nome conhecido (comércio eletrônico). Entram se as duas colunas existirem.
KNOWN_RATIOS: dict[str, tuple[str, str]] = {
    "conversion_rate": ("conversions", "sessions"),
    "avg_ticket": ("revenue", "conversions"),
}


class ProfileError(ValueError):
    """Pedido incompatível com os dados (coluna inexistente, razão mal definida…)."""


class Ratio(BaseModel):
    model_config = ConfigDict(frozen=True)

    numerator: str
    denominator: str
    is_rate: bool  # numerador <= denominador em todas as linhas: formatada como porcentagem
    source: Literal["known", "discovered", "proposed"]


class SkippedColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    reason: str
    distinct: int | None = None


class SchemaProfile(BaseModel):
    """Estrutura dos dados: o que pode ser métrica e o que pode ser dimensão."""

    model_config = ConfigDict(frozen=True)

    table: str = TABLE
    time_column: str
    dimensions: dict[str, tuple[str, ...]]
    measures: tuple[str, ...]
    ratios: dict[str, Ratio] = Field(default_factory=dict)
    skipped: tuple[SkippedColumn, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> Self:
        names = [self.time_column, *self.dimensions, *self.measures, *self.ratios]
        for name in names:
            if not NAME.match(name):
                raise ValueError(f"nome inválido no perfil: {name!r}")
        if len(set(names)) != len(names):
            raise ValueError("nomes repetidos no perfil")
        if not self.measures:
            raise ValueError("o perfil precisa de ao menos uma medida")
        for ratio in self.ratios.values():
            if {ratio.numerator, ratio.denominator} - set(self.measures):
                raise ValueError("razão com coluna que não é medida")
        return self

    # --- Ordem e papéis ------------------------------------------------------------------------

    def _depth(self) -> dict[str, int]:
        """Distância até a raiz do funil.

        0 = denominador que não é numerador de nada (sessões). Cada razão soma um passo
        (conversões = sessões × taxa). Medida fora de qualquer razão fica no fim da fila: sem
        saber de onde ela vem, uma queda nela não explica as outras.
        """
        denominators = {r.denominator for r in self.ratios.values()}
        depth = {m: 0 for m in self.measures if not self._as_numerator(m) and m in denominators}
        for _ in self.measures:  # relaxação: no máximo uma volta por medida
            for ratio in self.ratios.values():
                if ratio.denominator in depth:
                    candidate = depth[ratio.denominator] + 1
                    if candidate < depth.get(ratio.numerator, len(self.measures) + 1):
                        depth[ratio.numerator] = candidate
        return {m: depth.get(m, len(self.measures)) for m in self.measures}

    def _as_numerator(self, measure: str) -> list[str]:
        return [n for n, r in self.ratios.items() if r.numerator == measure]

    @property
    def metrics(self) -> tuple[str, ...]:
        """Métricas em ordem de prioridade para a métrica-raiz.

        Raízes do funil (sessões) primeiro, depois as razões (da mais próxima da raiz para a mais
        distante) e por fim as medidas derivadas (conversões, receita): uma queda de sessões
        explica a queda de receita, não o contrário.
        """
        depth = self._depth()
        roots = [m for m in self.measures if depth[m] == 0]
        ratios = sorted(self.ratios, key=lambda n: (depth[self.ratios[n].denominator], n))
        derived = sorted((m for m in self.measures if depth[m] > 0), key=lambda m: depth[m])
        return (*roots, *ratios, *derived)

    @property
    def volume(self) -> str:
        """Medida de volume (sessões): define o tamanho mínimo de uma fatia candidata."""
        depth = self._depth()
        denominators = {r.denominator for r in self.ratios.values()}
        for measure in self.measures:
            if depth[measure] == 0 and measure in denominators:
                return measure
        return self.measures[0]

    @property
    def rates(self) -> tuple[str, ...]:
        return tuple(n for n, r in self.ratios.items() if r.is_rate)

    @property
    def count_measures(self) -> tuple[str, ...]:
        """Volume e numeradores de taxas: as contagens que podem zerar numa fatia."""
        numerators = [r.numerator for r in self.ratios.values() if r.is_rate]
        return tuple(dict.fromkeys([self.volume, *numerators]))

    def ratio_parts(self, metric: str) -> tuple[str, str] | None:
        ratio = self.ratios.get(metric)
        return None if ratio is None else (ratio.numerator, ratio.denominator)

    def chain(self, measure: str) -> list[str]:
        """Fatores de `measure` = raiz × razão × razão… (ex.: receita = sessões × taxa × ticket)."""
        self.check_metric(measure)
        if measure not in self.measures:
            raise ProfileError(f"{measure} não é uma medida aditiva")
        factors: list[str] = []
        current, seen = measure, set()
        while current not in seen:
            seen.add(current)
            step = next(
                (
                    (n, r)
                    for n, r in sorted(self.ratios.items())
                    if r.numerator == current and r.denominator not in seen
                ),
                None,
            )
            if step is None:
                break
            name, ratio = step
            factors.insert(0, name)
            current = ratio.denominator
        return [current, *factors]

    @property
    def deepest_measure(self) -> str:
        """A medida mais distante da raiz que ainda se decompõe em fatores (receita)."""
        depth = self._depth()
        chained = [m for m in self.measures if 0 < depth[m] < len(self.measures)]
        if not chained:
            return self.measures[0]
        return max(chained, key=lambda m: (depth[m], -self.measures.index(m)))

    # --- Validação de pedidos ------------------------------------------------------------------

    def check_metric(self, name: str) -> str:
        if name not in self.metrics:
            raise ProfileError(f"métrica desconhecida: {name}")
        return name

    def check_dimension(self, name: str) -> str:
        if name not in self.dimensions:
            raise ProfileError(f"dimensão desconhecida: {name}")
        return name

    def check_dimensions(self, dimensions: dict[str, str]) -> dict[str, str]:
        for name, value in dimensions.items():
            self.check_dimension(name)
            if value not in self.dimensions[name]:
                raise ProfileError(f"valor inválido para {name}: {clean_text(str(value), 40)}")
        return dimensions

    # --- Evolução ------------------------------------------------------------------------------

    def with_ratio(self, name: str, ratio: Ratio) -> "SchemaProfile":
        if not NAME.match(name):
            raise ProfileError("nome de métrica inválido")
        if name in self.metrics or name in self.dimensions or name == self.time_column:
            raise ProfileError(f"o nome {name} já existe")
        if ratio.numerator == ratio.denominator:
            raise ProfileError("numerador e denominador iguais")
        for column in (ratio.numerator, ratio.denominator):
            if column not in self.measures:
                raise ProfileError(f"{column} não é uma medida aditiva")
        return self.model_copy(update={"ratios": {**self.ratios, name: ratio}})

    def with_dimension(self, name: str, values: tuple[str, ...]) -> "SchemaProfile":
        if name in self.dimensions:
            raise ProfileError(f"{name} já é dimensão")
        skipped = tuple(s for s in self.skipped if s.column != name)
        return self.model_copy(
            update={"dimensions": {**self.dimensions, name: values}, "skipped": skipped}
        )

    # --- Impressão digital ---------------------------------------------------------------------

    def structure(self) -> dict:
        return {
            "table": self.table,
            "time_column": self.time_column,
            "dimensions": {k: list(v) for k, v in self.dimensions.items()},
            "measures": list(self.measures),
            "ratios": {k: [r.numerator, r.denominator] for k, r in sorted(self.ratios.items())},
        }

    @property
    def fingerprint(self) -> str:
        text = json.dumps(self.structure(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def diff(self, previous: dict) -> list[str]:
        """O que mudou em relação a uma estrutura anterior (para o audit)."""
        now = self.structure()
        changes: list[str] = []
        for key in ("measures",):
            for name in sorted(set(now[key]) - set(previous.get(key, []))):
                changes.append(f"medida nova: {name}")
            for name in sorted(set(previous.get(key, [])) - set(now[key])):
                changes.append(f"medida removida: {name}")
        before = previous.get("dimensions", {})
        for name in sorted(set(now["dimensions"]) - set(before)):
            changes.append(f"dimensão nova: {name}")
        for name in sorted(set(before) - set(now["dimensions"])):
            changes.append(f"dimensão removida: {name}")
        for name in sorted(set(now["dimensions"]) & set(before)):
            added = set(now["dimensions"][name]) - set(before[name])
            removed = set(before[name]) - set(now["dimensions"][name])
            if added:
                changes.append(f"{name}: valor(es) novo(s) {', '.join(sorted(added))}")
            if removed:
                changes.append(f"{name}: valor(es) sumiram {', '.join(sorted(removed))}")
        if now["ratios"] != previous.get("ratios", {}):
            changes.append("razões mudaram")
        if now["time_column"] != previous.get("time_column"):
            changes.append("coluna de tempo mudou")
        return changes


# --- Descoberta ---------------------------------------------------------------------------------


def _valid_value(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_VALUE_LENGTH
        and clean_text(value, MAX_VALUE_LENGTH) == value
    )


def _is_text(column: pd.Series) -> bool:
    return column.dtype == object or pd.api.types.is_string_dtype(column)


def _distinct(column: pd.Series) -> tuple[list[object], bool]:
    """Valores distintos e se há vazio (uma passada só: em texto, `isna` linha a linha é lento)."""
    values = list(column.unique())
    has_na = any(v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA for v in values)
    return [v for v in values if not (v is None or v is pd.NA or isinstance(v, float))], has_na


def _is_iso_date(distinct: list[object]) -> bool:
    """Todos os valores no formato AAAA-MM-DD."""
    if not distinct or not all(isinstance(v, str) and len(v) == 10 for v in distinct):
        return False
    parsed = pd.to_datetime(pd.Series(distinct, dtype=object), format="%Y-%m-%d", errors="coerce")
    return bool(parsed.notna().all())


def _rate_pairs(frame: pd.DataFrame, measures: list[str]) -> list[tuple[str, str]]:
    """Taxas: duas contagens inteiras com numerador <= denominador em todas as linhas.

    Só contagens: receita (valor) quase sempre passa de conversões, e isso não faz dela uma taxa.
    """
    pairs = []
    counts = [m for m in measures if pd.api.types.is_integer_dtype(frame[m])]
    for num in counts:
        for den in counts:
            if num == den:
                continue
            a, b = frame[num].to_numpy(float), frame[den].to_numpy(float)
            if (a < 0).any() or (b < 0).any() or b.sum() <= 0 or a.sum() <= 0:
                continue
            if bool(np.all(a <= b)):
                pairs.append((num, den))
    return pairs


def profile_from_frame(frame: pd.DataFrame, table: str = TABLE) -> SchemaProfile:
    """Perfil da tabela já carregada."""
    if frame.empty:
        raise ProfileError("a tabela events está vazia")
    columns = [str(c) for c in frame.columns]
    for name in columns:
        if not NAME.match(name):
            raise ProfileError("nome de coluna fora do padrão (minúsculas, dígitos e _)")

    text = {
        c: _distinct(frame[c])
        for c in columns
        if _is_text(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    }
    time_candidates = [c for c, (values, na) in text.items() if not na and _is_iso_date(values)]
    if not time_candidates:
        raise ProfileError("nenhuma coluna de data (AAAA-MM-DD) encontrada")
    time_column = next((c for c in TIME_NAMES if c in time_candidates), time_candidates[0])

    dimensions: dict[str, tuple[str, ...]] = {}
    measures: list[str] = []
    skipped: list[SkippedColumn] = []
    for name in columns:
        if name == time_column:
            continue
        column = frame[name]
        if pd.api.types.is_bool_dtype(column):
            skipped.append(SkippedColumn(column=name, reason="booleana"))
        elif pd.api.types.is_numeric_dtype(column):
            measures.append(name)
        elif name in text:
            distinct, has_na = text[name]
            if len(distinct) > MAX_DIMENSION_VALUES:
                skipped.append(
                    SkippedColumn(column=name, reason="cardinalidade alta", distinct=len(distinct))
                )
            elif has_na or not all(_valid_value(v) for v in distinct):
                skipped.append(SkippedColumn(column=name, reason="valores vazios ou inválidos"))
            elif len(distinct) < 1:
                skipped.append(SkippedColumn(column=name, reason="vazia"))
            else:
                dimensions[name] = tuple(sorted(str(v) for v in distinct))
        else:
            skipped.append(SkippedColumn(column=name, reason="tipo não suportado"))

    ratios: dict[str, Ratio] = {}
    rate_pairs = set(_rate_pairs(frame, measures)) if measures else set()
    known_pairs = set()
    for name, (num, den) in KNOWN_RATIOS.items():
        if num in measures and den in measures and name not in columns:
            ratios[name] = Ratio(
                numerator=num, denominator=den, is_rate=(num, den) in rate_pairs, source="known"
            )
            known_pairs.add((num, den))
    for num, den in sorted(rate_pairs - known_pairs):
        name = f"{num}_per_{den}"
        if NAME.match(name) and name not in columns:
            ratios[name] = Ratio(numerator=num, denominator=den, is_rate=True, source="discovered")

    return SchemaProfile(
        table=table,
        time_column=time_column,
        dimensions=dimensions,
        measures=tuple(measures),
        ratios=ratios,
        skipped=tuple(skipped),
    )


def load_events(con: sqlite3.Connection, table: str = TABLE) -> pd.DataFrame:
    """Tabela inteira (somente leitura). O nome da tabela é uma constante validada."""
    if not NAME.match(table):
        raise ProfileError("nome de tabela inválido")
    return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def discover_profile(con: sqlite3.Connection, table: str = TABLE) -> SchemaProfile:
    return profile_from_frame(load_events(con, table), table)


def propose_ratio(
    profile: SchemaProfile, frame: pd.DataFrame, name: str, numerator: str, denominator: str
) -> SchemaProfile:
    """Valida uma razão proposta (pelo LLM) e devolve o perfil com ela."""
    for column in (numerator, denominator):
        if column not in profile.measures:
            raise ProfileError(f"{column} não é uma medida aditiva")
    den = frame[denominator].to_numpy(float)
    num = frame[numerator].to_numpy(float)
    if den.sum() <= 0:
        raise ProfileError("o denominador soma zero: a razão não é bem definida")
    is_rate = bool((num >= 0).all() and (den >= 0).all() and np.all(num <= den))
    ratio = Ratio(numerator=numerator, denominator=denominator, is_rate=is_rate, source="proposed")
    return profile.with_ratio(name, ratio)


def propose_dimension(profile: SchemaProfile, frame: pd.DataFrame, column: str) -> SchemaProfile:
    """Valida uma dimensão proposta: coluna de texto que ficou de fora só pela cardinalidade."""
    skipped = {s.column: s for s in profile.skipped}
    if column not in skipped:
        raise ProfileError(f"{column} não é uma coluna de texto disponível")
    if skipped[column].reason != "cardinalidade alta":
        raise ProfileError(f"{column} não pode ser dimensão ({skipped[column].reason})")
    values, has_na = _distinct(frame[column])
    if has_na or not all(_valid_value(v) for v in values):
        raise ProfileError(f"{column} tem valores vazios ou inválidos")
    if len(values) > HARD_MAX_DIMENSION_VALUES:
        raise ProfileError(f"{column} tem valores demais para uma dimensão")
    return profile.with_dimension(column, tuple(sorted(str(v) for v in values)))


# Perfil do gerador sintético: o mesmo que `discover_profile` acha num banco gerado.
DEFAULT_PROFILE = SchemaProfile(
    time_column="day",
    dimensions={name: tuple(sorted(values)) for name, values in DIMENSION_VALUES.items()},
    measures=("sessions", "conversions", "revenue"),
    ratios={
        "conversion_rate": Ratio(
            numerator="conversions", denominator="sessions", is_rate=True, source="known"
        ),
        "avg_ticket": Ratio(
            numerator="revenue", denominator="conversions", is_rate=False, source="known"
        ),
    },
)
