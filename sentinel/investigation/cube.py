"""Cubo de eventos em memória: códigos inteiros por dimensão e somas diárias por fatia.

Carrega a tabela `events` uma vez (somente leitura) e responde às consultas do drill-down
sem voltar ao banco. Tolera buracos (dia sem linhas vira zero) e guarda o que encontrou,
para que `check_data_quality` possa relatar.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sentinel.schemas import DIMENSION_VALUES

VALUE_COLUMNS = ("sessions", "conversions", "revenue")


@dataclass
class Cube:
    start: date
    n_days: int
    day_idx: np.ndarray
    codes: dict[str, np.ndarray]
    values: dict[str, np.ndarray]
    missing_days: list[date] = field(default_factory=list)
    duplicate_rows: int = 0
    null_values: int = 0

    @property
    def n_rows(self) -> int:
        return len(self.day_idx)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=self.n_days - 1)

    # --- Construção --------------------------------------------------------------------------

    @classmethod
    def from_frame(cls, events: pd.DataFrame) -> "Cube":
        if events.empty:
            raise ValueError("a tabela events está vazia")
        days = pd.to_datetime(events["day"])
        first = days.min()
        n_days = int((days.max() - first).days) + 1
        day_idx = (days - first).dt.days.to_numpy()

        codes: dict[str, np.ndarray] = {}
        for name, allowed in DIMENSION_VALUES.items():
            coded = pd.Categorical(events[name], categories=list(allowed)).codes.astype(np.int64)
            if (coded < 0).any():
                raise ValueError(f"valor desconhecido na dimensão {name}")
            codes[name] = coded

        nulls = int(events[list(VALUE_COLUMNS)].isna().sum().sum())
        values = {name: events[name].fillna(0).to_numpy(dtype=float) for name in VALUE_COLUMNS}

        key = day_idx.astype(np.int64)
        for name, allowed in DIMENSION_VALUES.items():
            key = key * len(allowed) + codes[name]
        duplicates = len(key) - len(np.unique(key))

        present = np.zeros(n_days, dtype=bool)
        present[day_idx] = True
        start = first.date()
        missing = [start + timedelta(days=int(i)) for i in np.flatnonzero(~present)]
        return cls(start, n_days, day_idx, codes, values, missing, duplicates, nulls)

    @classmethod
    def from_connection(cls, con: sqlite3.Connection) -> "Cube":
        columns = ", ".join(["day", *DIMENSION_VALUES, *VALUE_COLUMNS])
        return cls.from_frame(pd.read_sql_query(f"SELECT {columns} FROM events", con))

    # --- Consulta ----------------------------------------------------------------------------

    def pos(self, day: date) -> int:
        position = (day - self.start).days
        if not 0 <= position < self.n_days:
            raise ValueError(f"{day} fora dos dados ({self.start} a {self.end})")
        return position

    def mask(self, dimensions: dict[str, str]) -> np.ndarray:
        selected = np.ones(self.n_rows, dtype=bool)
        for name, value in dimensions.items():
            selected &= self.codes[name] == DIMENSION_VALUES[name].index(value)
        return selected

    def subset(self, keep: np.ndarray) -> "Cube":
        return Cube(
            start=self.start,
            n_days=self.n_days,
            day_idx=self.day_idx[keep],
            codes={n: c[keep] for n, c in self.codes.items()},
            values={n: v[keep] for n, v in self.values.items()},
            missing_days=self.missing_days,
            duplicate_rows=self.duplicate_rows,
            null_values=self.null_values,
        )

    def daily(self, keys: np.ndarray, n_keys: int) -> dict[str, np.ndarray]:
        """Soma diária de cada coluna por fatia: matrizes (dias × fatias)."""
        flat = self.day_idx * n_keys + keys
        size = self.n_days * n_keys
        return {
            name: np.bincount(flat, weights=weights, minlength=size).reshape(self.n_days, n_keys)
            for name, weights in self.values.items()
        }
