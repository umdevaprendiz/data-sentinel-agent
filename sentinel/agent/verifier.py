"""Verificador (seção 9.6 do CLAUDE.md): o código confere o que o LLM escreveu.

O relatório só passa se todo número vier de um placeholder, todo id existir, os fatos só falarem
das causas confirmadas, o status bater com as causas e nenhum texto contrariar a direção medida.
Limite conhecido: números por extenso ("três") e negações ("não caiu") não são vistos.
"""

import re
from typing import Literal

from pydantic import BaseModel

from sentinel.agent.placeholders import (
    PlaceholderError,
    finding_ids_in,
    resolve,
    strip_placeholders,
    tokens,
)
from sentinel.schemas import DIMENSION_VALUES, Finding, Report

Code = Literal[
    "loose_number",
    "bad_placeholder",
    "unknown_finding",
    "foreign_dimension",
    "incoherent_status",
    "unconfirmed_cause",
    "contradiction",
]

MAX_CAUSES = 2
_UP = re.compile(r"\b(subiu|subiram|aumentou|aumentaram|cresceu|cresceram|disparou)\b")
_DOWN = re.compile(
    r"\b(caiu|cairam|caíram|diminuiu|diminuíram|reduziu|reduziram|despencou|recuou|queda)\b"
)
_VERSIONS = sorted(
    (v for values in DIMENSION_VALUES.values() for v in values if any(c.isdigit() for c in v)),
    key=len,
    reverse=True,
)
_VERSION = re.compile(r"(?<![\d.])(?:" + "|".join(re.escape(v) for v in _VERSIONS) + r")(?!\.?\d)")


class Problem(BaseModel):
    code: Code
    message: str


class Verification(BaseModel):
    problems: list[Problem] = []

    @property
    def ok(self) -> bool:
        return not self.problems


def _loose_number(text: str) -> str | None:
    """Trecho com dígito fora de placeholder; versões conhecidas (`3.2`) não contam."""
    bare = _VERSION.sub(" ", strip_placeholders(text))
    found = re.search(r"\S*\d\S*", bare)
    return found.group(0) if found else None


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def _direction_words(text: str) -> set[str]:
    lowered = text.lower()
    words = set()
    if _UP.search(lowered):
        words.add("spike")
    if _DOWN.search(lowered):
        words.add("drop")
    return words


class _Check:
    def __init__(self, report: Report, findings: list[Finding]) -> None:
        self.report = report
        self.by_id = {f.id: f for f in findings}
        self.problems: list[Problem] = []
        causes = report.causes
        self.allowed = {name for c in causes for name in c.dimensions} | {
            value for c in causes for value in c.dimensions.values()
        }
        every_name = set(DIMENSION_VALUES) | {v for vs in DIMENSION_VALUES.values() for v in vs}
        self.foreign = sorted(every_name - self.allowed)

    def add(self, code: Code, message: str) -> None:
        self.problems.append(Problem(code=code, message=message))

    def known_id(self, where: str, finding_id: str) -> None:
        if finding_id not in self.by_id:
            self.add("unknown_finding", f"{where}: o Finding {finding_id} não existe")

    def reference_direction(self, cited: list[str]) -> str | None:
        """Direção medida nos fatos citados (ou em todos, se o texto não cita nenhum)."""

        def directions(ids: list[str]) -> set[str]:
            return {
                d
                for i in ids
                if i in self.by_id and (d := self.by_id[i].result.get("direction")) is not None
            }

        found = directions(cited) or directions(list(self.by_id))
        return next(iter(found)) if len(found) == 1 else None

    def text(self, where: str, text: str, cited: list[str], *, is_fact: bool) -> None:
        loose = _loose_number(text)
        if loose:
            self.add("loose_number", f"{where}: número fora de placeholder ({loose!r})")

        for token in tokens(text):
            try:
                resolve(token, self.by_id)
            except PlaceholderError as error:
                self.add("bad_placeholder", f"{where}: {error}")

        if is_fact:
            lowered = strip_placeholders(text).lower()
            for word in self.foreign:
                if _mentions(lowered, word):
                    self.add(
                        "foreign_dimension",
                        f"{where}: cita {word!r}, que não está nas causas confirmadas",
                    )

        reference = self.reference_direction([*cited, *finding_ids_in(text)])
        words = _direction_words(strip_placeholders(text))
        if reference is not None and words and reference not in words:
            self.add(
                "contradiction",
                f"{where}: o texto diz {'/'.join(sorted(words))}, a medição diz {reference}",
            )

    def status(self) -> None:
        report = self.report
        count = len(report.causes)
        if report.status == "identified" and not 1 <= count <= MAX_CAUSES:
            self.add("incoherent_status", f"identified exige de 1 a {MAX_CAUSES} causas ({count})")
        if report.status == "inconclusive" and (count or report.proposed_action):
            self.add("incoherent_status", "inconclusive não pode ter causa nem ação proposta")
        if not report.facts:
            self.add("incoherent_status", "o relatório não tem nenhum fato")

    def causes(self) -> None:
        for number, cause in enumerate(self.report.causes, start=1):
            for finding_id in cause.finding_ids:
                self.known_id(f"causa {number}", finding_id)
            proofs = [
                self.by_id[i]
                for i in cause.finding_ids
                if i in self.by_id and self.by_id[i].tool == "test_hypothesis"
            ]
            if not any(
                p.result.get("accepted") is True
                and cause.dimensions in (p.result.get("dimensions"), p.result.get("together_with"))
                for p in proofs
            ):
                self.add(
                    "unconfirmed_cause",
                    f"causa {number} ({cause.dimensions}): sem test_hypothesis aceito",
                )

    def run(self) -> Verification:
        report = self.report
        self.status()
        self.causes()
        for finding_id in report.evidence:
            self.known_id("evidência", finding_id)
        for number, fact in enumerate(report.facts, start=1):
            where = f"fato {number}"
            self.known_id(where, fact.finding_id)
            self.text(where, fact.text, [fact.finding_id], is_fact=True)
        for number, opinion in enumerate(report.opinions, start=1):
            where = f"opinião {number}"
            for finding_id in opinion.finding_ids:
                self.known_id(where, finding_id)
            self.text(where, opinion.text, opinion.finding_ids, is_fact=False)
        return Verification(problems=self.problems)


def verify(report: Report, findings: list[Finding]) -> Verification:
    """Confere o relatório contra os `Finding`s. `ok` só se nenhuma regra for violada."""
    return _Check(report, findings).run()
