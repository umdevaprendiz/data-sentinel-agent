"""Contratos de dados do projeto (seção 6 do CLAUDE.md)."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

# Métricas e dimensões são descobertas nos dados (decisão 21, `data/profile.py`): aqui só o formato
# do nome é fixo. As listas do gerador sintético ficam em `DIMENSION_VALUES`.
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,40}$"
Metric = Annotated[str, Field(pattern=NAME_PATTERN)]
Dimension = Annotated[str, Field(pattern=NAME_PATTERN)]
KNOWN_METRICS = ("sessions", "conversions", "conversion_rate", "avg_ticket", "revenue")
ChangeType = Literal["deploy", "price_change", "campaign_start", "campaign_end", "stock_out"]

# Dimensões do gerador sintético (a ordem das chaves é a ordem das colunas no banco).
DIMENSION_VALUES: dict[str, tuple[str, ...]] = {
    "region": ("norte", "nordeste", "centro_oeste", "sudeste", "sul"),
    "platform": ("android", "ios"),
    "app_version": ("3.0", "3.1", "3.2"),
    "product": ("laptop", "phone", "tablet", "watch", "headset", "charger"),
    "channel": ("organic", "paid_search", "social", "email"),
}


def _check_dimensions(dimensions: dict[str, str]) -> None:
    for name, value in dimensions.items():
        if name not in DIMENSION_VALUES:
            raise ValueError(f"dimensão desconhecida: {name!r}")
        if value not in DIMENSION_VALUES[name]:
            raise ValueError(f"valor inválido para {name}: {value!r}")


# --- Investigação e relatório -------------------------------------------------------------


class Anomaly(BaseModel):
    metric: Metric
    window_start: date
    window_end: date
    onset: date  # início estimado
    observed: float
    expected: float
    z_score: float  # robusto (MAD)
    direction: Literal["drop", "spike"]


class Finding(BaseModel):
    """Fato apurado por uma consulta. Todo número do relatório vem daqui."""

    id: str  # "f001"
    tool: str
    params: dict
    result: dict


class Hypothesis(BaseModel):
    dimensions: dict[str, str]  # {"platform": "android", "app_version": "3.2"}
    explained_share: float  # 0..1
    finding_ids: list[str]


class Statement(BaseModel):
    text: str  # só texto e placeholders {fNNN.campo}
    finding_id: str


class Opinion(BaseModel):
    kind: Literal["cause", "recommendation", "caveat"]
    text: str
    confidence: float = Field(ge=0, le=1)  # declarada pelo modelo
    finding_ids: list[str]


class ActionProposal(BaseModel):
    action_id: str  # sempre do catálogo
    params: dict[str, str]
    reversible: bool
    risk: Literal["low", "medium", "high"]


class Report(BaseModel):
    status: Literal["identified", "inconclusive"]
    facts: list[Statement]  # camada 1: verificada
    causes: list[Hypothesis]  # 1 ou 2; vazio se inconclusive
    opinions: list[Opinion]  # camada 2: opinião, não verificada
    evidence: list[str]  # ids de Finding
    proposed_action: ActionProposal | None = None


# --- Gerador de dados e cenários ------------------------------------------------------------


class Cause(BaseModel):
    """Efeito injetado numa fatia dos dados durante um período."""

    dimensions: dict[str, str]  # {} = afeta tudo (queda difusa)
    start: date
    end: date | None = None  # None = até o fim dos dados
    ramp_days: int = Field(default=0, ge=0)  # 0 = degrau; >0 = deriva linear até o fator final
    sessions_factor: float = Field(default=1.0, ge=0)
    conversion_factor: float = Field(default=1.0, ge=0)  # 0 = conversão zerada
    price_factor: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _check_dimensions(self.dimensions)
        if self.end is not None and self.end < self.start:
            raise ValueError("end antes de start")
        return self


class ChangeRecord(BaseModel):
    """Linha da tabela `changes`."""

    ts: date
    type: ChangeType
    description: str
    scope: dict[str, str]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _check_dimensions(self.scope)
        return self


class Scenario(BaseModel):
    id: str
    kind: str
    description: str
    anomaly_expected: bool
    expected_status: Literal["identified", "inconclusive", "none"]
    causes: list[Cause]  # vazio = sem causa
    changes: list[ChangeRecord]  # o que fica registrado em `changes` (pode omitir ou ser isca)
    window: tuple[date, date]
    noise_level: float = Field(default=1.0, gt=0)
    weekly_amplitude: float = Field(default=0.15, ge=0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.anomaly_expected != (self.expected_status != "none"):
            raise ValueError("expected_status 'none' só combina com anomaly_expected=False")
        if self.window[0] > self.window[1]:
            raise ValueError("window invertida")
        return self
