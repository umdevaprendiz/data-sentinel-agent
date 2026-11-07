"""Configuração lida do `.env` (seção 14 do CLAUDE.md)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_timeout_s: float = 300
    llm_seed: int = 0  # o modelo roda com temperatura 0 e esta semente
    llm_cache: bool = (
        False  # True = repete a resposta guardada para o mesmo (modelo, prompt, semente)
    )

    # Modo e autonomia
    plan_mode: Literal["deterministic", "llm"] = "deterministic"
    autonomy_level: int = Field(default=2, ge=1, le=3)
    dry_run: bool = True
    kill_switch: bool = False  # também ligado pela presença de logs/KILL_SWITCH

    # Dados
    db_path: Path = Path("data/sentinel.db")
    seed: int = 42
    logs_dir: Path = Path("logs")  # audit, decisões, incidentes, checkpoints e simulador

    # Detector
    z_threshold: float = 3.5
    sustained_z_threshold: float = 5.5
    min_consecutive_days: int = 2
    baseline_weeks: int = 8
    window_days: int = 7
    min_slice_sessions: int = 200

    # Drill-down (seção 9.2)
    ep_single: float = 0.6
    ep_pair: float = 0.8
    ep_multi_each: float = 0.3
    ep_multi_total: float = 0.8
    z_stable: float = 3.5
    ep_material: float = 0.1
    refine_keep: float = 0.9
    min_total_z: float = 3.5

    # Investigação
    max_rounds: int = 5
    time_budget_s: float = 600
    sql_max_rows: int = 200
    sql_timeout_s: float = 5

    # Policy (L3)
    auto_min_ep: float = 0.8
    auto_min_precision: float = 0.9
    auto_min_history: int = 10
    auto_max_per_hour: int = 2

    # Memória (pgvector); DSN vazio = memória só em RAM, que não persiste
    memory_dsn: SecretStr = SecretStr("")
    embed_model: str = "nomic-embed-text"
    embed_dim: int = Field(default=768, ge=1)
    memory_top_k: int = Field(default=3, ge=1)
    memory_min_similarity: float = Field(default=0.75, ge=0, le=1)

    # Incidentes e Telegram
    approval_timeout_s: float = 3600
    cooldown_h: float = 24
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""

    def secret_values(self) -> list[str]:
        """Valores que nunca podem aparecer em log, traceback ou mensagem de erro."""
        dsn = self.memory_dsn.get_secret_value()
        values = [self.telegram_bot_token.get_secret_value(), dsn]
        try:
            password = urlsplit(dsn).password if dsn else None
        except ValueError:  # DSN malformado: o valor inteiro já está na lista
            password = None
        if password:
            values += [password, unquote(password)]
        return [v for v in values if v]


@lru_cache
def get_settings() -> Settings:
    return Settings()
