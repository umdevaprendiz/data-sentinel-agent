"""Configuração lida do `.env` (seção 14 do CLAUDE.md)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_timeout_s: float = 300

    # Modo e autonomia
    plan_mode: Literal["deterministic", "llm"] = "deterministic"
    autonomy_level: int = Field(default=2, ge=1, le=3)
    dry_run: bool = True

    # Dados
    db_path: Path = Path("data/sentinel.db")
    seed: int = 42

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

    # Incidentes e Telegram
    approval_timeout_s: float = 3600
    cooldown_h: float = 24
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
