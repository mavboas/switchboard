"""Configuração do console (variáveis de ambiente ``SWITCHBOARD_*``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_KNOWLEDGE = Path(__file__).resolve().parents[4] / "examples" / "knowledge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWITCHBOARD_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://switchboard:switchboard@localhost:5432/switchboard"
    vector_backend: Literal["auto", "pgvector", "json"] = "auto"
    secret_key: str | None = None

    # onde está o router (usado pelo playground e pelo painel)
    router_url: str = "http://localhost:8080"
    router_public_url: str | None = None  # URL mostrada nos exemplos de uso (padrão: router_url)
    router_api_key: str | None = None

    # acesso ao console (HTTP Basic); sem senha o console fica aberto
    console_user: str = "admin"
    console_password: str | None = None

    # dados de demonstração na primeira subida (banco vazio)
    seed_demo: bool = True
    demo_knowledge_dir: str = str(_DEFAULT_KNOWLEDGE)
    demo_credito_url: str = "http://localhost:8101/mcp"
    demo_chamados_url: str = "http://localhost:8102/mcp"

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_upload_mb: int = 20

    @property
    def public_router_url(self) -> str:
        return (self.router_public_url or self.router_url).rstrip("/")
