"""Configuração do serviço router (variáveis de ambiente ``SWITCHBOARD_*``)."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWITCHBOARD_", env_file=".env", extra="ignore")

    # origem da configuração: banco (console) ou arquivo YAML (modo framework)
    database_url: str = "postgresql://switchboard:switchboard@localhost:5432/switchboard"
    config_file: str | None = None
    vector_backend: Literal["auto", "pgvector", "json"] = "auto"

    secret_key: str | None = None
    secret_key_file: str | None = None  # mesmo arquivo do console (gerado na primeira subida)
    allowed_env_secrets: str = "*_API_KEY,MCP_*"  # variáveis aceitas em env:NOME
    api_keys: str = ""  # lista separada por vírgula; vazio = API aberta
    # com a API aberta (sem api_keys), só estes nomes de host são atendidos (contra DNS rebinding)
    allowed_hosts: str = "localhost,127.0.0.1,[::1],router"

    default_profile: str = "default"
    config_ttl_s: float = 5.0  # quanto tempo o router reaproveita a config lida do banco
    agents_ttl_s: float = 30.0  # cache da descoberta MCP (tools/list)

    host: str = "127.0.0.1"  # a imagem Docker usa 0.0.0.0 (SWITCHBOARD_HOST)
    port: int = 8080
    log_level: str = "info"

    @property
    def allowed_host_list(self) -> list[str]:
        return [h.strip().lower() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]
