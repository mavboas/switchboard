"""Configuração do serviço router (variáveis de ambiente ``SWITCHBOARD_*``)."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWITCHBOARD_", env_file=".env", extra="ignore")

    # origem da configuração: banco (console) ou arquivo YAML (modo framework)
    database_url: str = "postgresql://switchboard:switchboard@localhost:5432/switchboard"
    config_file: str | None = None
    vector_backend: Literal["auto", "pgvector", "json"] = "auto"

    secret_key: str | None = None
    secret_key_file: str | None = None  # mesmo arquivo do console (gerado na primeira subida)
    allowed_env_secrets: str = "*_API_KEY,MCP_*,A2A_*"  # variáveis aceitas em env:NOME
    api_keys: str = ""  # lista separada por vírgula; vazio = API aberta
    # com a API aberta (sem api_keys), só estes nomes de host são atendidos (contra DNS rebinding)
    allowed_hosts: str = "localhost,127.0.0.1,[::1],router"

    default_profile: str = "default"
    config_ttl_s: float = 5.0  # quanto tempo o router reaproveita a config lida do banco
    agents_ttl_s: float = (
        30.0  # cache da descoberta (tools/list dos conectores, Agent Card dos agentes)
    )

    # delegação a agentes A2A
    # URL pela qual os agentes alcançam este router (push notifications em /a2a/push/...).
    # Vazia = sem push: o router acompanha os contratos só por polling.
    public_url: str | None = None
    max_wait_s: float = 60.0  # teto para o wait_s pedido pelo cliente
    supervisor_tick_s: float = 1.0  # frequência do supervisor de contratos (polling e prazos)
    # hosts aceitos em callback_url (webhook do cliente); vazio = callbacks desligados
    callback_hosts: str = ""
    max_push_bytes: int = 1_000_000

    host: str = "127.0.0.1"  # a imagem Docker usa 0.0.0.0 (SWITCHBOARD_HOST)
    port: int = 8080
    log_level: str = "info"

    @property
    def allowed_host_list(self) -> list[str]:
        hosts = [h.strip().lower() for h in self.allowed_hosts.split(",") if h.strip()]
        public = urlsplit(self.public_url).hostname if self.public_url else None
        if public and public.lower() not in hosts:
            hosts.append(public.lower())  # os agentes chamam o push por este nome
        return hosts

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def callback_host_list(self) -> set[str]:
        return {h.strip().lower() for h in self.callback_hosts.split(",") if h.strip()}
