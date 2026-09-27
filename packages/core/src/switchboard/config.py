"""Especificações de configuração (compartilhadas pelo modo YAML e pelo banco).

O console grava estas mesmas entidades no banco; o modo framework as lê de um
``switchboard.yaml``. O motor de roteamento só enxerga as specs, nunca a
origem, então os dois caminhos se comportam igual.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigError

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

DEFAULT_SYSTEM_PROMPT = (
    "Você é o assistente virtual da empresa. Responda em português do Brasil, com "
    "cordialidade e objetividade. Não invente fatos."
)

ProviderKind = Literal["openai", "anthropic", "offline"]
TransportKind = Literal["streamable-http", "sse"]


def _check_name(value: str) -> str:
    value = value.strip()
    if not NAME_RE.match(value):
        raise ValueError(
            "use só letras minúsculas, números, '-' ou '_' (até 63 caracteres), começando "
            "por letra ou número"
        )
    return value


Name = Annotated[str, AfterValidator(_check_name)]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Spec):
    """Conexão com um provedor de LLM (e, opcionalmente, de embeddings)."""

    name: Name
    provider: ProviderKind = "openai"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = Field(default=None, description="texto, env:NOME ou enc:...")
    api_key_header: str = "Authorization"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    temperature: float | None = 0.2
    max_tokens: int | None = 4096
    timeout_s: float = 60.0
    json_mode: bool = True

    @model_validator(mode="after")
    def _needs_model(self) -> ModelSpec:
        if self.provider != "offline" and not self.model.strip():
            raise ValueError(f"o modelo '{self.name}' precisa do campo 'model'")
        return self


class AgentSpec(_Spec):
    """Um agente acionável via MCP (um servidor MCP; suas tools são as capacidades)."""

    name: Name
    description: str = ""
    url: str
    transport: TransportKind = "streamable-http"
    auth_token: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    enabled: bool = True
    timeout_s: float = 30.0

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("a URL do agente MCP precisa começar com http:// ou https://")
        return value


class EmbedderSpec(_Spec):
    """Como os trechos de uma base de conhecimento viram vetores.

    ``hashing`` roda offline e é determinístico (bom para começar e para
    testes); ``model`` usa um endpoint ``/embeddings`` compatível com OpenAI,
    reaproveitando a URL e a chave de um :class:`ModelSpec` (``connection``).
    """

    kind: Literal["hashing", "model"] = "hashing"
    connection: str | None = None
    model: str | None = None
    dim: int = 512

    @model_validator(mode="after")
    def _check(self) -> EmbedderSpec:
        if self.kind == "model" and not (self.connection and self.model):
            raise ValueError("embedder 'model' precisa de 'connection' (nome do modelo) e 'model'")
        return self

    @property
    def label(self) -> str:
        if self.kind == "hashing":
            return f"hashing-{self.dim}"
        return f"{self.connection}:{self.model}"


class KnowledgeBaseSpec(_Spec):
    """Uma base de conhecimento consultada pelo RAG."""

    name: Name
    description: str = ""
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    chunk_size: int = Field(default=800, ge=200, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    paths: list[str] = Field(default_factory=list, description="só no modo YAML")


class ProfileSpec(_Spec):
    """Um roteador configurado: modelo + agentes + bases + comportamento."""

    name: Name = "default"
    description: str = ""
    model: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    agents: list[str] = Field(default_factory=list)
    knowledge_bases: list[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)
    synthesize: bool = True
    allow_clarify: bool = True


class SwitchboardSpec(_Spec):
    """Configuração completa do modo framework (``switchboard.yaml``)."""

    models: list[ModelSpec]
    agents: list[AgentSpec] = Field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseSpec] = Field(default_factory=list)
    profiles: list[ProfileSpec]

    @model_validator(mode="after")
    def _references(self) -> SwitchboardSpec:
        for kind, items in (
            ("modelo", self.models),
            ("agente", self.agents),
            ("base", self.knowledge_bases),
            ("perfil", self.profiles),
        ):
            seen: set[str] = set()
            for item in items:
                if item.name in seen:
                    raise ValueError(f"{kind} '{item.name}' declarado mais de uma vez")
                seen.add(item.name)
        models = {m.name for m in self.models}
        agents = {a.name for a in self.agents}
        kbs = {k.name for k in self.knowledge_bases}
        for kb in self.knowledge_bases:
            if kb.embedder.kind == "model" and kb.embedder.connection not in models:
                raise ValueError(
                    f"base '{kb.name}': embedder usa a conexão '{kb.embedder.connection}', "
                    "que não existe em models"
                )
        for p in self.profiles:
            if p.model not in models:
                raise ValueError(f"perfil '{p.name}': modelo '{p.model}' não existe em models")
            for a in p.agents:
                if a not in agents:
                    raise ValueError(f"perfil '{p.name}': agente '{a}' não existe em agents")
            for k in p.knowledge_bases:
                if k not in kbs:
                    raise ValueError(f"perfil '{p.name}': base '{k}' não existe em knowledge_bases")
        return self

    def model(self, name: str) -> ModelSpec:
        return next(m for m in self.models if m.name == name)

    def profile(self, name: str | None) -> ProfileSpec:
        if name is None:
            return self.profiles[0]
        for p in self.profiles:
            if p.name == name:
                return p
        raise ConfigError(f"perfil '{name}' não existe")


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(text: str) -> str:
    """Troca ``${VAR}`` e ``${VAR:-padrao}`` pelo valor do ambiente."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise ConfigError(f"variável de ambiente {name} não definida (usada no YAML)")
            return default
        return value

    return _ENV_PATTERN.sub(repl, text)


def _interpolate_tree(value):
    if isinstance(value, str):
        return _interpolate(value)
    if isinstance(value, list):
        return [_interpolate_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_tree(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> SwitchboardSpec:
    """Lê e valida um ``switchboard.yaml``."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"não consegui ler {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML inválido: {exc}") from exc
    data = _interpolate_tree(data)  # só nos valores: comentários ficam de fora
    try:
        return SwitchboardSpec.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"{path}: configuração inválida:\n{format_validation_error(exc)}"
        ) from exc


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(raiz)"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)
