"""Manifesto de um plugin do Switchboard (``plugin.yaml``).

Um plugin é um pacote **declarativo**: dados (YAML e Markdown), nunca código. Ele
traz capacidades (conectores MCP, agentes A2A, bases de conhecimento) e o *know-how*
de usá-las (playbooks, no formato Agent Skills). Nada do pacote é executado pelo
router; o que o plugin precisa de quem instala (endereços, tokens) entra por
``config``, e os valores sensíveis nunca ficam no pacote.

Exemplo mínimo::

    name: acme-credito
    version: 1.0.0
    description: Simulação e análise de crédito da Acme.
    config:
      CREDITO_URL: {description: URL do servidor MCP de crédito}
      CREDITO_TOKEN: {description: Token do servidor, sensitive: true}
    connectors:
      - name: credito
        url: ${config.CREDITO_URL}
        auth_token: ${config.CREDITO_TOKEN}
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import PLUGIN_NAME_RE, TransportKind
from .semver import is_semver, parse_specifier

API_VERSION = "switchboard.plugin/v1"

LOCAL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
CONFIG_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
PLACEHOLDER_RE = re.compile(r"\$\{config\.([A-Za-z0-9_]+)\}")
EXACT_PLACEHOLDER_RE = re.compile(r"^\$\{config\.([A-Za-z0-9_]+)\}$")

# nomes que fariam o plugin se passar pelo próprio Switchboard ou por terceiros
RESERVED_PREFIXES = ("switchboard", "anthropic", "claude", "typesafe", "jev")
RESERVED_NAMES = frozenset(
    {"core", "system", "admin", "builtin", "official", "router", "console", "default", "plugin"}
    # nomes de rotas do console e da API (/plugins/new, /api/plugins/usage…)
    | {"new", "usage", "events", "validate", "package"}
)

MANIFEST_FILES = ("plugin.yaml", "plugin.yml", "plugin.json")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _local_name(value: str) -> str:
    value = value.strip()
    if not LOCAL_NAME_RE.match(value):
        raise ValueError(
            "use só letras minúsculas, números, '-' ou '_' (até 32 caracteres), começando por "
            "letra ou número"
        )
    return value


def _http_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value and not value.startswith(("http://", "https://")):
        raise ValueError("a URL precisa começar com http:// ou https://")
    return value or None


MAX_URL_CHARS = 500  # o tamanho da coluna de endereço dos conectores e agentes
MAX_VERSION_CHARS = 40


def _endpoint_template(value: str) -> str:
    """Endereço de um componente: http(s) ou um ``${config.X}``, sem espaços nem controle."""
    value = value.strip()
    if not value:
        raise ValueError("informe o endereço (url)")
    if len(value) > MAX_URL_CHARS:
        raise ValueError(f"o endereço passa de {MAX_URL_CHARS} caracteres")
    if any(ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf") for ch in value):
        raise ValueError("o endereço não pode ter espaços nem caracteres de controle ou invisíveis")
    if not value.startswith(("http://", "https://", "${config.")):
        raise ValueError("a URL precisa começar com http://, https:// ou ${config.CHAVE}")
    return value


class ConfigField(_Strict):
    """Um valor que quem instala informa (endereço, token, região…)."""

    description: str = Field(default="", max_length=300)
    sensitive: bool = Field(
        default=False, description="segredo: vai cifrado (ou como env:NOME) e nunca é exibido"
    )
    required: bool = True
    default: str | None = None

    @field_validator("default", mode="before")
    @classmethod
    def _scalar(cls, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float)):
            return str(value)
        raise ValueError("o padrão precisa ser texto, número ou booleano")


class PluginAuthor(_Strict):
    name: str = Field(min_length=1, max_length=120)
    email: str | None = Field(default=None, max_length=200)
    url: str | None = None

    @field_validator("url")
    @classmethod
    def _url(cls, value: str | None) -> str | None:
        return _http_url(value)


class Requires(_Strict):
    switchboard: str | None = Field(
        default=None, description="versões do Switchboard compatíveis, ex.: '>=0.2,<1'"
    )

    @field_validator("switchboard")
    @classmethod
    def _spec(cls, value: str | None) -> str | None:
        if value is not None:
            parse_specifier(value)
        return value


class PluginConnector(_Strict):
    """Servidor MCP do plugin (vira o conector ``<plugin>.<name>``)."""

    name: str
    description: str = Field(default="", max_length=500)
    url: str
    transport: TransportKind = "streamable-http"
    auth_token: str | None = Field(
        default=None, description="só ${config.CHAVE} (uma chave 'sensitive')"
    )
    allowed_tools: list[str] = Field(default_factory=list, max_length=200)
    timeout_s: float = Field(default=30.0, gt=0, le=300)

    _name = field_validator("name")(_local_name)
    _url = field_validator("url")(_endpoint_template)


class PluginAgent(_Strict):
    """Agente A2A do plugin (vira o agente ``<plugin>.<name>``)."""

    name: str
    description: str = Field(default="", max_length=500)
    url: str
    auth_token: str | None = Field(
        default=None, description="só ${config.CHAVE} (uma chave 'sensitive')"
    )
    allowed_skills: list[str] = Field(default_factory=list, max_length=200)
    timeout_s: float = Field(default=15.0, gt=0, le=300)
    deadline_s: float | None = Field(default=None, gt=0, le=7 * 24 * 3600)
    push: bool = True
    allow_cross_origin: bool = False

    _name = field_validator("name")(_local_name)
    _url = field_validator("url")(_endpoint_template)


class PluginKnowledgeBase(_Strict):
    """Base de conhecimento do plugin, indexada na instalação a partir de arquivos do pacote."""

    name: str
    description: str = Field(default="", max_length=500)
    paths: list[str] = Field(default_factory=lambda: ["knowledge"], max_length=50)
    chunk_size: int = Field(default=800, ge=200, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    _name = field_validator("name")(_local_name)

    @field_validator("paths")
    @classmethod
    def _paths(cls, value: list[str]) -> list[str]:
        out = []
        for raw in value:
            path = raw.strip().strip("/")
            parts = path.split("/")
            if (
                not path
                or "\\" in path
                or raw.startswith("/")
                or any(p in ("", ".", "..") for p in parts)
            ):
                raise ValueError(f"caminho inválido: {raw!r} (use caminhos relativos do pacote)")
            out.append(path)
        return out


class PluginPlaybook(_Strict):
    """Como usar um playbook (skill) no roteamento. Tudo é opcional."""

    skill: str
    applies_to: list[str] = Field(
        default_factory=list,
        description="conectores/agentes do plugin a que o playbook se refere; sem eles disponíveis, não entra",
        max_length=50,
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="expressões que, no pedido, ativam o playbook (além da descrição da skill)",
        max_length=30,
    )
    always: bool = Field(default=False, description="entra em todos os pedidos")

    @field_validator("triggers")
    @classmethod
    def _triggers(cls, value: list[str]) -> list[str]:
        out = [t.strip() for t in value if t and t.strip()]
        if any(len(t) > 80 for t in out):
            raise ValueError("cada gatilho tem até 80 caracteres")
        return out


def _reserved(name: str) -> str | None:
    if name in RESERVED_NAMES:
        return f"'{name}' é um nome reservado"
    for prefix in RESERVED_PREFIXES:
        if name.startswith(prefix):
            return f"nomes que começam por '{prefix}' são reservados"
    return None


class PluginManifest(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_version: Literal["switchboard.plugin/v1"] = Field(default=API_VERSION, alias="schema")
    name: str
    version: str
    description: str = Field(min_length=1, max_length=500)
    author: PluginAuthor | None = None
    license: str | None = Field(default=None, max_length=64)
    homepage: str | None = None
    keywords: list[str] = Field(default_factory=list, max_length=10)
    requires: Requires = Field(default_factory=Requires)
    config: dict[str, ConfigField] = Field(default_factory=dict)
    connectors: list[PluginConnector] = Field(default_factory=list)
    agents: list[PluginAgent] = Field(default_factory=list)
    knowledge_bases: list[PluginKnowledgeBase] = Field(default_factory=list)
    skills: list[str] | None = Field(
        default=None, description="pastas em skills/ a usar (padrão: todas com SKILL.md)"
    )
    playbooks: list[PluginPlaybook] = Field(default_factory=list)
    routing_examples: dict[str, list[str]] = Field(
        default_factory=dict,
        description="'conector-ou-agente/capacidade' -> frases de exemplo, que ajudam a decidir",
    )

    @field_validator("author", mode="before")
    @classmethod
    def _author(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if not PLUGIN_NAME_RE.match(value):
            raise ValueError(
                "use 2 a 30 caracteres: letras minúsculas, números e '-', começando por letra"
            )
        problem = _reserved(value)
        if problem:
            raise ValueError(problem)
        return value

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        value = value.strip()
        if not is_semver(value):
            raise ValueError("a versão segue o semver: MAJOR.MINOR.PATCH (ex.: 1.2.0)")
        if len(value) > MAX_VERSION_CHARS:
            raise ValueError(f"a versão tem no máximo {MAX_VERSION_CHARS} caracteres")
        return value

    @field_validator("homepage")
    @classmethod
    def _homepage(cls, value: str | None) -> str | None:
        return _http_url(value)

    @field_validator("keywords")
    @classmethod
    def _keywords(cls, value: list[str]) -> list[str]:
        out = [k.strip() for k in value if k and k.strip()]
        if any(len(k) > 30 for k in out):
            raise ValueError("cada palavra-chave tem até 30 caracteres")
        return out

    @field_validator("config")
    @classmethod
    def _config_keys(cls, value: dict[str, ConfigField]) -> dict[str, ConfigField]:
        for key in value:
            if not CONFIG_KEY_RE.match(key):
                raise ValueError(
                    f"chave de config '{key}' inválida: use MAIÚSCULAS, números e '_' (ex.: CREDITO_URL)"
                )
        return value

    @field_validator("skills")
    @classmethod
    def _skills(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out = [s.strip().strip("/") for s in value]
        if any(not s or "/" in s or s in (".", "..") for s in out):
            raise ValueError("skills lista nomes de pastas de skills/ (sem caminhos)")
        return out

    @model_validator(mode="after")
    def _consistency(self) -> PluginManifest:
        owners: dict[str, str] = {}
        for kind, items in (
            ("conector", self.connectors),
            ("agente", self.agents),
            ("base", self.knowledge_bases),
        ):
            seen: set[str] = set()
            for item in items:
                if item.name in seen:
                    raise ValueError(f"{kind} '{item.name}' declarado mais de uma vez")
                seen.add(item.name)
                if kind != "base":
                    # conector e agente compartilham o dono "plugin.nome" nas chaves de capacidade
                    if item.name in owners:
                        raise ValueError(
                            f"'{item.name}' é usado por um {owners[item.name]} e por um {kind}: "
                            "conectores e agentes precisam de nomes diferentes dentro do plugin"
                        )
                    owners[item.name] = kind
        for pb in self.playbooks:
            for owner in pb.applies_to:
                if owner not in owners:
                    raise ValueError(
                        f"playbook '{pb.skill}': applies_to '{owner}' não é um conector nem "
                        "um agente deste plugin"
                    )
        seen_skills: set[str] = set()
        for pb in self.playbooks:
            if pb.skill in seen_skills:
                raise ValueError(f"playbook '{pb.skill}' declarado mais de uma vez")
            seen_skills.add(pb.skill)
        for key, examples in self.routing_examples.items():
            owner, slash, capability = key.partition("/")
            if not slash or not capability or owner not in owners:
                raise ValueError(
                    f"routing_examples: '{key}' precisa ser 'conector-ou-agente/capacidade' "
                    "de um componente deste plugin"
                )
            if len(examples) > 8:
                raise ValueError(f"routing_examples '{key}': até 8 exemplos")
            if any(not isinstance(e, str) or not e.strip() or len(e) > 300 for e in examples):
                raise ValueError(f"routing_examples '{key}': frases de 1 a 300 caracteres")
        return self

    # -- consultas -----------------------------------------------------------

    @property
    def owners(self) -> dict[str, str]:
        """Nome local -> ``connector`` | ``agent``."""
        out = {c.name: "connector" for c in self.connectors}
        out.update({a.name: "agent" for a in self.agents})
        return out

    def placeholders(self) -> set[str]:
        """Chaves de config usadas nas URLs e tokens do plugin."""
        used: set[str] = set()
        for item in (*self.connectors, *self.agents):
            for text in (item.url, item.auth_token):
                if text:
                    used.update(PLACEHOLDER_RE.findall(text))
        return used

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def url_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
