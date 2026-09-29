"""Leitura do Agent Card (A2A v1.0), tolerante com cards 0.3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import AgentError
from .protocol import A2A_VERSION, JSONRPC_BINDING


@dataclass(frozen=True)
class AgentInterface:
    url: str
    binding: str
    version: str


@dataclass(frozen=True)
class SkillCard:
    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ()


@dataclass
class AgentCard:
    name: str
    description: str
    version: str
    interfaces: list[AgentInterface]
    push: bool
    streaming: bool
    extensions: list[dict[str, Any]]
    skills: list[SkillCard]
    default_input_modes: list[str] = field(default_factory=list)
    default_output_modes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def extension(self, uri: str) -> dict[str, Any] | None:
        return next((e for e in self.extensions if e.get("uri") == uri), None)

    def jsonrpc_interface(self) -> AgentInterface | None:
        """A interface JSON-RPC 1.x (a que o cliente do Switchboard fala)."""
        candidates = [i for i in self.interfaces if i.binding.upper() == JSONRPC_BINDING]
        for interface in candidates:
            if interface.version.split(".")[0] == A2A_VERSION.split(".")[0]:
                return interface
        return None

    @classmethod
    def parse(cls, data: Any) -> AgentCard:
        if not isinstance(data, Mapping):
            raise AgentError("o Agent Card precisa ser um objeto JSON")
        name = str(data.get("name") or "").strip()
        if not name:
            raise AgentError("Agent Card sem o campo 'name'")
        interfaces = []
        for item in data.get("supportedInterfaces") or []:
            if isinstance(item, Mapping) and item.get("url"):
                interfaces.append(
                    AgentInterface(
                        url=str(item["url"]),
                        binding=str(item.get("protocolBinding") or JSONRPC_BINDING),
                        version=str(item.get("protocolVersion") or A2A_VERSION),
                    )
                )
        if not interfaces and data.get("url"):  # card 0.3: url + preferredTransport
            interfaces.append(
                AgentInterface(
                    url=str(data["url"]),
                    binding=str(data.get("preferredTransport") or JSONRPC_BINDING),
                    version=str(data.get("protocolVersion") or "0.3"),
                )
            )
        caps = data.get("capabilities") if isinstance(data.get("capabilities"), Mapping) else {}
        extensions = [dict(e) for e in caps.get("extensions") or [] if isinstance(e, Mapping)]
        skills = []
        for raw in data.get("skills") or []:
            if not isinstance(raw, Mapping) or not raw.get("id"):
                continue
            skills.append(
                SkillCard(
                    id=str(raw["id"]),
                    name=str(raw.get("name") or raw["id"]),
                    description=str(raw.get("description") or "").strip(),
                    tags=tuple(str(t) for t in raw.get("tags") or ()),
                    examples=tuple(str(e) for e in raw.get("examples") or ()),
                    input_modes=tuple(str(m) for m in raw.get("inputModes") or ()),
                    output_modes=tuple(str(m) for m in raw.get("outputModes") or ()),
                )
            )
        return cls(
            name=name,
            description=str(data.get("description") or "").strip(),
            version=str(data.get("version") or ""),
            interfaces=interfaces,
            push=bool(caps.get("pushNotifications")),
            streaming=bool(caps.get("streaming")),
            extensions=extensions,
            skills=skills,
            default_input_modes=[str(m) for m in data.get("defaultInputModes") or []],
            default_output_modes=[str(m) for m in data.get("defaultOutputModes") or []],
            raw=dict(data),
        )
