"""Importação de plugins no formato do Claude (``.claude-plugin/plugin.json``).

O formato de plugin do Claude agrupa skills, comandos, subagentes, hooks e servidores
MCP. Só uma parte disso cabe num pacote declarativo do Switchboard, e é essa parte que
a importação traz:

* ``skills/`` (o mesmo formato Agent Skills) → playbooks;
* servidores MCP **remotos** (``http``/``sse``, de ``.mcp.json`` ou ``mcpServers``) →
  conectores, com o ``Authorization: Bearer ${...}`` virando ``auth_token`` e as
  variáveis (``${user_config.x}``, ``${VAR}``) virando ``config``;
* ``userConfig`` → ``config`` (``sensitive`` é mantido).

O que **não** vem, sempre dito num achado ``note``/``warning``: servidores MCP locais
(``command``/stdio), hooks, comandos, subagentes e scripts. O router nunca executa
código de plugin, e subagentes do Claude não são agentes A2A.

O resultado é um manifesto nativo (dicionário); quem valida e varre é o ``bundle``, igual
a um pacote nativo. O pacote guardado é sempre o nativo (``plugin.yaml`` gerado), então
reler o que foi instalado não passa por aqui de novo.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .findings import Finding
from .manifest import CONFIG_KEY_RE

CLAUDE_MANIFEST = ".claude-plugin/plugin.json"
CLAUDE_MCP_FILE = ".mcp.json"
CLAUDE_MARKETPLACE = ".claude-plugin/marketplace.json"

_VAR_RE = re.compile(r"\$\{(?:user_config\.)?([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_BEARER_RE = re.compile(r"^\s*Bearer\s+(\$\{[^}]+\})\s*$", re.IGNORECASE)
_LOCAL_RE = re.compile(r"[^a-z0-9_-]+")

# componentes do formato do Claude que o Switchboard não usa: pasta ou campo -> por quê
_UNSUPPORTED = {
    "commands": "comandos (slash commands) são atalhos para a pessoa digitar; o router recebe pedidos em linguagem natural",
    "agents": "subagentes do Claude são prompts, não agentes A2A (cadastre o agente A2A em 'agents')",
    "hooks": "hooks executam comandos na máquina; o router nunca executa código de plugin",
    "lspServers": "servidores de linguagem (LSP) não se aplicam a um roteador",
    "outputStyles": "estilos de saída não se aplicam a um roteador",
    "monitors": "monitores executam processos; o router nunca executa código de plugin",
}


@dataclass
class Converted:
    manifest: dict[str, Any] | None = None
    findings: list[Finding] = field(default_factory=list)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"chave repetida: {key!r}")
        out[key] = value
    return out


def parse_json(data: bytes, where: str) -> tuple[Any, list[Finding]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, [
            Finding("block", "manifest.encoding", "o arquivo precisa estar em UTF-8", where)
        ]
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates), []
    except (ValueError, RecursionError) as exc:
        return None, [
            Finding("block", "manifest.invalid", f"JSON inválido: {str(exc)[:160]}", where)
        ]


def config_key(raw: str) -> str:
    """``api_token`` -> ``API_TOKEN`` (as chaves de config do Switchboard são MAIÚSCULAS)."""
    key = re.sub(r"[^A-Za-z0-9_]", "_", raw).upper()
    return key if key and key[0].isalpha() else f"V_{key}"


def local_name(raw: str) -> str:
    name = _LOCAL_RE.sub("-", raw.strip().lower()).strip("-_")
    return name[:32].strip("-_")


class _Builder:
    def __init__(self, files: Mapping[str, bytes]):
        self.files = files
        self.findings: list[Finding] = []
        self.config: dict[str, dict[str, Any]] = {}
        self.connectors: list[dict[str, Any]] = []

    def note(self, code: str, message: str, where: str = "", level: str = "note") -> None:
        self.findings.append(Finding(level, code, message, where))  # type: ignore[arg-type]

    def declare(self, key: str, *, sensitive: bool = False, default: str | None = None) -> None:
        field_ = self.config.setdefault(key, {"description": "importado do plugin do Claude"})
        if sensitive:
            field_["sensitive"] = True
        if default is not None and "default" not in field_ and not field_.get("sensitive"):
            field_["default"] = default
            field_["required"] = False

    def rewrite(self, text: str, where: str, *, sensitive: bool = False) -> str:
        """Troca ``${user_config.x}`` e ``${VAR}`` por ``${config.X}``."""

        def repl(match: re.Match[str]) -> str:
            key = config_key(match.group(1))
            if match.group(1) == "CLAUDE_PLUGIN_ROOT":
                self.note(
                    "import.plugin_root",
                    "${CLAUDE_PLUGIN_ROOT} aponta para arquivos locais, que não existem aqui",
                    where,
                    "warning",
                )
            self.declare(key, sensitive=sensitive, default=match.group(2))
            return "${config." + key + "}"

        return _VAR_RE.sub(repl, text)

    # -- partes do manifesto ---------------------------------------------------

    def user_config(self, raw: Any) -> None:
        if raw is None:
            return
        where = f"{CLAUDE_MANIFEST}: userConfig"
        if not isinstance(raw, dict):
            self.note(
                "import.user_config", "userConfig precisa ser um mapa; ignorado", where, "warning"
            )
            return
        for name, spec in raw.items():
            key = config_key(str(name))
            if not CONFIG_KEY_RE.match(key):
                self.note(
                    "import.user_config", f"chave '{name}' inválida; ignorada", where, "warning"
                )
                continue
            spec = spec if isinstance(spec, dict) else {}
            entry: dict[str, Any] = {
                "description": str(spec.get("description") or spec.get("title") or "")[:300],
                "sensitive": bool(spec.get("sensitive", False)),
            }
            if "default" in spec and spec["default"] is not None:
                if entry["sensitive"]:
                    self.note(
                        "import.sensitive_default",
                        f"o padrão de '{name}' (sensitive) foi descartado: segredo não vai no pacote",
                        where,
                        "warning",
                    )
                else:
                    entry["default"] = str(spec["default"])
                    entry["required"] = False
            elif spec.get("required") is False:
                entry["required"] = False
            if spec.get("type") in ("file", "directory"):
                self.note(
                    "import.user_config_path",
                    f"'{name}' pede um caminho de {spec['type']} na máquina da pessoa; "
                    "isso não se aplica aqui",
                    where,
                )
            self.config[key] = entry

    def mcp_servers(self, servers: Any, where: str) -> None:
        if not isinstance(servers, dict):
            self.note("import.mcp", "mcpServers precisa ser um mapa; ignorado", where, "warning")
            return
        for raw_name, cfg in servers.items():
            item = f"{where}: {raw_name}"
            name = local_name(str(raw_name))
            if not name:
                self.note(
                    "import.mcp",
                    f"servidor '{raw_name}' com nome inválido; ignorado",
                    where,
                    "warning",
                )
                continue
            if not isinstance(cfg, dict):
                self.note("import.mcp", "configuração inválida; ignorado", item, "warning")
                continue
            kind = str(cfg.get("type") or ("http" if "url" in cfg else "stdio")).lower()
            if kind == "stdio" or "command" in cfg:
                self.note(
                    "import.stdio_server",
                    "servidor MCP local (stdio) não importado: ele executaria um comando na "
                    "máquina do router. Hospede-o como servidor remoto (http) e cadastre-o",
                    item,
                    "warning",
                )
                continue
            if kind not in ("http", "sse", "streamable-http", "streamablehttp"):
                self.note("import.mcp", f"tipo '{kind}' não suportado; ignorado", item, "warning")
                continue
            url = cfg.get("url")
            if not isinstance(url, str) or not url.strip():
                self.note("import.mcp", "servidor sem url; ignorado", item, "warning")
                continue
            if any(c["name"] == name for c in self.connectors):
                self.note(
                    "import.mcp", f"nome '{name}' repetido; o segundo foi ignorado", item, "warning"
                )
                continue
            connector: dict[str, Any] = {
                "name": name,
                "url": self.rewrite(url.strip(), f"{item}: url"),
                "transport": "sse" if kind == "sse" else "streamable-http",
            }
            headers = cfg.get("headers")
            if isinstance(headers, dict):
                ignored = []
                for header, value in headers.items():
                    if str(header).lower() == "authorization" and isinstance(value, str):
                        match = _BEARER_RE.match(value)
                        if match:
                            connector["auth_token"] = self.rewrite(
                                match.group(1), f"{item}: headers", sensitive=True
                            )
                            continue
                        self.note(
                            "import.header_dropped",
                            "Authorization que não é 'Bearer ${variável}' não foi importado "
                            "(token escrito no pacote vaza; use uma variável)",
                            item,
                            "warning",
                        )
                        continue
                    ignored.append(str(header))
                if ignored:
                    self.note(
                        "import.header_ignored",
                        f"cabeçalhos ignorados: {', '.join(ignored)} (só Authorization: Bearer é usado)",
                        item,
                    )
            for extra in ("headersHelper", "oauth"):
                if extra in cfg:
                    self.note(
                        "import.auth_ignored",
                        f"'{extra}' ignorado: a autenticação do conector é por token (auth_token)",
                        item,
                    )
            self.connectors.append(connector)


def convert(files: Mapping[str, bytes]) -> Converted:
    """Converte um pacote no formato do Claude para um manifesto nativo (dicionário)."""
    out = Converted()
    raw, problems = parse_json(files[CLAUDE_MANIFEST], CLAUDE_MANIFEST)
    out.findings.extend(problems)
    if raw is None:
        return out
    if not isinstance(raw, dict):
        out.findings.append(
            Finding(
                "block",
                "manifest.invalid",
                "o manifesto precisa ser um objeto JSON",
                CLAUDE_MANIFEST,
            )
        )
        return out

    builder = _Builder(files)
    manifest: dict[str, Any] = {}
    name = raw.get("name")
    manifest["name"] = name.strip() if isinstance(name, str) else name
    version = raw.get("version")
    if isinstance(version, str) and version.strip():
        manifest["version"] = version.strip()
    else:
        manifest["version"] = "0.0.0"
        builder.note(
            "import.no_version",
            "o plugin do Claude não declara versão: usando 0.0.0 (o uso por versão fica agrupado nela)",
            CLAUDE_MANIFEST,
            "warning",
        )
    description = raw.get("description")
    if isinstance(description, str) and description.strip():
        manifest["description"] = description.strip()[:500]
    else:
        manifest["description"] = f"Plugin {manifest['name']} importado do formato do Claude"
        builder.note("import.no_description", "sem description no manifesto", CLAUDE_MANIFEST)
    for key in ("author", "license", "homepage", "keywords"):
        if raw.get(key) is not None:
            manifest[key] = raw[key]

    builder.user_config(raw.get("userConfig"))

    # servidores MCP: o arquivo padrão, o caminho em mcpServers ou o mapa embutido
    inline = raw.get("mcpServers")
    sources: list[tuple[Any, str]] = []
    if CLAUDE_MCP_FILE in files:
        sources.append((CLAUDE_MCP_FILE, CLAUDE_MCP_FILE))
    if isinstance(inline, dict):
        sources.append((inline, f"{CLAUDE_MANIFEST}: mcpServers"))
    elif isinstance(inline, str):
        target = inline.removeprefix("./")
        if target != CLAUDE_MCP_FILE:
            sources.append((target, target))
    for source, where in sources:
        if isinstance(source, str):
            if source not in files:
                builder.note("import.mcp", f"{source} não existe no pacote", where, "warning")
                continue
            data, problems = parse_json(files[source], source)
            builder.findings.extend(problems)
            if data is None:
                continue
            servers = data.get("mcpServers", data) if isinstance(data, dict) else None
        else:
            servers = source
        builder.mcp_servers(servers, where)

    for key, why in _UNSUPPORTED.items():
        present_field = raw.get(key) is not None
        present_dir = any(p.startswith(f"{key}/") for p in files)
        if present_field or present_dir:
            builder.note("import.not_supported", f"'{key}' não importado: {why}", CLAUDE_MANIFEST)
    if CLAUDE_MARKETPLACE in files:
        builder.note(
            "import.marketplace",
            "marketplace.json ignorado: cada plugin é instalado por vez",
            CLAUDE_MARKETPLACE,
        )
    known = {
        "name", "version", "description", "author", "license", "homepage", "keywords",
        "userConfig", "mcpServers", "repository", "skills", *_UNSUPPORTED,
    }  # fmt: skip
    unknown = sorted(set(raw) - known)
    if unknown:
        builder.note(
            "import.ignored_fields",
            f"campos do manifesto ignorados: {', '.join(unknown)}",
            CLAUDE_MANIFEST,
        )

    if builder.config:
        manifest["config"] = builder.config
    if builder.connectors:
        manifest["connectors"] = builder.connectors
    out.manifest = manifest
    out.findings.extend(builder.findings)
    return out
