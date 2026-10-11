"""Plugins no modo framework: declarados em ``plugins:`` do ``switchboard.yaml``.

No YAML quem manda é o arquivo do operador: o plugin é lido do disco na subida, a
configuração (``config:``) entra com ``${VAR}``/``env:NOME`` e os achados ``held`` da
varredura só passam se o operador os aceitar por código (``allow:``). Qualquer problema
(pacote com ``block``, achado sem aceite, configuração faltando) interrompe a subida com
uma mensagem que diz o que fazer: em arquivo de configuração, falhar cedo é melhor que
rodar um plugin pela metade.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PluginRef
from ..errors import ConfigError, PluginError
from .bundle import ExpandedPlugin, PluginBundle, load_plugin
from .findings import Finding, PluginRejected, format_findings


@dataclass
class LoadedPlugin:
    ref: PluginRef
    bundle: PluginBundle
    expanded: ExpandedPlugin
    accepted: list[Finding] = field(default_factory=list)  # achados held aceitos por 'allow'

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def version(self) -> str:
        return self.bundle.version


def load_plugins(
    refs: Sequence[PluginRef], base_dir: str | Path, *, switchboard_version: str | None = None
) -> list[LoadedPlugin]:
    """Lê, valida e expande os plugins ligados (``enabled: true``) do YAML."""
    base = Path(base_dir)
    out: list[LoadedPlugin] = []
    for ref in refs:
        if not ref.enabled:
            continue
        label = f"plugin '{ref.name}' ({ref.path})"
        path = Path(ref.path)
        bundle = load_plugin(
            path if path.is_absolute() else base / path, switchboard_version=switchboard_version
        )
        try:
            bundle.require_ok()
        except PluginRejected as exc:
            raise ConfigError(f"{label} recusado:\n{format_findings(exc.findings)}") from exc
        if bundle.name != ref.name:
            raise ConfigError(
                f"{label}: o manifesto se chama '{bundle.name}'; use o mesmo nome em plugins[].name"
            )
        pending = bundle.unapproved(allow_codes=ref.allow)
        if pending:
            codes = sorted({f.code for f in pending})
            raise ConfigError(
                f"{label} tem achados da varredura de segurança que exigem aprovação:\n"
                f"{format_findings(pending)}\n"
                "Revise o pacote e, se aceitar, acrescente ao plugin no YAML: "
                f"allow: [{', '.join(codes)}]"
            )
        try:
            expanded = bundle.expand(ref.config)
        except PluginError as exc:
            raise ConfigError(f"{label}: {exc}") from exc
        if expanded.missing:
            raise ConfigError(
                f"{label}: falta configurar {', '.join(expanded.missing)} "
                "(plugins[].config no YAML)"
            )
        accepted = [f for f in bundle.held if f.code in set(ref.allow)]
        out.append(LoadedPlugin(ref, bundle, expanded, accepted))
    return out
