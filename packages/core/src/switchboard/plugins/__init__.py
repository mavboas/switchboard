"""Plugins: pacotes declarativos que o router recebe (subir/atualizar/remover).

Um plugin traz **capacidades** (conectores MCP, agentes A2A, bases de conhecimento) e o
**know-how** de usá-las (playbooks, no formato Agent Skills). É dado, nunca código: o
router não executa nada de um plugin. O fluxo é

1. ler o pacote com limites (``load_plugin``: diretório, ``.zip`` ou bytes);
2. validar o manifesto e varrer o conteúdo (achados ``block``/``held``/``warning``/``note``);
3. aplicar a configuração de quem instala (``bundle.expand(config)``): endereços e tokens
   nunca vêm no pacote;
4. ligar o plugin a perfis; o uso é contado por plugin e por versão.

Veja ``docs/plugins.md`` e ``docs/adr/0002-plugins-declarativos.md``.
"""

from .archive import DEFAULT_LIMITS, Limits, pack, read_source
from .bundle import ExpandedPlugin, PluginBundle, build_bundle, load_plugin, resolve_config
from .findings import (
    LEVEL_LABELS,
    LEVELS,
    Finding,
    PluginRejected,
    blocking,
    counts,
    format_findings,
    held,
    sorted_findings,
)
from .manifest import API_VERSION, MANIFEST_FILES, PluginManifest
from .playbook import Playbook
from .semver import compare_versions, is_semver, satisfies
from .skills import Skill
from .yaml_mode import LoadedPlugin, load_plugins

__all__ = [
    "API_VERSION",
    "DEFAULT_LIMITS",
    "LEVELS",
    "LEVEL_LABELS",
    "MANIFEST_FILES",
    "ExpandedPlugin",
    "Finding",
    "Limits",
    "LoadedPlugin",
    "Playbook",
    "PluginBundle",
    "PluginManifest",
    "PluginRejected",
    "Skill",
    "blocking",
    "build_bundle",
    "compare_versions",
    "counts",
    "format_findings",
    "held",
    "is_semver",
    "load_plugin",
    "load_plugins",
    "pack",
    "read_source",
    "resolve_config",
    "satisfies",
    "sorted_findings",
]
