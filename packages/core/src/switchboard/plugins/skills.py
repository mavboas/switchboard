"""Skills de um plugin: o formato Agent Skills (``skills/<nome>/SKILL.md``).

No Switchboard, uma skill de plugin é um **playbook**: instruções de especialista que
o roteador junta ao prompt do LLM quando o pedido tem a ver com a skill (ver
:mod:`switchboard.routing.playbooks`). O arquivo segue a especificação aberta de
Agent Skills, então o mesmo ``SKILL.md`` vale em outras ferramentas:

* frontmatter YAML com ``name`` (até 64 caracteres, minúsculas/números/hífens,
  igual ao nome da pasta) e ``description`` (até 1024: o que faz e quando usar);
* corpo em Markdown (a especificação recomenda menos de 500 linhas).

``references/`` e ``assets/`` são aceitos no pacote e varridos, mas o router não os
carrega (nada de arquivos lidos sob demanda); ``allowed-tools`` é ignorado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from .findings import Finding

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_COMPATIBILITY = 500
LONG_BODY_LINES = 500
KNOWN_KEYS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str  # skills/<nome>/SKILL.md
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    extra_files: tuple[str, ...] = ()  # references/, assets/ (não usados pelo router)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader que recusa chaves repetidas (``a: 1`` e depois ``a: 2`` esconderia valor)."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"chave repetida: {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: _construct_mapping(loader, node, deep=True),
)
# datas ficam como texto: o manifesto não tem campos de data, e "2024-02-30" não pode
# derrubar a leitura com um ValueError
_UniqueKeyLoader.add_constructor(
    "tag:yaml.org,2002:timestamp", lambda loader, node: loader.construct_scalar(node)
)


def safe_load_yaml(text: str) -> Any:
    """``yaml.safe_load`` que recusa chaves repetidas e âncoras/aliases.

    Aliases (``&a`` / ``*a``) permitem uma "bomba" de expansão (poucas linhas, bilhões de
    nós); como um plugin não precisa deles, são recusados antes de montar o documento.
    """
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent):
                raise yaml.YAMLError("âncoras e aliases do YAML (&, *, <<) não são permitidos")
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader com checagem extra
    except RecursionError as exc:
        raise yaml.YAMLError("aninhamento fundo demais") from exc
    except (ValueError, TypeError, OverflowError) as exc:
        # número com milhares de dígitos, chave que não é texto nem número…
        raise yaml.YAMLError(f"valor inválido no YAML: {str(exc).splitlines()[0][:120]}") from exc


def decode_text(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def parse_skill(
    path: str, data: bytes, *, directory: str, extra_files: tuple[str, ...] = ()
) -> tuple[Skill | None, list[Finding]]:
    """Lê e valida um ``SKILL.md``; ``None`` se não dá para usar (há um achado ``block``)."""
    findings: list[Finding] = []

    def block(code: str, message: str) -> None:
        findings.append(Finding("block", code, message, path))

    text = decode_text(data)
    if text is None:
        block("skill.encoding", "o SKILL.md precisa estar em UTF-8")
        return None, findings
    match = _FRONTMATTER_RE.match(text)
    if not match:
        block(
            "skill.frontmatter",
            "falta o frontmatter YAML (bloco entre '---' no começo do arquivo) com name e description",
        )
        return None, findings
    try:
        meta = safe_load_yaml(match.group(1))
    except yaml.YAMLError as exc:
        block("skill.frontmatter", f"frontmatter YAML inválido: {str(exc).splitlines()[0]}")
        return None, findings
    if not isinstance(meta, dict):
        block("skill.frontmatter", "o frontmatter precisa ser um mapa YAML")
        return None, findings

    name = meta.get("name")
    description = meta.get("description")
    problems = False
    if not isinstance(name, str) or not name.strip():
        block("skill.name", "falta o campo name")
        problems = True
    else:
        name = name.strip()
        if len(name) > MAX_NAME or not SKILL_NAME_RE.match(name):
            block(
                "skill.name",
                f"name '{name}' inválido: até {MAX_NAME} caracteres, só minúsculas, números e "
                "hífens (sem hífen no começo/fim nem hífens seguidos)",
            )
            problems = True
        elif name != directory:
            block("skill.name", f"name '{name}' precisa ser igual ao nome da pasta '{directory}'")
            problems = True
    if not isinstance(description, str) or not description.strip():
        block("skill.description", "falta o campo description (o que a skill faz e quando usar)")
        problems = True
    elif len(description.strip()) > MAX_DESCRIPTION:
        block("skill.description", f"description passa de {MAX_DESCRIPTION} caracteres")
        problems = True
    if problems:
        return None, findings

    compatibility = meta.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str) or len(compatibility) > MAX_COMPATIBILITY
    ):
        findings.append(
            Finding(
                "warning",
                "skill.compatibility",
                f"compatibility deve ser texto de até {MAX_COMPATIBILITY} caracteres; ignorado",
                path,
            )
        )
        compatibility = None
    metadata_raw = meta.get("metadata") or {}
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}
    else:
        findings.append(
            Finding("warning", "skill.metadata", "metadata precisa ser um mapa; ignorado", path)
        )
    license_ = meta.get("license")
    for key in meta:
        if key == "allowed-tools":
            findings.append(
                Finding(
                    "note",
                    "skill.allowed_tools_ignored",
                    "allowed-tools é ignorado: o que o plugin pode acionar é decidido "
                    "pelos conectores e agentes dele",
                    path,
                )
            )
        elif key not in KNOWN_KEYS:
            findings.append(Finding("note", "skill.unknown_field", f"campo '{key}' ignorado", path))

    body = text[match.end() :].strip()
    lines = body.count("\n") + 1 if body else 0
    if not body:
        findings.append(Finding("warning", "skill.empty_body", "o corpo da skill está vazio", path))
    elif lines > LONG_BODY_LINES:
        findings.append(
            Finding(
                "warning",
                "skill.long_body",
                f"{lines} linhas (a especificação recomenda menos de {LONG_BODY_LINES}); "
                "o router corta o playbook no limite do prompt",
                path,
            )
        )
    if extra_files:
        findings.append(
            Finding(
                "note",
                "skill.references_unused",
                f"{len(extra_files)} arquivo(s) em references/ ou assets/: o router não os carrega",
                path,
            )
        )
    return (
        Skill(
            name=name,
            description=description.strip(),
            body=body,
            path=path,
            license=str(license_) if license_ else None,
            compatibility=compatibility,
            metadata=metadata,
            extra_files=extra_files,
        ),
        findings,
    )


def discover_skills(
    files: dict[str, bytes], only: list[str] | None = None
) -> tuple[list[Skill], list[Finding]]:
    """Acha ``skills/<pasta>/SKILL.md`` no pacote (ou só as pastas de ``only``)."""
    findings: list[Finding] = []
    directories = sorted(
        {
            p.split("/")[1]
            for p in files
            if p.startswith("skills/") and p.count("/") >= 2 and p.split("/")[2:] != []
        }
    )
    names = directories if only is None else only
    if only is not None:
        for wanted in only:
            if f"skills/{wanted}/SKILL.md" not in files:
                findings.append(
                    Finding(
                        "block",
                        "skill.missing",
                        f"o manifesto lista a skill '{wanted}', mas skills/{wanted}/SKILL.md não existe",
                        "plugin.yaml: skills",
                    )
                )
    skills: list[Skill] = []
    for directory in names:
        path = f"skills/{directory}/SKILL.md"
        if path not in files:
            if only is None:
                findings.append(
                    Finding(
                        "warning",
                        "skill.no_skill_md",
                        f"a pasta skills/{directory}/ não tem SKILL.md e será ignorada",
                        f"skills/{directory}",
                    )
                )
            continue
        extras = tuple(
            sorted(
                p
                for p in files
                if p.startswith(f"skills/{directory}/")
                and p != path
                and not p.endswith("/SKILL.md")
            )
        )
        skill, skill_findings = parse_skill(
            path, files[path], directory=directory, extra_files=extras
        )
        findings.extend(skill_findings)
        if skill is not None:
            skills.append(skill)
    return skills, findings
