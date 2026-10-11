"""Carregar um plugin: pacote → manifesto validado, skills, documentos e achados.

``load_plugin`` nunca levanta por problema de conteúdo: devolve um :class:`PluginBundle`
com a lista de achados (``block``, ``held``, ``warning``, ``note``) e quem chamou decide
(``bundle.require_ok()`` recusa o que tem ``block``; ``bundle.unapproved()`` diz o que
ainda espera aprovação). ``bundle.expand(config)`` junta o plugin com a configuração de
quem instala e entrega os conectores, agentes, bases e playbooks já qualificados
(``<plugin>.<nome>``) que o resto do Switchboard usa.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from ..config import AgentSpec, ConnectorSpec, EmbedderSpec, KnowledgeBaseSpec, qualify
from ..errors import IngestError, PluginError
from ..rag.loaders import extract_text
from . import claude
from .archive import DEFAULT_LIMITS, Limits, Package, digest_files, pack, read_source
from .findings import Finding, PluginRejected, blocking, counts, held, sorted_findings
from .manifest import (
    EXACT_PLACEHOLDER_RE,
    MANIFEST_FILES,
    MAX_URL_CHARS,
    PLACEHOLDER_RE,
    PluginManifest,
)
from .playbook import Playbook
from .scan import classify_host, dedupe, scan_manifest, scan_secrets, scan_skills, scan_text
from .skills import Skill, decode_text, discover_skills, safe_load_yaml

MAX_CONFIG_VALUE = 2000
MAX_PER_CODE_LOW = 25  # avisos e notas iguais além disso viram "e mais N"
HTML_SUFFIXES = (".html", ".htm")


def _current_version() -> str:
    from .. import __version__  # import tardio: o pacote raiz importa este módulo

    return __version__


# --------------------------------------------------------------------------
# o pacote carregado


@dataclass
class PluginBundle:
    origin: str
    format: str  # "switchboard" | "claude" (importado)
    files: dict[str, bytes]
    findings: list[Finding]
    manifest: PluginManifest | None = None
    skills: list[Skill] = field(default_factory=list)
    playbooks: list[Playbook] = field(default_factory=list)
    documents: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # base local -> {caminho: texto}

    # -- estado --------------------------------------------------------------

    @property
    def digest(self) -> str:
        return digest_files(self.files)

    @property
    def name(self) -> str:
        return self.manifest.name if self.manifest else ""

    @property
    def version(self) -> str:
        return self.manifest.version if self.manifest else ""

    @property
    def blocked(self) -> list[Finding]:
        return blocking(self.findings)

    @property
    def held(self) -> list[Finding]:
        return held(self.findings)

    @property
    def ok(self) -> bool:
        return self.manifest is not None and not self.blocked

    def require_ok(self) -> PluginBundle:
        if not self.ok:
            raise PluginRejected(self.findings)
        return self

    def unapproved(
        self, approved: Iterable[str] = (), allow_codes: Iterable[str] = ()
    ) -> list[Finding]:
        """Achados ``held`` que ainda faltam aprovar.

        ``approved`` são fingerprints aprovados no console; ``allow_codes`` são os códigos
        aceitos por quem escreveu o YAML (``allow:``).
        """
        ok_fingerprints = set(approved)
        ok_codes = set(allow_codes)
        return [
            f for f in self.held if f.fingerprint not in ok_fingerprints and f.code not in ok_codes
        ]

    def pack(self) -> bytes:
        """O pacote normalizado (zip determinístico): é o que fica guardado."""
        return pack(self.files)

    # -- configuração --------------------------------------------------------

    def expand(
        self, config: Mapping[str, str] | None = None, *, embedder: EmbedderSpec | None = None
    ) -> ExpandedPlugin:
        """Aplica a configuração de quem instala e devolve as peças prontas para uso."""
        manifest = self.manifest
        if manifest is None:
            raise PluginError("plugin sem manifesto válido")
        values, missing = resolve_config(manifest, config)
        out = ExpandedPlugin(plugin=manifest.name, version=manifest.version, missing=missing)
        for conn in manifest.connectors:
            url = _fill(conn.url, values)
            token = _fill_token(conn.auth_token, values)
            if url is None:
                continue
            _check_endpoint(url, qualify(manifest.name, conn.name))
            out.connectors.append(
                _build(
                    ConnectorSpec,
                    name=qualify(manifest.name, conn.name),
                    description=conn.description,
                    url=url,
                    transport=conn.transport,
                    auth_token=token,
                    allowed_tools=list(conn.allowed_tools),
                    timeout_s=conn.timeout_s,
                )
            )
        for agent in manifest.agents:
            url = _fill(agent.url, values)
            token = _fill_token(agent.auth_token, values)
            if url is None:
                continue
            _check_endpoint(url, qualify(manifest.name, agent.name))
            out.agents.append(
                _build(
                    AgentSpec,
                    name=qualify(manifest.name, agent.name),
                    description=agent.description,
                    url=url,
                    auth_token=token,
                    allowed_skills=list(agent.allowed_skills),
                    timeout_s=agent.timeout_s,
                    deadline_s=agent.deadline_s,
                    push=agent.push,
                    allow_cross_origin=agent.allow_cross_origin,
                )
            )
        for kb in manifest.knowledge_bases:
            qualified = qualify(manifest.name, kb.name)
            out.knowledge_bases.append(
                KnowledgeBaseSpec(
                    name=qualified,
                    description=kb.description,
                    embedder=embedder or EmbedderSpec(),
                    chunk_size=kb.chunk_size,
                    chunk_overlap=kb.chunk_overlap,
                )
            )
            out.documents[qualified] = dict(self.documents.get(kb.name, {}))
        out.playbooks = list(self.playbooks)
        out.examples = self.routing_examples()
        return out

    def routing_examples(self) -> dict[str, list[str]]:
        """Frases de exemplo por capacidade, com o componente já qualificado (``plugin.dono/capacidade``)."""
        manifest = self.manifest
        if manifest is None:
            return {}
        out: dict[str, list[str]] = {}
        for key, examples in manifest.routing_examples.items():
            owner, _, capability = key.partition("/")
            out[f"{qualify(manifest.name, owner)}/{capability}"] = [e.strip() for e in examples]
        return out

    # -- apresentação --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Resumo para API e console (nada sensível: o pacote não traz segredos)."""
        m = self.manifest
        info: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "origin": self.origin,
            "digest": self.digest,
            "findings": [f.to_dict() for f in sorted_findings(self.findings)],
            "counts": counts(self.findings),
            "ok": self.ok,
        }
        if m is None:
            return info
        info.update(
            description=m.description,
            author=m.author.model_dump(exclude_none=True) if m.author else None,
            license=m.license,
            homepage=m.homepage,
            keywords=m.keywords,
            requires=m.requires.model_dump(exclude_none=True),
            config=[
                {
                    "key": key,
                    "description": f.description,
                    "sensitive": f.sensitive,
                    "required": f.required,
                    "default": f.default,
                }
                for key, f in m.config.items()
            ],
            connectors=[
                {
                    "name": c.name,
                    "description": c.description,
                    "url": c.url,
                    "transport": c.transport,
                    "allowed_tools": c.allowed_tools,
                }
                for c in m.connectors
            ],
            agents=[
                {
                    "name": a.name,
                    "description": a.description,
                    "url": a.url,
                    "allowed_skills": a.allowed_skills,
                }
                for a in m.agents
            ],
            knowledge_bases=[
                {
                    "name": k.name,
                    "description": k.description,
                    "documents": len(self.documents.get(k.name, {})),
                }
                for k in m.knowledge_bases
            ],
            playbooks=[
                {
                    "skill": p.skill,
                    "description": p.description,
                    "applies_to": list(p.applies_to),
                    "triggers": list(p.triggers),
                    "always": p.always,
                    "chars": len(p.body),
                }
                for p in self.playbooks
            ],
        )
        return info


@dataclass
class ExpandedPlugin:
    """O plugin com a configuração aplicada: o que entra no roteador."""

    plugin: str
    version: str
    connectors: list[ConnectorSpec] = field(default_factory=list)
    agents: list[AgentSpec] = field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseSpec] = field(default_factory=list)
    documents: dict[str, dict[str, str]] = field(default_factory=dict)  # base qualificada -> docs
    playbooks: list[Playbook] = field(default_factory=list)
    examples: dict[str, list[str]] = field(
        default_factory=dict
    )  # 'plugin.dono/capacidade' -> frases
    missing: list[str] = field(default_factory=list)  # config obrigatória sem valor

    @property
    def complete(self) -> bool:
        return not self.missing


# --------------------------------------------------------------------------
# configuração


def check_config_value(key: str, value: str) -> None:
    """Valor de config como foi digitado: tamanho e caracteres (antes de cifrar)."""
    if len(value) > MAX_CONFIG_VALUE:
        raise PluginError(f"config {key}: valor com mais de {MAX_CONFIG_VALUE} caracteres")
    if any(ord(ch) < 32 or ch == "\x7f" for ch in value):
        raise PluginError(
            f"config {key}: o valor não pode ter quebra de linha nem caractere de controle"
        )


def resolve_config(
    manifest: PluginManifest, values: Mapping[str, str] | None
) -> tuple[dict[str, str], list[str]]:
    """Valores efetivos (informados ou padrão) e as chaves obrigatórias que faltam."""
    given = {k: v for k, v in (values or {}).items() if v is not None}
    unknown = sorted(set(given) - set(manifest.config))
    if unknown:
        known = ", ".join(manifest.config) or "nenhuma"
        raise PluginError(
            f"config desconhecida no plugin '{manifest.name}': {', '.join(unknown)} (declaradas: {known})"
        )
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, spec in manifest.config.items():
        raw = given.get(key)
        if raw is None or raw == "":
            if spec.default is not None:
                resolved[key] = spec.default
            elif spec.required:
                missing.append(key)
            continue
        if not spec.sensitive:  # segredos são conferidos ao entrar (o cifrado é bem maior)
            check_config_value(key, raw)
        resolved[key] = raw
    return resolved, missing


def _check_endpoint(url: str, component: str) -> None:
    """O endereço final (pacote + configuração) de um componente.

    Rede interna é escolha legítima de quem instala; usuário e senha no endereço e o
    endpoint de metadados da nuvem, não: o token vai em ``auth_token``, e nenhum conector
    ou agente tem por que falar com o metadados.
    """
    if len(url) > MAX_URL_CHARS:
        raise PluginError(f"{component}: o endereço passa de {MAX_URL_CHARS} caracteres")
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError as exc:
        raise PluginError(f"{component}: endereço inválido ({exc})") from exc
    if parts.scheme not in ("http", "https") or not host:
        raise PluginError(f"{component}: o endereço precisa ser http(s)://host/…")
    if "@" in parts.netloc:
        raise PluginError(
            f"{component}: o endereço não pode ter usuário e senha; o token vai na config "
            "sensível usada em auth_token"
        )
    if classify_host(host) == "metadata":
        raise PluginError(f"{component}: o endereço aponta para o endpoint de metadados da nuvem")


def _fill(template: str, values: Mapping[str, str]) -> str | None:
    """Troca ``${config.X}``; ``None`` se algum valor falta (o componente fica de fora)."""
    missing = False

    def repl(match: Any) -> str:
        nonlocal missing
        key = match.group(1)
        if key not in values:
            missing = True
            return match.group(0)
        return values[key]

    out = PLACEHOLDER_RE.sub(repl, template)
    return None if missing else out


def _fill_token(template: str | None, values: Mapping[str, str]) -> str | None:
    if not template:
        return None
    match = EXACT_PLACEHOLDER_RE.match(template.strip())
    if match is None:
        return None  # recusado antes, na varredura (secret.literal)
    return values.get(match.group(1)) or None


def _build(cls: type, **kwargs: Any) -> Any:
    try:
        return cls(**kwargs)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        )
        raise PluginError(
            f"a configuração do plugin gera um {cls.__name__} inválido ({kwargs.get('name')}): {errors}"
        ) from exc


# --------------------------------------------------------------------------
# leitura


def load_plugin(
    source: str | Path | bytes,
    *,
    switchboard_version: str | None = None,
    limits: Limits = DEFAULT_LIMITS,
) -> PluginBundle:
    """Lê um diretório, um ``.zip`` ou os bytes de um ``.zip`` e devolve o pacote analisado.

    ``switchboard_version=""`` desliga a checagem de ``requires.switchboard``.
    """
    package, origin = read_source(source, limits=limits)
    return build_bundle(package, origin, switchboard_version=switchboard_version)


def _read_manifest(files: dict[str, bytes], findings: list[Finding]) -> tuple[Any, str | None]:
    """O manifesto nativo: ``(dados, nome do arquivo)``; ``(None, None)`` se não há."""
    present = [name for name in MANIFEST_FILES if name in files]
    if not present:
        return None, None
    if len(present) > 1:
        findings.append(
            Finding(
                "block",
                "manifest.ambiguous",
                f"mais de um manifesto ({', '.join(present)}): deixe só um",
                present[0],
            )
        )
        return None, present[0]
    name = present[0]
    if name.endswith(".json"):
        data, problems = claude.parse_json(files[name], name)
        findings.extend(problems)
        return data, name
    text = decode_text(files[name].removeprefix(b"\xef\xbb\xbf"))
    if text is None:
        findings.append(
            Finding("block", "manifest.encoding", "o manifesto precisa estar em UTF-8", name)
        )
        return None, name
    try:
        return safe_load_yaml(text), name
    except yaml.YAMLError as exc:
        findings.append(
            Finding("block", "manifest.invalid", f"YAML inválido: {str(exc).splitlines()[0]}", name)
        )
        return None, name


def _validation_findings(exc: ValidationError, label: str) -> list[Finding]:
    out: list[Finding] = []
    for err in exc.errors()[:25]:
        loc = ".".join(str(p) for p in err["loc"]) or "(raiz)"
        message = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            message = "campo desconhecido (confira a grafia ou a versão do formato)"
        elif err["type"] == "missing":
            message = "campo obrigatório"
        out.append(Finding("block", "manifest.invalid", message, f"{label}: {loc}"))
    return out


def _dump_manifest(manifest: PluginManifest, origin: str) -> bytes:
    data = manifest.model_dump(mode="json", by_alias=True, exclude_defaults=True, exclude_none=True)
    data.pop("schema", None)
    head = f"# gerado na importação de {origin}\n"
    return (head + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)).encode("utf-8")


def build_bundle(
    package: Package, origin: str, *, switchboard_version: str | None = None
) -> PluginBundle:
    version = switchboard_version if switchboard_version is not None else _current_version()
    files = dict(package.files)
    findings = list(package.findings)
    bundle = PluginBundle(origin=origin, format="switchboard", files=files, findings=findings)

    data, label = _read_manifest(files, findings)
    if label is None:
        if claude.CLAUDE_MANIFEST in files:
            bundle.format = "claude"
            for i, f in enumerate(findings):
                if f.code == "package.disallowed_file":  # código nunca é guardado nem executado
                    findings[i] = Finding(
                        "note",
                        "import.dropped_file",
                        "arquivo descartado na importação (plugins só trazem texto declarativo; "
                        "o router nunca executa código)",
                        f.where,
                    )
            converted = claude.convert(files)
            findings.extend(converted.findings)
            data, label = converted.manifest, f"{claude.CLAUDE_MANIFEST} (convertido)"
        else:
            findings.append(
                Finding(
                    "block",
                    "manifest.missing",
                    "não achei plugin.yaml (nem .claude-plugin/plugin.json) na raiz do pacote",
                )
            )
    elif claude.CLAUDE_MANIFEST in files:
        findings.append(
            Finding(
                "note",
                "import.ignored_claude_manifest",
                "o pacote tem plugin.yaml e .claude-plugin/plugin.json: vale o plugin.yaml",
                claude.CLAUDE_MANIFEST,
            )
        )

    manifest: PluginManifest | None = None
    if data is not None:
        if not isinstance(data, dict):
            findings.append(
                Finding("block", "manifest.invalid", "o manifesto precisa ser um mapa", label or "")
            )
        else:
            try:
                manifest = PluginManifest.model_validate(data)
            except ValidationError as exc:
                findings.extend(_validation_findings(exc, label or "plugin.yaml"))
    bundle.manifest = manifest

    if manifest is None:
        # sem manifesto não dá para saber o que é texto lido pelo roteador; ainda assim
        # vale avisar de segredos e de texto escondido
        findings.extend(_scan_files(files, used=set()))
        bundle.findings = _finish(findings)
        return bundle

    if bundle.format == "claude":
        files["plugin.yaml"] = _dump_manifest(manifest, claude.CLAUDE_MANIFEST)

    skills, skill_findings = discover_skills(files, manifest.skills)
    findings.extend(skill_findings)
    bundle.skills = skills
    bundle.playbooks = _playbooks(manifest, skills, findings)
    bundle.documents = _documents(manifest, files, findings)

    findings.extend(
        scan_manifest(
            manifest,
            switchboard_version=version,
            skill_bodies={sk.name: sk.body for sk in skills},
        )
    )
    findings.extend(scan_skills(skills, manifest, files))
    doc_paths: set[str] = set()
    for docs in bundle.documents.values():
        for path in docs:
            if path in doc_paths:
                continue
            doc_paths.add(path)
            findings.extend(_scan_document(path, files[path], docs[path], manifest))
    used = {s.path for s in skills} | doc_paths
    findings.extend(_scan_files(files, used=used))
    bundle.findings = _finish(findings)
    return bundle


def _playbooks(
    manifest: PluginManifest, skills: list[Skill], findings: list[Finding]
) -> list[Playbook]:
    known = {s.name for s in skills}
    declared = {pb.skill: pb for pb in manifest.playbooks}
    for skill in declared:
        if skill not in known:
            findings.append(
                Finding(
                    "block",
                    "playbook.unknown_skill",
                    f"o playbook '{skill}' não corresponde a nenhuma skill do pacote",
                    "plugin.yaml: playbooks",
                )
            )
    out: list[Playbook] = []
    for skill in skills:
        if not skill.body:
            continue
        pb = declared.get(skill.name)
        out.append(
            Playbook(
                plugin=manifest.name,
                version=manifest.version,
                skill=skill.name,
                description=skill.description,
                body=skill.body,
                applies_to=tuple(qualify(manifest.name, o) for o in (pb.applies_to if pb else ())),
                triggers=tuple(pb.triggers) if pb else (),
                always=pb.always if pb else False,
            )
        )
    return out


def _documents(
    manifest: PluginManifest, files: dict[str, bytes], findings: list[Finding]
) -> dict[str, dict[str, str]]:
    """Textos de cada base de conhecimento, tirados dos arquivos do pacote."""
    out: dict[str, dict[str, str]] = {}
    for kb in manifest.knowledge_bases:
        docs: dict[str, str] = {}
        where = f"plugin.yaml: knowledge_bases.{kb.name}"
        for path in kb.paths:
            matches = (
                [path] if path in files else sorted(p for p in files if p.startswith(path + "/"))
            )
            if not matches:
                findings.append(
                    Finding("block", "kb.path_missing", f"'{path}' não existe no pacote", where)
                )
                continue
            for match in matches:
                if match in docs:
                    continue
                try:
                    docs[match] = extract_text(match, files[match])
                except IngestError as exc:
                    findings.append(Finding("warning", "kb.unreadable", f"ignorado: {exc}", match))
        if not docs:
            findings.append(
                Finding("block", "kb.empty", "a base não tem nenhum documento com texto", where)
            )
        out[kb.name] = docs
    return out


def _scan_document(
    path: str, raw: bytes, extracted: str, manifest: PluginManifest
) -> list[Finding]:
    """Documento de base: as regras de linguagem no texto exato que é indexado (o mesmo
    ``extract_text`` da ingestão: HTML sem tags, outras codificações, sem NUL), e a forma
    (invisíveis, controle, blocos codificados) também no arquivo cru."""
    owners = manifest.owners
    return scan_text(
        extracted, path, scope="knowledge", plugin=manifest.name, local_owners=owners
    ) + scan_text(raw.decode("utf-8", errors="replace"), path, structural_only=True)


def _scan_files(files: Mapping[str, bytes], *, used: set[str]) -> list[Finding]:
    """Segredos em qualquer arquivo; texto escondido nos que o router não lê."""
    findings: list[Finding] = []
    for path in sorted(files):
        text = files[path].decode("utf-8", errors="replace")
        findings.extend(scan_secrets(text, path))
        if path in used:
            continue
        findings.extend(scan_text(text, path, structural_only=True))
    return findings


def _finish(findings: list[Finding]) -> list[Finding]:
    """Sem repetição e com avisos/notas iguais agrupados (blocks e held aparecem todos)."""
    out: list[Finding] = []
    per_code: dict[tuple[str, str], int] = {}
    skipped: dict[tuple[str, str], int] = {}
    for f in sorted_findings(dedupe(findings)):
        if f.level in ("warning", "note"):
            key = (f.level, f.code)
            per_code[key] = per_code.get(key, 0) + 1
            if per_code[key] > MAX_PER_CODE_LOW:
                skipped[key] = skipped.get(key, 0) + 1
                continue
        out.append(f)
    for (level, code), n in skipped.items():
        out.append(Finding(level, code, f"mais {n} ocorrência(s) deste achado omitidas"))  # type: ignore[arg-type]
    return sorted_findings(out)
