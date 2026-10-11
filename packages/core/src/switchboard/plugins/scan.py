"""Varredura de segurança de um plugin (triagem por regras, em português e inglês).

O que um plugin pode fazer de errado, mesmo sendo só dados:

* **vazar segredos** — token ou chave colada no manifesto, nas skills ou nos documentos;
* **apontar o router para onde não deve** — rede interna, endpoint de metadados da
  nuvem, URL com usuário e senha;
* **envenenar o prompt** — texto que vira instrução para o LLM: mandar ignorar as
  regras, esconder algo do usuário, pedir senha, acionar ferramentas "em todo pedido",
  atrapalhar outras ferramentas, buscar instruções num endereço externo, ou esconder
  tudo isso em caracteres invisíveis, comentários HTML ou texto codificado;
* **usar o que não declarou** — mandar chamar capacidades de outro plugin.

É uma **triagem**: pega o que dá para detectar por regra, para que uma pessoa revise o
que sobra. Quem protege de verdade é o resto do desenho (o plugin é só declarativo, os
segredos vão cifrados, os achados ``held`` exigem aprovação, tudo é auditado por versão).
Falsos positivos são esperados e se resolvem aprovando; falsos negativos, não — por isso
o texto de um plugin deve ser lido por alguém antes de aprovar um plugin de terceiros.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from ..text import fold
from .findings import Finding, Level
from .manifest import (
    EXACT_PLACEHOLDER_RE,
    PLACEHOLDER_RE,
    PluginManifest,
)
from .playbook import broad_trigger
from .semver import satisfies
from .skills import Skill

MAX_PER_CODE = 3  # ocorrências do mesmo achado por arquivo; o resto vira "e mais N"
EVIDENCE_CHARS = 110

TEXT_SCOPES = ("description", "playbook", "knowledge")


# --------------------------------------------------------------------------
# regras de texto

_NEGATORS = (
    "nunca",
    "nao",
    "jamais",
    "never",
    "do not",
    "don't",
    "dont",
    "evite",
    "avoid",
    "proibido",
    "forbidden",
    "without",
)


@dataclass(frozen=True)
class _Rule:
    code: str
    level: Level
    pattern: re.Pattern[str]
    message: str
    negatable: bool = False  # "nunca peça a senha" não é um pedido de senha
    scopes: frozenset[str] | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_VERB_IGNORE = r"(?:ignor\w*|disregard|forget|esquec\w*|desconsider\w*|descart\w*|desobedec\w*|override|sobrescrev\w*)"
_SCOPE_ALL = r"(?:previous|prior|above|earlier|preceding|all|any|every|anteriores?|acima|todas?|qualquer|tudo|demais)"
_NOUN_RULES = r"(?:instruction\w*|prompt\w*|rule\w*|guideline\w*|polic\w+|instruc\w*|regra\w*|orienta\w*|diretriz\w*|comando\w*|restric\w*)"
_SCOPE_AFTER = r"(?:previous|prior|above|earlier|preceding|anteriores?|acima|precedentes?)"

TEXT_RULES: tuple[_Rule, ...] = (
    _Rule(
        "inj.override",
        "held",
        _rx(
            rf"\b{_VERB_IGNORE}\b[^.\n]{{0,40}}\b{_SCOPE_ALL}\b[^.\n]{{0,40}}\b{_NOUN_RULES}"
            rf"|\b{_VERB_IGNORE}\b[^.\n]{{0,40}}\b{_NOUN_RULES}\b[^.\n]{{0,20}}\b{_SCOPE_AFTER}\b"
        ),
        "manda o modelo ignorar instruções ou regras anteriores",
    ),
    _Rule(
        "inj.role_reset",
        "held",
        _rx(
            r"\b(?:you are now|you must now|from now on,? you|a partir de agora,? (?:voce|vc|o assistente)"
            r"|new (?:system )?(?:prompt|instructions)|novo (?:system )?prompt|novas instrucoes"
            r"|system prompt\s*:|developer message\s*:|mensagem de sistema\s*:)"
        ),
        "tenta redefinir o papel ou o prompt do sistema",
    ),
    _Rule(
        "inj.role_marker",
        "held",
        _rx(r"^\s*(?:#{1,6}\s*)?(?:system|assistant|developer|sistema|assistente)\s*:"),
        "linha que imita uma mensagem de sistema ou do assistente",
    ),
    _Rule(
        "inj.concealment",
        "held",
        _rx(
            r"\b(?:do not|don't|never|nao|nunca|sem|jamais)\b[^.\n:;]{0,30}\b(?:tell|inform|mention|reveal|"
            r"disclose|notify|show|avisar|avise|informar|informe|mencionar|mencione|revelar|revele|"
            r"contar|conte|dizer|diga|comunicar|comunique|notificar|notifique|mostrar|mostre|expor|"
            r"exponha)\b[^.\n:;]{0,40}\b(?:user|users|usuario|usuarios|cliente|clientes|customer|"
            r"human|pessoa)\b"
        ),
        "manda esconder algo do usuário",
    ),
    _Rule(
        "inj.concealment",
        "held",
        _rx(
            r"\b(?:secretly|silently|covertly|em segredo|secretamente|silenciosamente|discretamente|"
            r"sem que o (?:usuario|cliente) (?:saiba|perceba|note|veja))\b"
        ),
        "manda agir em segredo ou sem que o usuário perceba",
    ),
    _Rule(
        "inj.always_on",
        "held",
        _rx(
            r"\b(?:for (?:every|all|any|each)|em (?:todo|toda|todos|todas|qualquer)|regardless of|"
            r"independentemente d\w+|sempre que houver)\b[^.\n]{0,40}\b(?:request|requests|message|messages|"
            r"query|queries|pedido|pedidos|mensagem|mensagens|pergunta|perguntas|conversa|conversation)\b"
            r"[^.\n]{0,80}\b(?:call|invoke|run|execute|use|send|chame|invoque|acione|utilize|envie|dispare)\b"
        ),
        "manda acionar algo em todo pedido, mesmo sem o usuário ter pedido",
    ),
    _Rule(
        "inj.always_on",
        "held",
        _rx(
            r"\b(?:call|invoke|run|execute|send|chame|invoque|acione|envie|dispare)\b[^.\n]{0,80}\b(?:for (?:every|all|any|each)|"
            r"em (?:todo|toda|todos|todas|qualquer)|regardless of|independentemente d\w+)\b[^.\n]{0,30}\b"
            r"(?:request|requests|message|messages|query|queries|pedido|pedidos|mensagem|mensagens|"
            r"pergunta|perguntas|conversa|conversation)\b"
        ),
        "manda acionar algo em todo pedido, mesmo sem o usuário ter pedido",
    ),
    _Rule(
        "inj.exfiltration",
        "held",
        _rx(
            r"\b(?:send|post|upload|forward|exfiltrate|leak|transmit|envie|enviar|encaminhe|vaze|transmita|"
            r"publique)\b[^.\n]{0,80}\b(?:conversation|chat history|history|system prompt|credentials?|tokens?|"
            r"secrets?|passwords?|api keys?|conversa|historico|credenciais|senhas?|segredos?|chaves?)\b"
        ),
        "manda enviar conversa, credenciais ou segredos para fora",
        negatable=True,
    ),
    _Rule(
        "inj.prompt_leak",
        "held",
        _rx(
            r"\b(?:reveal|show|print|repeat|display|output|leak|mostre|revele|repita|imprima|exiba)\b[^.\n]{0,30}"
            r"\b(?:system prompt|your instructions|hidden instructions|prompt do sistema|suas instrucoes|"
            r"instrucoes ocultas|instrucoes internas)\b"
        ),
        "manda revelar o prompt ou as instruções internas",
        negatable=True,
    ),
    _Rule(
        "inj.tool_interference",
        "held",
        _rx(
            r"\b(?:do not|don't|never|nao|nunca|evite|avoid)\b[^.\n]{0,30}\b(?:use|call|invoke|trust|utilize|"
            r"usar|chamar|confie|confiar|utilizar)\b[^.\n]{0,40}\b(?:other|another|different|outras?|outros?|"
            r"demais|qualquer outr\w+)\b[^.\n]{0,20}\b(?:tools?|agents?|connectors?|plugins?|ferramentas?|"
            r"agentes?|conectores?)"
        ),
        "manda não usar as outras ferramentas, agentes ou plugins",
    ),
    _Rule(
        "inj.tool_interference",
        "held",
        _rx(
            r"\b(?:only use|use only|use apenas|use so|use somente|use exclusivamente|prefer only)\b[^.\n]{0,30}"
            r"\b(?:this|these|esta|este|esse|essa|nossa|nosso|nossos|nossas)\b[^.\n]{0,15}"
            r"\b(?:tool|tools|plugin|ferramenta|ferramentas|agent|agente)"
        ),
        "tenta fazer o roteador usar só este plugin, em vez das outras ferramentas",
    ),
    _Rule(
        "inj.external_instructions",
        "held",
        _rx(
            r"\b(?:follow|obey|read|load|fetch|download|retrieve|siga|obedeca|leia|carregue|baixe|busque|consulte)"
            r"\b[^.\n]{0,50}\b(?:instructions?|instrucoes|rules?|regras|commands?|comandos|prompt)\b[^.\n]{0,40}"
            r"\b(?:from|at|in|em|de|no|na)\b[^.\n]{0,20}https?://"
        ),
        "manda buscar ou seguir instruções de um endereço externo",
    ),
    _Rule(
        "inj.credential_request",
        "held",
        _rx(
            r"\b(?:ask|request|collect|require|peca|pedir|solicite|solicitar|colete|exija|pergunte)\b[^.\n]{0,50}"
            r"\b(?:passwords?|senhas?|pin|cvv|cvc|card number|numero do cartao|cartao de credito|tokens?|"
            r"api key|chave de api|2fa|codigo de verificacao|verification code|otp|secrets?)\b"
        ),
        "manda pedir ao usuário senha, token ou dado de pagamento",
        negatable=True,
    ),
    _Rule(
        "text.control_tokens",
        "block",
        _rx(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext|begin_of_text|end_of_text|eot_id|"
            r"start_header_id|end_header_id)\|>|\[/?inst\]|<</?sys>>|</?(?:system|assistant|developer)(?:\s[^>]*)?>"
        ),
        "marcador de papel de chat (<|im_start|>, [INST], <system>…): tenta forjar mensagens de sistema",
    ),
    _Rule(
        "text.data_uri",
        "held",
        _rx(r"data:[a-z0-9.+/-]{1,60};base64,"),
        "conteúdo embutido em data: URI",
    ),
    _Rule(
        "promo.directive",
        "warning",
        _rx(
            r"\b(?:always|sempre|obrigatoriamente|must|deve)\b[^.\n]{0,25}\b(?:recommend|promote|prefer|"
            r"recomende\w*|indique\w*|sugira|promova|priorize)\b"
        ),
        "diretriz de promoção: manda sempre recomendar algo",
    ),
)

_BLOB_RE = re.compile(r"[A-Za-z0-9+/_-]{160,}={0,2}")
_HEX_BLOB_RE = re.compile(r"(?:[0-9a-fA-F]{2}){60,}")
_BIDI_RE = re.compile("[؜‎‏‪-‮⁦-⁩]")
# caracteres de formato Unicode (categoria Cf: largura zero, marca de byte, hífen suave…) e
# seletores de variação: são achados, e somem do texto que as regras de linguagem leem
_FORMAT_CHARS = "".join(
    ch
    for ch in map(chr, range(0x10000))
    if unicodedata.category(ch) == "Cf" and not _BIDI_RE.match(ch)
)
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(_FORMAT_CHARS)}︀-️\U000e0100-\U000e01ef]")
_TAG_CHARS_RE = re.compile("[\U000e0000-\U000e007f]")
_HIDDEN_RE = re.compile(f"{_ZERO_WIDTH_RE.pattern}|{_BIDI_RE.pattern}|{_TAG_CHARS_RE.pattern}")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class _Comment(NamedTuple):
    line: int  # a linha em que o comentário começa
    inner: str  # o que está entre <!-- e -->
    whole: str  # o comentário inteiro


def _html_comments(text: str) -> Iterator[_Comment]:
    """Os comentários HTML fechados, numa passada só (linear, mesmo com muitos '<!--')."""
    pos = counted = newlines = 0
    while (start := text.find("<!--", pos)) != -1:
        end = text.find("-->", start + 4)
        if end == -1:
            return
        newlines += text.count("\n", counted, start)
        counted = start
        yield _Comment(newlines + 1, text[start + 4 : end], text[start : end + 3])
        pos = end + 3


_URL_RE = re.compile(r"https?://[^\s)>\]\"'`<]+", re.IGNORECASE)
_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((https?://[^)\s]+)\)|<img\b[^>]*?\bsrc\s*=\s*[\"']?(https?://[^\"'\s>]+)",
    re.IGNORECASE,
)
_TEMPLATED_RE = re.compile(r"\{\{|\$\{|%7b|\{[a-z_]+\}|<[a-z_]+>", re.IGNORECASE)
_BACKTICK_REF_RE = re.compile(r"(?<!`)`([a-z0-9][a-z0-9_.-]{0,62})/([a-z0-9_]+)`(?!`)")
_TLDS = frozenset(
    "com org net io br gov edu dev app ai co info me us uk eu de fr es pt it nl ru cn jp in au ca".split()
)
_NOT_OWNERS = frozenset(
    {"references", "assets", "skills", "knowledge", "scripts", "docs", "src", "bin", "etc", "usr"}
)


def _negated(folded_line: str, start: int) -> bool:
    before = folded_line[max(0, start - 30) : start]
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", before) for n in _NEGATORS)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EVIDENCE_CHARS else text[: EVIDENCE_CHARS - 1] + "…"


@dataclass
class _Hit:
    level: Level
    code: str
    message: str
    line: int
    evidence: str
    subject: str = ""  # o conteúdo inteiro (a linha toda): é o que a aprovação cobre


def _hits_to_findings(path: str, hits: Sequence[_Hit]) -> list[Finding]:
    """No máximo ``MAX_PER_CODE`` achados por código e arquivo; o resto vira uma contagem.

    A contagem cobre o conteúdo de todas as ocorrências que ela resume: aprovar "e mais N"
    não aprova uma ocorrência nova ou alterada numa versão seguinte.
    """
    out: list[Finding] = []
    per_code: dict[tuple[str, str], list[_Hit]] = {}
    for hit in hits:
        per_code.setdefault((hit.code, hit.message), []).append(hit)
    for (code, message), group in per_code.items():
        for hit in group[:MAX_PER_CODE]:
            where = f"{path}:{hit.line}" if hit.line else path
            out.append(Finding(hit.level, code, message, where, hit.evidence, hit.subject))
        rest = group[MAX_PER_CODE:]
        if rest:
            out.append(
                Finding(
                    rest[0].level,
                    code,
                    f"{message} (mais {len(rest)} ocorrência(s))",
                    path,
                    subject="\n".join(h.subject or h.evidence for h in rest),
                )
            )
    return out


def _scan_paragraphs(lines: Sequence[str], scope: str, hits: list[_Hit]) -> None:
    """Regras de linguagem sobre cada parágrafo inteiro: a frase quebrada em duas linhas
    ("Ignore all" / "previous instructions") também conta. Regra que já casou numa linha do
    parágrafo não se repete."""
    paragraphs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for number, raw in enumerate(lines, start=1):
        if raw.strip():
            current.append((number, raw.strip()))
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    for para in paragraphs:
        if len(para) < 2:
            continue
        first, last = para[0][0], para[-1][0]
        joined = " ".join(text for _, text in para)
        plain = fold(_HIDDEN_RE.sub("", unicodedata.normalize("NFKC", joined)))
        done = {h.code for h in hits if first <= h.line <= last}
        for rule in TEXT_RULES:
            if rule.code in done or (rule.scopes is not None and scope not in rule.scopes):
                continue
            for match in rule.pattern.finditer(plain):
                if rule.negatable and _negated(plain, match.start()):
                    continue
                hits.append(_Hit(rule.level, rule.code, rule.message, first, _clip(joined), joined))
                break


def scan_text(
    text: str,
    path: str,
    *,
    scope: str = "description",
    plugin: str | None = None,
    local_owners: Iterable[str] = (),
    structural_only: bool = False,
) -> list[Finding]:
    """Varre um texto que pode acabar no prompt do LLM (``scope``: description, playbook, knowledge).

    Com ``structural_only`` só procura o que se esconde na forma do texto (caracteres
    invisíveis e blocos codificados), sem as regras de linguagem: serve para os
    arquivos que o router não lê (README, ``references/``) e para o HTML cru de um documento.
    """
    hits: list[_Hit] = []
    owners = set(local_owners)
    lines = text.split("\n")
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        # caracteres invisíveis (inspecionados na linha original)
        if _TAG_CHARS_RE.search(raw):
            hits.append(
                _Hit(
                    "block",
                    "text.hidden_unicode",
                    "caracteres Unicode de 'tag' (invisíveis) escondendo texto",
                    number,
                    "",
                )
            )
        if _BIDI_RE.search(raw):
            hits.append(
                _Hit(
                    "block",
                    "text.hidden_unicode",
                    "caracteres de controle de direção do texto (bidi), usados para disfarçar conteúdo",
                    number,
                    _clip(raw),
                )
            )
        body = raw[1:] if number == 1 and raw.startswith("﻿") else raw
        if _ZERO_WIDTH_RE.search(body):
            hits.append(
                _Hit(
                    "held",
                    "text.zero_width",
                    "caracteres de largura zero (invisíveis) no texto",
                    number,
                    _clip(body.replace("​", "⟨ZWSP⟩")),
                )
            )
        if _CONTROL_RE.search(raw):
            hits.append(
                _Hit("held", "text.control_chars", "caracteres de controle no texto", number, "")
            )
        if (
            any(ch.isalpha() for ch in raw) and (_BLOB_RE.search(raw) or _HEX_BLOB_RE.search(raw))
        ) and not _only_urls(raw):
            hits.append(
                _Hit(
                    "held",
                    "text.encoded_blob",
                    "trecho longo que parece codificado (base64/hex), que pode esconder instruções",
                    number,
                    _clip(raw),
                )
            )
        if structural_only:
            continue
        # regras de linguagem: casam no texto sem acento, minúsculo e sem invisíveis
        plain = fold(_HIDDEN_RE.sub("", unicodedata.normalize("NFKC", raw)))
        for rule in TEXT_RULES:
            if rule.scopes is not None and scope not in rule.scopes:
                continue
            for match in rule.pattern.finditer(plain):
                if rule.negatable and _negated(plain, match.start()):
                    continue
                hits.append(_Hit(rule.level, rule.code, rule.message, number, _clip(raw)))
                break
        if scope != "description":
            for image in _IMAGE_RE.finditer(raw):
                image_url = image.group(1) or image.group(2)
                if _TEMPLATED_RE.search(image_url):
                    hits.append(
                        _Hit(
                            "held",
                            "text.markdown_exfil",
                            "imagem remota com variável no endereço: pode levar dados da conversa para fora",
                            number,
                            _clip(raw),
                        )
                    )
                else:
                    hits.append(
                        _Hit(
                            "warning",
                            "text.remote_image",
                            "imagem remota: quem renderiza a resposta faz uma requisição a um servidor externo",
                            number,
                            _clip(image_url),
                        )
                    )
            if plugin is not None:
                _scan_refs(raw, number, plugin, owners, hits)
    if not structural_only:
        _scan_paragraphs(lines, scope, hits)
    if structural_only:
        _fill_subjects(hits, lines)
        return _hits_to_findings(path, hits)
    # comentários HTML (invisíveis quando o Markdown é renderizado) podem passar várias linhas
    for comment in _html_comments(text):
        if len(comment.inner.strip()) >= 20:
            line = comment.line
            hits.append(
                _Hit(
                    "held",
                    "text.html_comment",
                    "comentário HTML com conteúdo: não aparece quando o texto é renderizado, mas o modelo lê",
                    line,
                    _clip(comment.inner),
                    comment.whole,
                )
            )
    _fill_subjects(hits, lines)
    findings = _hits_to_findings(path, hits)
    # hosts nas URLs do texto
    hosts: list[str] = []
    for url in _URL_RE.findall(text):
        host = _host_of(url)
        if not host:
            continue
        kind = classify_host(host)
        if kind == "metadata":
            findings.append(
                Finding(
                    "block",
                    "net.metadata_host",
                    f"referência ao endpoint de metadados da nuvem ({host})",
                    path,
                    _clip(url),
                )
            )
        elif host not in hosts:
            hosts.append(host)
    if hosts and scope != "description":
        shown = ", ".join(hosts[:8]) + (f" (+{len(hosts) - 8})" if len(hosts) > 8 else "")
        findings.append(Finding("note", "text.urls", f"o texto cita endereços: {shown}", path))
    return findings


def _fill_subjects(hits: Sequence[_Hit], lines: Sequence[str]) -> None:
    """O que a aprovação de cada achado de linha cobre: a linha inteira, não só o trecho mostrado."""
    for hit in hits:
        if not hit.subject and 0 < hit.line <= len(lines):
            hit.subject = lines[hit.line - 1]


def _only_urls(line: str) -> bool:
    return not _URL_RE.sub("", line).strip(" -*>\t")


def _scan_refs(raw: str, line: int, plugin: str, local_owners: set[str], hits: list[_Hit]) -> None:
    for match in _BACKTICK_REF_RE.finditer(raw):
        owner, name = match.group(1), match.group(2)
        if "." in owner:
            other, _, rest = owner.partition(".")
            if rest in _TLDS:
                continue  # `exemplo.com/caminho`
            if other != plugin:
                hits.append(
                    _Hit(
                        "held",
                        "capability.foreign_ref",
                        f"manda usar uma capacidade de outro plugin ({owner}/{name}): um plugin só "
                        "deve orientar o uso do que ele mesmo traz",
                        line,
                        _clip(raw),
                    )
                )
            continue
        if owner in local_owners or owner in _NOT_OWNERS:
            continue
        hits.append(
            _Hit(
                "warning",
                "capability.unknown_ref",
                f"cita `{owner}/{name}`, que não é um conector ou agente declarado neste plugin",
                line,
                _clip(raw),
            )
        )


# --------------------------------------------------------------------------
# segredos

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chave de API (sk-…)", re.compile(r"\bsk-(?:ant-|proj-|live-)?[A-Za-z0-9_-]{20,}")),
    (
        "token do GitHub",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    ),
    ("chave de acesso da AWS", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("token do Slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("chave do Google", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    (
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,1024}\.[A-Za-z0-9_-]{10,1024}\.[A-Za-z0-9_-]{10,1024}"),
    ),
    ("chave privada", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("token Bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{24,}")),
    (
        "senha ou token atribuído a um valor",
        re.compile(
            r"""(?ix)\b(?:password|passwd|senha|secret|api[_-]?key|token|auth[_-]?token)\b\s*["']?\s*[:=]\s*["']?
            (?!\$\{|env:|enc:|<|\*|x{3}|\.{3}|your|seu|sua|exemplo|example|changeme|null|none|true|false)
            [A-Za-z0-9/+_=.-]{16,}"""
        ),
    ),
)


def scan_secrets(text: str, path: str) -> list[Finding]:
    findings: list[Finding] = []
    seen: Counter[str] = Counter()
    for number, line in enumerate(text.split("\n"), start=1):
        for label, pattern in _SECRET_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            # um segredo por linha basta: a mesma chave costuma casar com mais de um padrão
            seen[label] += 1
            if seen[label] <= MAX_PER_CODE:
                value = match.group(0)
                findings.append(
                    Finding(
                        "block",
                        "secret.leaked",
                        f"parece haver um segredo no pacote ({label}); segredos entram pela "
                        "configuração (config com sensitive: true), nunca no pacote",
                        f"{path}:{number}",
                        f"{value[:4]}…({len(value)} caracteres)",
                    )
                )
            break
    return findings


# --------------------------------------------------------------------------
# endereços

METADATA_HOSTS = frozenset(
    {"169.254.169.254", "metadata.google.internal", "metadata", "100.100.100.200", "fd00:ec2::254"}
)
_PRIVATE_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
    ".svc",
    ".cluster.local",
    ".intranet",
    ".corp",
)


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


# formas numéricas de IPv4 que o resolvedor aceita: 127.1, 0x7f.0.0.1, 0177.0.0.1, 2130706433
_NUMERIC_HOST_RE = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+)){0,3}", re.I)


def _ip_of(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """O IP que o nome representa, como o sistema o resolveria (ou ``None`` se é um nome)."""
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
            host.split("%", 1)[0]
        )
    except ValueError:
        if not _NUMERIC_HOST_RE.fullmatch(host):
            return None
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return None
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is not None:
            return embedded
        if 1 < int(ip) < 2**32:  # ::a.b.c.d (IPv4 "compatível", obsoleto)
            return ipaddress.IPv4Address(int(ip))
    return ip


_IDEO_DOTS = str.maketrans({"。": ".", "｡": ".", "．": "."})


def classify_host(host: str) -> str:
    """``metadata`` | ``private`` | ``public`` (nomes de rede interna e IPs não públicos)."""
    host = unicodedata.normalize("NFKC", host).translate(_IDEO_DOTS)
    host = host.lower().strip().strip("[]").rstrip(".")
    if host in METADATA_HOSTS:
        return "metadata"
    ip = _ip_of(host)
    if ip is None:
        if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES) or "." not in host:
            return "private"
        return "public"
    if str(ip) in METADATA_HOSTS:
        return "metadata"
    if not ip.is_global or ip.is_multicast:
        return "private"
    return "public"


_MARK = "\x00"  # lugar de um valor que só quem instala informa
_AUTHORITY_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*|\x00)://([^/?#]*)")


def _author_url(template: str, config: Mapping[str, object]) -> str:
    """O endereço como o pacote o define: config com padrão vira o padrão; sem padrão, uma marca."""

    def repl(match: re.Match[str]) -> str:
        default = getattr(config.get(match.group(1)), "default", None)
        return _MARK if default is None else str(default)

    return PLACEHOLDER_RE.sub(repl, template)


def _split_host_port(authority: str) -> tuple[str, str]:
    if authority.startswith("["):
        end = authority.find("]")
        if end != -1:
            rest = authority[end + 1 :]
            return authority[1:end], rest[1:] if rest.startswith(":") else ""
    host, sep, port = authority.rpartition(":")
    if sep and (port.isdigit() or _MARK in port):
        return host, port
    return authority, ""


def _fixed_domain(host: str) -> str | None:
    """O domínio que o pacote fixa num host montado com config (``${config.T}.fornecedor.com``)."""
    tail = host.rsplit(_MARK, 1)[-1].strip(".-")
    if not tail:
        return None
    if "." in tail or tail == "localhost" or ("." + tail).endswith(_PRIVATE_SUFFIXES):
        return tail
    return None


_CREDENTIAL_PARAM_RE = re.compile(
    r"api[_-]?key|apikey|access[_-]?token|auth(?:orization)?|client[_-]?secret|key|password"
    r"|passwd|secret|senha|sig|signature|token",
    re.IGNORECASE,
)


def _credential_in_url(rendered: str, where: str) -> list[Finding]:
    """Credencial na query do endereço (``?api_key=…``): ela vai em claro nos logs."""
    _, sep, query = rendered.partition("?")
    if not sep:
        return []
    for pair in query.partition("#")[0].split("&"):
        name = unquote(pair.partition("=")[0]).strip()
        if _CREDENTIAL_PARAM_RE.fullmatch(name):
            return [
                Finding(
                    "block",
                    "net.credential_in_url",
                    f"o endereço leva a credencial '{name}' na query: ela vai em claro nos logs; "
                    "use config sensitive e auth_token",
                    where,
                    _clip(name),
                )
            ]
    return []


def scan_endpoint(
    template: str, where: str, config: Mapping[str, object]
) -> tuple[list[Finding], str | None]:
    """Triagem do endereço de um conector ou agente; devolve também o host que o pacote fixa.

    Um host escrito no pacote (ou no ``default`` de uma config) decide para onde vão os
    pedidos e o token: rede interna e host fixo exigem aprovação, e o host e a porta entram
    na impressão digital do achado, então trocar de host numa versão nova pede aprovação de
    novo. Quando o host inteiro vem de quem instala, não há o que triar no pacote.
    """
    findings: list[Finding] = []
    literal = _AUTHORITY_RE.match(template)
    if literal is not None and "@" in literal.group(2):
        findings.append(
            Finding(
                "block",
                "net.userinfo",
                "o endereço traz usuário e senha (ou um '@' antes do host); use config "
                "sensitive e auth_token",
                where,
                _clip(literal.group(2).rsplit("@", 1)[-1]),
            )
        )
    rendered = _author_url(template, config)
    match = _AUTHORITY_RE.match(rendered)
    if match is None:
        return findings, None  # o endereço inteiro vem de quem instala
    scheme, authority = match.group(1).lower(), match.group(2)
    scheme_label = "…" if scheme == _MARK else scheme
    if "@" in authority:
        if not findings:
            findings.append(
                Finding(
                    "block",
                    "net.userinfo",
                    "o endereço traz usuário e senha (no valor padrão da config); use config "
                    "sensitive e auth_token",
                    where,
                    _clip(authority.rsplit("@", 1)[-1].replace(_MARK, "…")),
                )
            )
        authority = authority.rsplit("@", 1)[-1]
    findings.extend(_credential_in_url(rendered, where))
    host, port = _split_host_port(authority)
    if _MARK in host:
        # o pacote escreveu parte do host: se esse trecho já aponta para metadados ou para a
        # rede interna, a triagem não depende do que quem instala informar
        literal = host.replace(_MARK, "")
        literal_kind = (
            classify_host(literal) if "." in literal or literal == "localhost" else "public"
        )
        if literal_kind in ("metadata", "private"):
            kind, shown = literal_kind, f"{literal}…"
        else:
            fixed = _fixed_domain(host)
            if fixed is None:
                return findings, None  # o host vem de quem instala
            kind, shown = classify_host(fixed), f"*.{fixed}"
    else:
        host = host.strip("[]")
        if not host:
            return findings, None
        kind, shown = classify_host(host), host.lower()
    target = f"{scheme_label}://{shown}" + (f":{port}" if port and _MARK not in port else "")
    if kind == "metadata":
        findings.append(
            Finding(
                "block",
                "net.metadata_host",
                f"o endereço aponta para o endpoint de metadados da nuvem ({shown})",
                where,
                target,
            )
        )
        return findings, None
    if kind == "private":
        findings.append(
            Finding(
                "held",
                "net.private_host",
                f"o plugin traz o endereço '{shown}', da rede interna ou da própria máquina: aprove "
                "só se for esperado (o ideal é o endereço vir da config, escolhido por quem instala)",
                where,
                target,
            )
        )
        return findings, None
    findings.append(
        Finding(
            "held",
            "net.fixed_host",
            f"o plugin fixa o host '{shown}': os pedidos (e o token, se houver) vão para ele. "
            "Aprove se confia nesse destino; uma versão que troque o host pede aprovação de novo",
            where,
            target,
        )
    )
    if scheme == "http":
        findings.append(
            Finding(
                "warning",
                "net.insecure_http",
                f"'{shown}' é acessado por http (sem TLS); o tráfego, inclusive o token, vai em claro",
                where,
            )
        )
    return findings, shown


# --------------------------------------------------------------------------
# o plugin inteiro


def scan_manifest(
    manifest: PluginManifest,
    *,
    switchboard_version: str | None = None,
    skill_bodies: Mapping[str, str] | None = None,
) -> list[Finding]:
    """Regras do manifesto: configuração, segredos, endereços e componentes.

    ``skill_bodies`` (skill -> texto) entra na impressão digital dos playbooks ``always``:
    aprovar um playbook que vale para todo pedido aprova aquele texto, não o nome dele.
    """
    findings: list[Finding] = []
    config = manifest.config

    wanted = manifest.requires.switchboard
    if switchboard_version and wanted and not satisfies(switchboard_version, wanted):
        findings.append(
            Finding(
                "block",
                "requires.switchboard",
                f"o plugin exige Switchboard {wanted}; esta instalação é a {switchboard_version}",
                "plugin.yaml: requires.switchboard",
            )
        )

    for key, field in config.items():
        if field.sensitive and field.default is not None:
            findings.append(
                Finding(
                    "block",
                    "secret.default",
                    f"a config '{key}' é sensitive e traz um valor padrão: segredo não vai no pacote",
                    f"plugin.yaml: config.{key}",
                )
            )

    used: set[str] = set()
    hosts: list[str] = []

    def check_url_placeholders(text: str | None, where: str) -> None:
        """Num endereço só entram config comuns e obrigatórias (ou com padrão)."""
        if not text:
            return
        for key in PLACEHOLDER_RE.findall(text):
            used.add(key)
            field = config.get(key)
            if field is None:
                findings.append(
                    Finding(
                        "block",
                        "config.undeclared",
                        f"${{config.{key}}} não está declarada em config",
                        where,
                    )
                )
            elif not field.required and field.default is None:
                findings.append(
                    Finding(
                        "block",
                        "config.optional_in_url",
                        f"a config '{key}' é opcional e sem padrão, mas o endereço depende dela: "
                        "use required: true ou um default",
                        where,
                    )
                )
            elif field.sensitive:
                findings.append(
                    Finding(
                        "block",
                        "config.sensitive_misplaced",
                        f"a config sensitive '{key}' só pode ser usada em auth_token "
                        "(num endereço ela vazaria em logs)",
                        where,
                    )
                )

    for kind, items in (("connectors", manifest.connectors), ("agents", manifest.agents)):
        for item in items:
            base = f"plugin.yaml: {kind}.{item.name}"
            check_url_placeholders(item.url, f"{base}.url")
            endpoint_findings, host = scan_endpoint(item.url, f"{base}.url", config)
            findings.extend(endpoint_findings)
            if host and host not in hosts:
                hosts.append(host)
            token = item.auth_token
            if token:
                exact = EXACT_PLACEHOLDER_RE.match(token.strip())
                if exact is None:
                    findings.append(
                        Finding(
                            "block",
                            "secret.literal",
                            "auth_token precisa ser uma referência ${config.CHAVE}; token escrito "
                            "no pacote vaza para quem o instalar",
                            f"{base}.auth_token",
                        )
                    )
                else:
                    key = exact.group(1)
                    used.add(key)
                    field = config.get(key)
                    if field is None:
                        findings.append(
                            Finding(
                                "block",
                                "config.undeclared",
                                f"${{config.{key}}} não está declarada em config",
                                f"{base}.auth_token",
                            )
                        )
                    elif not field.sensitive:
                        findings.append(
                            Finding(
                                "block",
                                "config.token_not_sensitive",
                                f"a config '{key}' alimenta um token e precisa de sensitive: true "
                                "(senão ficaria visível e em texto puro)",
                                f"{base}.auth_token",
                            )
                        )
        for item in items:
            base = f"plugin.yaml: {kind}.{item.name}"
            if kind == "connectors" and not item.allowed_tools:  # type: ignore[attr-defined]
                findings.append(
                    Finding(
                        "warning",
                        "connector.no_allowed_tools",
                        "sem allowed_tools: todas as tools que o servidor expuser (hoje ou depois de "
                        "uma atualização dele) ficam disponíveis ao roteador",
                        base,
                    )
                )
            if kind == "agents" and item.allow_cross_origin:  # type: ignore[attr-defined]
                findings.append(
                    Finding(
                        "held",
                        "agent.cross_origin",
                        "allow_cross_origin: o Agent Card pode apontar o JSON-RPC para outro host "
                        "que não o do cadastro",
                        base,
                    )
                )

    for pb in manifest.playbooks:
        if pb.always:
            body = (skill_bodies or {}).get(pb.skill, "")
            findings.append(
                Finding(
                    "held",
                    "playbook.always_on",
                    f"o playbook '{pb.skill}' entra em todos os pedidos do roteador (always: true): "
                    "confira o texto antes de aprovar",
                    f"plugin.yaml: playbooks.{pb.skill}",
                    subject="\n".join(
                        [body, "applies_to: " + ",".join(pb.applies_to), "triggers: "]
                        + list(pb.triggers)
                    ),
                )
            )
        broad = [t for t in pb.triggers if broad_trigger(t)]
        if broad:
            findings.append(
                Finding(
                    "warning",
                    "playbook.broad_trigger",
                    f"gatilho(s) genérico(s) demais, ignorado(s) na escolha do playbook: "
                    f"{', '.join(broad[:8])}",
                    f"plugin.yaml: playbooks.{pb.skill}",
                )
            )

    for key in config:
        if key not in used:
            findings.append(
                Finding(
                    "note",
                    "config.unused",
                    f"a config '{key}' não é usada",
                    f"plugin.yaml: config.{key}",
                )
            )
    if hosts:
        findings.append(
            Finding(
                "note",
                "net.hosts",
                "o plugin vai contatar: " + ", ".join(hosts),
                "plugin.yaml",
            )
        )

    # textos do manifesto que o roteador lê
    owners = manifest.owners
    texts = [("plugin.yaml: description", manifest.description)]
    for kind, items in (
        ("connectors", manifest.connectors),
        ("agents", manifest.agents),
        ("knowledge_bases", manifest.knowledge_bases),
    ):
        texts.extend((f"plugin.yaml: {kind}.{i.name}.description", i.description) for i in items)
    for key, examples in manifest.routing_examples.items():
        texts.extend((f"plugin.yaml: routing_examples.{key}", example) for example in examples)
    for where, text in texts:
        if text:
            findings.extend(scan_text(text, where, scope="description", local_owners=owners))
    return findings


def scan_skills(
    skills: Sequence[Skill], manifest: PluginManifest, files: Mapping[str, bytes]
) -> list[Finding]:
    """Varre o arquivo ``SKILL.md`` inteiro (frontmatter e corpo), com os números de linha dele."""
    findings: list[Finding] = []
    owners = manifest.owners
    for skill in skills:
        raw = files[skill.path].decode("utf-8", errors="replace")
        findings.extend(
            scan_text(raw, skill.path, scope="playbook", plugin=manifest.name, local_owners=owners)
        )
    return findings


def scan_documents(documents: Mapping[str, str], manifest: PluginManifest) -> list[Finding]:
    findings: list[Finding] = []
    for path, text in documents.items():
        findings.extend(
            scan_text(
                text, path, scope="knowledge", plugin=manifest.name, local_owners=manifest.owners
            )
        )
    return findings


def dedupe(findings: Iterable[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.level, f.code, f.where, f.message)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
