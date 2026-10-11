"""Achados da validação e da varredura de segurança de um plugin.

Cada achado tem um nível, que decide o que acontece com o plugin:

* ``block`` — o pacote é recusado e nada é instalado;
* ``held`` — o pacote é instalado, mas só fica ativo depois que um administrador
  aprova (no console) ou o aceita por código (``allow`` no YAML);
* ``warning`` — risco ou má prática que vale revisar, sem barrar nada;
* ``note`` — informação (hosts contatados, partes ignoradas numa importação…).

A varredura é uma triagem automática: ela pega o que é detectável por regra, mas
não substitui a revisão de quem aprova nem a barreira de fato, que é o plugin ser
só declarativo (o router nunca executa código de plugin).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import PluginError

Level = Literal["block", "held", "warning", "note"]
LEVELS: tuple[Level, ...] = ("block", "held", "warning", "note")  # do mais grave ao mais leve

_LINE_SUFFIX = re.compile(r":\d+$")

LEVEL_LABELS: dict[str, str] = {
    "block": "bloqueia a instalação",
    "held": "exige aprovação",
    "warning": "aviso",
    "note": "informação",
}


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    message: str
    where: str = ""
    evidence: str = ""
    # o conteúdo inteiro que a aprovação cobre (a linha toda, o texto do playbook, o host e a
    # porta…): entra só na impressão digital; ``evidence`` é o resumo que se mostra
    subject: str = field(default="", repr=False)
    # impressão digital já calculada (achados lidos do banco, onde o ``subject`` não fica)
    stored_fingerprint: str = field(default="", repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """Identifica o achado entre versões do mesmo plugin.

        Código + arquivo (sem o número da linha, que muda a cada edição) + o SHA-256 do
        conteúdo que disparou a regra (``subject``, ou o trecho mostrado): o mesmo problema
        continua "o mesmo" se o arquivo muda de lugar, e qualquer mudança no que foi
        aprovado (inclusive depois do trecho mostrado) vira um achado novo.
        """
        if self.stored_fingerprint:
            return self.stored_fingerprint
        where = _LINE_SUFFIX.sub("", self.where)
        basis = self.subject or self.evidence
        if basis:
            return f"{self.code}@{where}#{hashlib.sha256(basis.encode('utf-8')).hexdigest()}"
        return f"{self.code}@{where}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "where": self.where,
            "evidence": self.evidence,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            level=data.get("level", "note"),
            code=str(data.get("code") or ""),
            message=str(data.get("message") or ""),
            where=str(data.get("where") or ""),
            evidence=str(data.get("evidence") or ""),
            stored_fingerprint=str(data.get("fingerprint") or ""),
        )

    def __str__(self) -> str:
        where = f" ({self.where})" if self.where else ""
        return f"[{self.level}] {self.code}: {self.message}{where}"


def blocking(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "block"]


def held(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "held"]


def sorted_findings(findings: Iterable[Finding]) -> list[Finding]:
    order = {level: i for i, level in enumerate(LEVELS)}
    return sorted(findings, key=lambda f: (order.get(f.level, 9), f.code, f.where))


def counts(findings: Iterable[Finding]) -> dict[str, int]:
    out = dict.fromkeys(LEVELS, 0)
    for f in findings:
        out[f.level] = out.get(f.level, 0) + 1
    return out


def format_findings(findings: Iterable[Finding]) -> str:
    lines = []
    for f in sorted_findings(findings):
        where = f" — {f.where}" if f.where else ""
        evidence = f"\n      «{f.evidence}»" if f.evidence else ""
        lines.append(f"  - [{f.level}] {f.code}: {f.message}{where}{evidence}")
    return "\n".join(lines)


class PluginRejected(PluginError):
    """O pacote não pôde virar um plugin (ou tem achados que bloqueiam a instalação)."""

    def __init__(self, findings: Iterable[Finding]):
        self.findings = sorted_findings(findings)
        blocks = blocking(self.findings) or self.findings
        super().__init__("plugin recusado:\n" + format_findings(blocks))
