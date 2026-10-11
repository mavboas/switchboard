"""Versões semânticas e a restrição ``requires.switchboard`` dos plugins.

Só o necessário: validar ``MAJOR.MINOR.PATCH[-pre][+build]``, ordenar versões (para
distinguir atualização de volta atrás) e conferir uma restrição simples como
``>=0.2,<1``.
"""

from __future__ import annotations

import re

_ID = r"[0-9A-Za-z-]+"
SEMVER_RE = re.compile(
    rf"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-({_ID}(?:\.{_ID})*))?(?:\+({_ID}(?:\.{_ID})*))?$"
)
_CLAUSE_RE = re.compile(r"^(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+){0,2})$")


def is_semver(text: str) -> bool:
    return bool(SEMVER_RE.match(text))


def _pre_key(pre: str | None) -> tuple:
    if pre is None:
        return (1,)  # sem pré-lançamento vem depois de qualquer pré-lançamento
    parts = []
    for ident in pre.split("."):
        parts.append((0, int(ident), "") if ident.isdigit() else (1, 0, ident))
    return (0, tuple(parts))


def version_key(version: str) -> tuple:
    """Chave de ordenação (a parte ``+build`` não conta, como no semver)."""
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"versão inválida: {version!r}")
    major, minor, patch, pre, _build = match.groups()
    return (int(major), int(minor), int(patch), _pre_key(pre))


def compare_versions(a: str, b: str) -> int:
    ka, kb = version_key(a), version_key(b)
    return (ka > kb) - (ka < kb)


def _loose(version: str) -> tuple[int, int, int]:
    nums = [int(x) for x in version.split(".")]
    nums += [0] * (3 - len(nums))
    return nums[0], nums[1], nums[2]


def parse_specifier(spec: str) -> list[tuple[str, tuple[int, int, int]]]:
    clauses = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        match = _CLAUSE_RE.match(raw)
        if not match:
            raise ValueError(
                f"restrição inválida: {raw!r} (use algo como '>=0.2' ou '>=0.2,<1', "
                "com os operadores >=, >, <=, <, == e !=)"
            )
        clauses.append((match.group(1), _loose(match.group(2))))
    if not clauses:
        raise ValueError("restrição vazia")
    return clauses


def satisfies(version: str, spec: str) -> bool:
    """``version`` (ex.: ``0.2.0``) cumpre a restrição ``spec`` (ex.: ``>=0.2,<1``)?"""
    current = _loose(version.split("-")[0].split("+")[0])
    for op, wanted in parse_specifier(spec):
        ok = {
            ">=": current >= wanted,
            ">": current > wanted,
            "<=": current <= wanted,
            "<": current < wanted,
            "==": current == wanted,
            "!=": current != wanted,
        }[op]
        if not ok:
            return False
    return True
