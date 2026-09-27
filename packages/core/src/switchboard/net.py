"""Utilitários de rede compartilhados pelos serviços HTTP."""

from __future__ import annotations

from collections.abc import Iterable


def host_allowed(hostname: str | None, allowed: Iterable[str]) -> bool:
    """Confere o nome de host do pedido contra uma allowlist (``*`` e ``*.dominio`` valem).

    Usado contra DNS rebinding: uma página maliciosa que resolve o próprio
    domínio para 127.0.0.1 continua mandando ``Host: dominio-malicioso``.
    """
    patterns = [p.strip().lower().strip("[]") for p in allowed if p and p.strip()]
    if "*" in patterns:
        return True
    if not hostname:
        return False
    host = hostname.lower().strip("[]")
    for pattern in patterns:
        if pattern.startswith("*.") and host.endswith(pattern[1:]):
            return True
        if host == pattern:
            return True
    return False
