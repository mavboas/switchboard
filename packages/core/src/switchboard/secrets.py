"""Segredos (chaves de API de modelos, tokens de agentes MCP).

Um campo de segredo aceita três formas:

* ``env:NOME`` — referência a uma variável de ambiente, lida em tempo de
  execução no processo que usa o segredo (recomendado: o valor nunca vai para
  o banco);
* ``enc:<token>`` — valor cifrado com Fernet usando ``SWITCHBOARD_SECRET_KEY``
  (é o que o console grava quando você cola a chave no formulário);
* texto puro — aceito só no modo framework (YAML), nunca gravado no banco.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Mapping

from cryptography.fernet import Fernet, InvalidToken

from .errors import SecretError

ENV_PREFIX = "env:"
ENC_PREFIX = "enc:"


def _fernet_key(passphrase: str) -> bytes:
    # Qualquer string serve como SWITCHBOARD_SECRET_KEY: derivamos 32 bytes
    # estáveis com SHA-256 para montar a chave Fernet.
    return base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode("utf-8")).digest())


def is_env_ref(value: str | None) -> bool:
    return bool(value) and value.startswith(ENV_PREFIX)


def resolve_env(value: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve ``env:NOME``; qualquer outro valor volta como veio."""
    if not value:
        return None
    if value.startswith(ENV_PREFIX):
        env = os.environ if environ is None else environ
        return env.get(value[len(ENV_PREFIX) :].strip()) or None
    return value


class SecretBox:
    """Cifra e resolve segredos guardados no banco."""

    def __init__(self, key: str | None):
        self._fernet = Fernet(_fernet_key(key)) if key else None

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def seal(self, value: str | None) -> str | None:
        """Prepara um segredo para gravar no banco.

        Vazio vira ``None``; ``env:`` e ``enc:`` são mantidos; texto puro é
        cifrado (exige ``SWITCHBOARD_SECRET_KEY``).
        """
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if value.startswith(ENV_PREFIX):
            name = value[len(ENV_PREFIX) :].strip()
            if not name:
                raise SecretError("Informe o nome da variável depois de 'env:'.")
            return ENV_PREFIX + name
        if value.startswith(ENC_PREFIX):
            return value
        if self._fernet is None:
            raise SecretError(
                "Para salvar a chave cifrada, defina SWITCHBOARD_SECRET_KEY no console e no "
                "router. Outra opção: use env:NOME_DA_VARIAVEL."
            )
        return ENC_PREFIX + self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def open(self, stored: str | None, environ: Mapping[str, str] | None = None) -> str | None:
        """Resolve o valor utilizável de um segredo gravado."""
        if not stored:
            return None
        if stored.startswith(ENC_PREFIX):
            if self._fernet is None:
                raise SecretError(
                    "Há segredos cifrados no banco, mas SWITCHBOARD_SECRET_KEY não está definida."
                )
            try:
                return self._fernet.decrypt(stored[len(ENC_PREFIX) :].encode("ascii")).decode(
                    "utf-8"
                )
            except InvalidToken as exc:
                raise SecretError(
                    "Não foi possível decifrar um segredo: SWITCHBOARD_SECRET_KEY mudou?"
                ) from exc
        return resolve_env(stored, environ)

    @staticmethod
    def describe(stored: str | None) -> str:
        """Texto seguro para exibir na UI (nunca mostra o valor)."""
        if not stored:
            return "—"
        if stored.startswith(ENV_PREFIX):
            return stored
        if stored.startswith(ENC_PREFIX):
            return "•••••• (cifrada)"
        return "•••••• (texto puro)"
