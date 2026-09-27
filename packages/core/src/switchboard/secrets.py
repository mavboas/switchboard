"""Segredos (chaves de API de modelos, tokens de agentes MCP).

Um campo de segredo aceita três formas:

* ``env:NOME`` — referência a uma variável de ambiente, lida em tempo de
  execução no processo que usa o segredo (recomendado: o valor nunca vai para
  o banco). No console/router só valem nomes liberados em
  ``SWITCHBOARD_ALLOWED_ENV_SECRETS`` (padrão ``*_API_KEY,MCP_*``) — assim
  quem edita a configuração não consegue ler outras variáveis do processo,
  como a própria ``SWITCHBOARD_SECRET_KEY``;
* ``enc:<token>`` — valor cifrado com Fernet a partir da chave mestra
  (``SWITCHBOARD_SECRET_KEY`` ou o arquivo gerado na primeira subida);
* texto puro — aceito só no modo framework (YAML), nunca gravado no banco.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
import re
import secrets
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .errors import SecretError

ENV_PREFIX = "env:"
ENC_PREFIX = "enc:"
MASK = "••••••"  # o que a UI mostra no lugar de um valor cifrado; ao salvar, mantém o atual
DEFAULT_ALLOWED_ENV = ("*_API_KEY", "MCP_*")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KDF_SALT = b"switchboard/secretbox/v1"
_KDF_ROUNDS = 200_000


def _fernet_key(passphrase: str) -> bytes:
    # Aceita qualquer string como chave mestra. PBKDF2 deixa caro testar
    # senhas fracas; a chave gerada automaticamente já tem 256 bits.
    raw = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), _KDF_SALT, _KDF_ROUNDS)
    return base64.urlsafe_b64encode(raw)


def parse_patterns(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ALLOWED_ENV
    items = value.split(",") if isinstance(value, str) else list(value)
    return tuple(p.strip() for p in items if p and p.strip())


def env_ref_allowed(name: str, patterns: Iterable[str] = DEFAULT_ALLOWED_ENV) -> bool:
    """Diz se ``env:name`` pode ser usado a partir da configuração editável."""
    if not _ENV_NAME_RE.match(name) or name.upper().startswith("SWITCHBOARD_"):
        return False
    return any(fnmatch.fnmatchcase(name.upper(), p.upper()) for p in patterns)


def is_env_ref(value: str | None) -> bool:
    return bool(value) and value.startswith(ENV_PREFIX)


def resolve_env(value: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve ``env:NOME`` sem restrições (modo YAML, arquivo do operador)."""
    if not value:
        return None
    if value.startswith(ENV_PREFIX):
        env = os.environ if environ is None else environ
        return env.get(value[len(ENV_PREFIX) :].strip()) or None
    return value


def load_master_key(value: str | None, path: str | None) -> str | None:
    """Chave mestra: a variável, se definida; senão o arquivo, criado na primeira vez.

    Console e router apontam para o mesmo arquivo (volume compartilhado); quem
    subir primeiro gera 256 bits aleatórios, o outro lê.
    """
    if value:
        return value
    if not path:
        return None
    file = Path(path)
    for _attempt in range(50):
        if file.exists():
            content = file.read_text(encoding="utf-8").strip()
            if content:
                return content
            time.sleep(0.1)  # o outro processo ainda está escrevendo
            continue
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        key = secrets.token_urlsafe(32)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(key)
        return key
    raise SecretError(f"não consegui ler a chave mestra em {path}")


class SecretBox:
    """Cifra e resolve segredos guardados no banco."""

    def __init__(self, key: str | None, allowed_env: str | Iterable[str] | None = None):
        self._fernet = Fernet(_fernet_key(key)) if key else None
        self.allowed_env = parse_patterns(allowed_env)

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def _check_env(self, name: str) -> None:
        if not env_ref_allowed(name, self.allowed_env):
            allowed = ", ".join(self.allowed_env) or "nenhuma"
            raise SecretError(
                f"A variável {name} não está liberada para uso como segredo (liberadas: {allowed}). "
                "Ajuste SWITCHBOARD_ALLOWED_ENV_SECRETS no console e no router se precisar."
            )

    def seal(self, value: str | None) -> str | None:
        """Prepara um segredo para gravar no banco.

        Vazio vira ``None``; ``env:`` (se liberada) e ``enc:`` são mantidos;
        texto puro é cifrado (exige chave mestra).
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
            self._check_env(name)
            return ENV_PREFIX + name
        if value.startswith(ENC_PREFIX):
            return value
        if self._fernet is None:
            raise SecretError(
                "Para salvar a chave cifrada, o console precisa de uma chave mestra "
                "(SWITCHBOARD_SECRET_KEY ou SWITCHBOARD_SECRET_KEY_FILE). Outra opção: env:NOME."
            )
        return ENC_PREFIX + self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def open(self, stored: str | None, environ: Mapping[str, str] | None = None) -> str | None:
        """Resolve o valor utilizável de um segredo gravado."""
        if not stored:
            return None
        if stored.startswith(ENC_PREFIX):
            if self._fernet is None:
                raise SecretError(
                    "Há segredos cifrados no banco, mas não há chave mestra configurada."
                )
            try:
                token = stored[len(ENC_PREFIX) :].encode("ascii")
                return self._fernet.decrypt(token).decode("utf-8")
            except InvalidToken as exc:
                raise SecretError(
                    "Não foi possível decifrar um segredo: a chave mestra mudou?"
                ) from exc
        if stored.startswith(ENV_PREFIX):
            self._check_env(stored[len(ENV_PREFIX) :].strip())
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
