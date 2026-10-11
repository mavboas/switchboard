"""Cache de descoberta "stale-while-revalidate" (conectores MCP e agentes A2A).

Um servidor lento ou fora do ar nunca trava os pedidos: enquanto houver
resultado em cache (mesmo vencido), ele é devolvido na hora e a nova
descoberta roda em segundo plano. Só o primeiro pedido para um servidor ainda
desconhecido espera a descoberta (que o chamador limita com timeout).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

S = TypeVar("S", bound=BaseModel)
I = TypeVar("I")  # noqa: E741 - informação descoberta


def spec_key(spec: BaseModel) -> str:
    raw = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class DiscoveryCache(Generic[S, I]):
    def __init__(
        self,
        probe: Callable[[S], Awaitable[I]],
        *,
        online: Callable[[I], bool],
        name_of: Callable[[I], str],
        ttl_s: float = 30.0,
        offline_ttl_s: float = 15.0,
    ):
        self._probe = probe
        self._online = online
        self._name_of = name_of
        self.ttl_s = ttl_s
        self.offline_ttl_s = offline_ttl_s
        self._cache: dict[str, tuple[float, I]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._background: set[asyncio.Task[Any]] = set()

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
            return
        for key, (_, info) in list(self._cache.items()):
            if self._name_of(info) == name:
                self._cache.pop(key, None)

    async def get(self, spec: S, *, refresh: bool = False) -> I:
        key = spec_key(spec)
        cached = self._cache.get(key)
        if cached and not refresh:
            expires, info = cached
            if expires <= time.monotonic():
                self._revalidate(spec, key)  # devolve o que tem e atualiza por trás
            return info
        return await self._probe_and_store(spec, key, force=refresh)

    def _revalidate(self, spec: S, key: str) -> None:
        lock = self._locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            return  # já tem uma descoberta em andamento
        task = asyncio.get_running_loop().create_task(self._probe_and_store(spec, key, force=True))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _probe_and_store(self, spec: S, key: str, *, force: bool) -> I:
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(key)
            if cached and not force and cached[0] > time.monotonic():
                return cached[1]
            info = await self._probe(spec)
            ttl = self.ttl_s if self._online(info) else self.offline_ttl_s
            self._cache[key] = (time.monotonic() + ttl, info)
            return info


def describe_error(exc: BaseException) -> str:
    """Mensagem curta de uma exceção (abre ExceptionGroup do anyio/TaskGroup)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    text = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"[:400]
