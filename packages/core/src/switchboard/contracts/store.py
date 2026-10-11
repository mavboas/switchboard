"""Onde ficam execuções, contratos, eventos e spans.

:class:`ContractStore` é o contrato de persistência usado pelo
:class:`~switchboard.contracts.manager.ContractManager`. Há duas
implementações: em memória (modo framework/YAML e testes) e SQL
(:class:`switchboard.storage.contracts.SqlContractStore`, usada pelo router
com banco).

Concorrência: ``save_contract`` usa *optimistic locking* (``version``) e
``update_run`` aceita a lista de status esperados — é assim que dois eventos
simultâneos (push e polling, ou dois processos do router) não se atropelam e
que a consolidação acontece uma única vez.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol

from ..tracing import Span, utcnow
from . import states
from .models import ContractEvent, ContractRecord, RunRecord


class ContractStore(Protocol):
    async def create_run(self, run: RunRecord) -> None: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def update_run(
        self, run_id: str, *, expect: Iterable[str] | None = None, **fields: Any
    ) -> bool: ...

    async def add_contract(self, contract: ContractRecord, events: list[ContractEvent]) -> None: ...

    async def get_contract(self, contract_id: str) -> ContractRecord | None: ...

    async def save_contract(
        self, contract: ContractRecord, events: list[ContractEvent]
    ) -> bool: ...

    async def run_contracts(self, run_id: str) -> list[ContractRecord]: ...

    async def contract_events(self, contract_id: str) -> list[ContractEvent]: ...

    async def due_contracts(self, now: datetime, limit: int = 50) -> list[ContractRecord]: ...

    async def claim(self, contract_id: str, now: datetime, until: datetime) -> bool: ...

    async def release(self, contract_id: str) -> None: ...

    async def open_runs(self) -> list[RunRecord]: ...

    async def add_spans(self, spans: list[Span]) -> None: ...


OPEN_RUN_STATUSES = frozenset({"pending", "needs_input", "consolidating"})


class MemoryContractStore:
    """Implementação em memória (um processo).

    Guarda no máximo ``max_runs`` execuções: acima disso, as encerradas mais
    antigas saem junto com contratos, eventos e spans.
    """

    def __init__(self, *, max_runs: int = 1000) -> None:
        self.max_runs = max_runs
        self._runs: dict[str, RunRecord] = {}
        self._contracts: dict[str, ContractRecord] = {}
        self._events: dict[str, list[ContractEvent]] = {}
        self._leases: dict[str, datetime] = {}
        self._spans: dict[str, list[Span]] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    # -- execuções --------------------------------------------------------------

    async def create_run(self, run: RunRecord) -> None:
        async with self._lock:
            self._runs[run.id] = copy.deepcopy(run)
            self._prune()

    def _prune(self) -> None:
        excess = len(self._runs) - self.max_runs
        if excess <= 0:
            return
        finished = sorted(
            (r for r in self._runs.values() if r.status in ("completed", "failed")),
            key=lambda r: r.created_at,
        )
        for run in finished[:excess]:
            self._runs.pop(run.id, None)
            self._spans.pop(run.id, None)
            for cid in [c.id for c in self._contracts.values() if c.run_id == run.id]:
                self._contracts.pop(cid, None)
                self._events.pop(cid, None)
                self._leases.pop(cid, None)

    async def get_run(self, run_id: str) -> RunRecord | None:
        run = self._runs.get(run_id)
        return copy.deepcopy(run) if run else None

    async def update_run(
        self, run_id: str, *, expect: Iterable[str] | None = None, **fields: Any
    ) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            if expect is not None and run.status not in set(expect):
                return False
            for key, value in fields.items():
                setattr(run, key, value)
            run.updated_at = utcnow()
            return True

    # -- contratos --------------------------------------------------------------

    def _add_events(self, events: list[ContractEvent]) -> None:
        for event in events:
            self._seq += 1
            stored = copy.deepcopy(event)
            stored.id = self._seq
            self._events.setdefault(event.contract_id, []).append(stored)

    async def add_contract(self, contract: ContractRecord, events: list[ContractEvent]) -> None:
        async with self._lock:
            if contract.id in self._contracts:
                raise ValueError(f"contrato {contract.id} já existe")
            self._contracts[contract.id] = copy.deepcopy(contract)
            self._add_events(events)

    async def get_contract(self, contract_id: str) -> ContractRecord | None:
        contract = self._contracts.get(contract_id)
        return copy.deepcopy(contract) if contract else None

    async def save_contract(self, contract: ContractRecord, events: list[ContractEvent]) -> bool:
        async with self._lock:
            current = self._contracts.get(contract.id)
            if current is None or current.version != contract.version:
                return False
            contract.version += 1
            contract.updated_at = utcnow()
            self._contracts[contract.id] = copy.deepcopy(contract)
            self._add_events(events)
            return True

    async def run_contracts(self, run_id: str) -> list[ContractRecord]:
        return sorted(
            (copy.deepcopy(c) for c in self._contracts.values() if c.run_id == run_id),
            key=lambda c: c.created_at,
        )

    async def contract_events(self, contract_id: str) -> list[ContractEvent]:
        return copy.deepcopy(self._events.get(contract_id, []))

    async def due_contracts(self, now: datetime, limit: int = 50) -> list[ContractRecord]:
        due = []
        for contract in self._contracts.values():
            if contract.state not in states.OPEN:
                continue
            lease = self._leases.get(contract.id)
            if lease is not None and lease > now:
                continue
            check = contract.next_check_at is not None and contract.next_check_at <= now
            late = contract.deadline_at is not None and contract.deadline_at <= now
            if check or late:
                due.append(copy.deepcopy(contract))
        due.sort(key=lambda c: c.next_check_at or c.created_at)
        return due[:limit]

    async def claim(self, contract_id: str, now: datetime, until: datetime) -> bool:
        async with self._lock:
            lease = self._leases.get(contract_id)
            if lease is not None and lease > now:
                return False
            self._leases[contract_id] = until
            return True

    async def release(self, contract_id: str) -> None:
        async with self._lock:
            self._leases.pop(contract_id, None)

    async def open_runs(self) -> list[RunRecord]:
        return [copy.deepcopy(r) for r in self._runs.values() if r.status in OPEN_RUN_STATUSES]

    # -- spans ------------------------------------------------------------------

    async def add_spans(self, spans: list[Span]) -> None:
        async with self._lock:
            for span in spans:
                self._spans.setdefault(span.trace_id, []).append(copy.deepcopy(span))

    async def run_spans(self, run_id: str) -> list[Span]:
        return copy.deepcopy(self._spans.get(run_id, []))

    async def list_contracts(self, *, limit: int = 100) -> list[ContractRecord]:
        return sorted(
            (copy.deepcopy(c) for c in self._contracts.values()),
            key=lambda c: c.created_at,
            reverse=True,
        )[:limit]
