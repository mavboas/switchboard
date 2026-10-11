"""Serviço assíncrono dos plugins: instalar um pacote e manter as bases dele indexadas.

Indexar documentos chama o embedder (que pode ser um serviço remoto), então esta parte é
assíncrona e fica separada das operações de banco de :mod:`switchboard.storage.plugins`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import anyio.to_thread
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import qualify
from ..errors import PluginConflict, SwitchboardError
from ..plugins import PluginBundle, load_plugin
from ..rag.loaders import guess_title
from ..secrets import SecretBox
from .db import Database
from .knowledge import KnowledgeService, content_hash
from .orm import Document, KnowledgeBase
from .plugins import InstallResult, install_plugin, plugin_owned, require_plugin


class PluginService:
    """Instala plugins e mantém as bases de conhecimento deles indexadas."""

    def __init__(self, db: Database, knowledge: KnowledgeService, box: SecretBox):
        self.db = db
        self.knowledge = knowledge
        self.box = box

    async def install(
        self,
        source: Any,
        *,
        config: Mapping[str, Any] | None = None,
        clear: Iterable[str] = (),
        actor: str = "console",
        allow_downgrade: bool = False,
    ) -> InstallResult:
        """Lê o pacote, instala (ou atualiza) e indexa os documentos."""
        bundle = await anyio.to_thread.run_sync(load_plugin, source)
        return await self.install_bundle(
            bundle, config=config, clear=clear, actor=actor, allow_downgrade=allow_downgrade
        )

    async def install_bundle(
        self,
        bundle: PluginBundle,
        *,
        config: Mapping[str, Any] | None = None,
        clear: Iterable[str] = (),
        actor: str = "console",
        allow_downgrade: bool = False,
    ) -> InstallResult:
        def run() -> InstallResult:
            try:
                with self.db.session() as session:
                    return install_plugin(
                        session,
                        bundle,
                        box=self.box,
                        config=config,
                        clear=clear,
                        actor=actor,
                        allow_downgrade=allow_downgrade,
                    )
            except IntegrityError as exc:  # duas primeiras instalações do mesmo plugin juntas
                raise PluginConflict(
                    f"o plugin '{bundle.name}' acabou de ser instalado por outro envio: confira a "
                    "versão instalada e envie de novo se preciso"
                ) from exc

        result = await anyio.to_thread.run_sync(run)
        try:
            await self.ingest(result.plugin.name, bundle)
        except SwitchboardError as exc:
            # o plugin já está instalado; só os documentos ficaram para trás (embedder fora do ar,
            # por exemplo). Quem instalou fica sabendo e repete a sincronização depois.
            result.notes.append(
                f"o plugin foi instalado, mas indexar os documentos falhou ({exc}); "
                "corrija o embedder e sincronize os documentos do plugin"
            )
        return result

    async def ingest(
        self, name: str, bundle: PluginBundle | None = None
    ) -> dict[str, dict[str, int]]:
        """Deixa as bases do plugin com exatamente os documentos do pacote (por conteúdo)."""

        def plan() -> dict[int, tuple[str, dict[str, int], list[tuple[str, str, str]]]]:
            with self.db.session() as session:
                row = require_plugin(session, name)
                source = bundle or load_plugin(row.package, switchboard_version="")
                kbs = {k.name: k.id for k in plugin_owned(session, KnowledgeBase, row.id)}
                out: dict[int, tuple[str, dict[str, int], list[tuple[str, str, str]]]] = {}
                for local, docs in source.documents.items():
                    qualified = qualify(row.name, local)
                    kb_id = kbs.get(qualified)
                    if kb_id is None:
                        continue
                    existing = {
                        digest: doc_id
                        for doc_id, digest in session.execute(
                            select(Document.id, Document.content_hash).where(
                                Document.kb_id == kb_id
                            )
                        )
                    }
                    wanted = [
                        (path, guess_title(path, text), text) for path, text in sorted(docs.items())
                    ]
                    out[kb_id] = (qualified, existing, wanted)
                return out

        summary: dict[str, dict[str, int]] = {}
        for kb_id, (qualified, existing, wanted) in (await anyio.to_thread.run_sync(plan)).items():
            keep = {content_hash(text) for _, _, text in wanted}
            removed = 0
            for digest, doc_id in existing.items():
                if digest not in keep:
                    await self.knowledge.delete_document(kb_id, doc_id)
                    removed += 1
            added = 0
            for path, title, text in wanted:
                _, created = await self.knowledge.add_document(
                    kb_id, title=title, content=text, source=f"plugin:{path}"
                )
                added += int(created)
            summary[qualified] = {"added": added, "removed": removed, "documents": len(wanted)}
        return summary
