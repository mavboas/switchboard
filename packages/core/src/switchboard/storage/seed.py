"""Configuração de demonstração: roda sem nenhuma chave de API.

Cria (só se o banco estiver vazio: sem modelos, conectores, agentes, bases e
roteadores):

* o modelo ``offline`` (decisor heurístico);
* os conectores MCP de exemplo ``credito`` (simulador) e ``chamados``;
* os agentes A2A de exemplo ``analise-credito`` (tarefa longa, contrato
  completo) e ``risco`` (prevenção a fraude, contrato completo);
* a base ``manual-atendimento`` com os documentos de ``examples/knowledge``;
* o roteador ``default`` ligando tudo.

Depois é só trocar o modelo do roteador por um provedor real no console (e,
se quiser, ligar um modelo de decisão TypeSafe Jev).
"""

from __future__ import annotations

import logging
from pathlib import Path

import anyio.to_thread
from sqlalchemy import select

from ..rag.loaders import extract_text, guess_title, iter_files
from ..secrets import SecretBox
from .db import Database
from .knowledge import KnowledgeService
from .orm import Agent, Connector, KnowledgeBase, LlmModel, RouterProfile
from .repo import save_agent, save_connector, save_knowledge_base, save_model, save_profile

log = logging.getLogger(__name__)

DEMO_PROMPT = (
    "Você é o assistente virtual da Acme Serviços Financeiros (empresa fictícia de demonstração). "
    "Responda em português do Brasil, com cordialidade e objetividade. Não invente fatos."
)


CONNECTOR_DESCRIPTIONS = {
    "credito": "Simulações de financiamento e empréstimo (tabela Price e SAC).",
    "chamados": "Abertura e consulta de chamados de suporte.",
}

AGENT_DESCRIPTIONS = {
    "analise-credito": "Análise de propostas de crédito (política, capacidade de pagamento e limite).",
    "risco": "Prevenção a fraude: avalia o risco de uma operação ou cliente.",
}


async def seed_demo(
    db: Database,
    *,
    knowledge_dir: Path | None,
    connector_urls: dict[str, str],
    agent_urls: dict[str, str] | None = None,
    box: SecretBox | None = None,
) -> bool:
    """Popula o banco com a demo; devolve False se já havia configuração."""
    box = box or SecretBox(None)
    with db.session() as session:
        # só semeia um banco realmente vazio: se o usuário apagou o roteador da
        # demo mas manteve o resto, não recriamos nada (nem colidimos com nomes)
        for table in (RouterProfile, LlmModel, Connector, Agent, KnowledgeBase):
            if session.scalars(select(table.id).limit(1)).first() is not None:
                return False
        model = save_model(
            session, {"name": "offline", "provider": "offline", "preset": "offline"}, box=box
        )
        connector_ids = []
        for name, url in connector_urls.items():
            connector = save_connector(
                session,
                {"name": name, "url": url, "description": CONNECTOR_DESCRIPTIONS.get(name, "")},
                box=box,
            )
            connector_ids.append(connector.id)
        agent_ids = []
        for name, url in (agent_urls or {}).items():
            agent = save_agent(
                session,
                {"name": name, "url": url, "description": AGENT_DESCRIPTIONS.get(name, "")},
                box=box,
            )
            agent_ids.append(agent.id)
        kb = save_knowledge_base(
            session,
            {
                "name": "manual-atendimento",
                "description": "Manual de atendimento e políticas da Acme (fictícia).",
            },
        )
        save_profile(
            session,
            {
                "name": "default",
                "description": "Roteador de demonstração (modelo offline).",
                "model_id": model.id,
                "system_prompt": DEMO_PROMPT,
                "connector_ids": connector_ids,
                "agent_ids": agent_ids,
                "kb_ids": [kb.id],
                "top_k": 4,
                "min_score": 0.15,
                "wait_s": 8.0,
            },
        )
        kb_id = kb.id

    documents = await anyio.to_thread.run_sync(_read_documents, knowledge_dir)
    if documents:
        service = KnowledgeService(db)
        try:
            for title, source, content in documents:
                await service.add_document(kb_id, title=title, content=content, source=source)
        finally:
            await service.aclose()
    else:
        log.warning("diretório de conhecimento da demo não encontrado ou vazio: %s", knowledge_dir)
    return True


def _read_documents(knowledge_dir: Path | None) -> list[tuple[str, str, str]]:
    if knowledge_dir is None or not knowledge_dir.is_dir():
        return []
    documents = []
    for path in iter_files([str(knowledge_dir)]):
        text = extract_text(path.name, path.read_bytes())
        documents.append((guess_title(path.name, text), path.name, text))
    return documents


def demo_kb_id(db: Database) -> int | None:
    with db.session() as session:
        return session.scalar(
            select(KnowledgeBase.id).where(KnowledgeBase.name == "manual-atendimento")
        )
