"""Migração do banco do MVP e persistência SQL dos contratos."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    select,
    text,
)

from switchboard.config import AgentSpec
from switchboard.contracts import (
    ContractManager,
    OpenRequest,
    RunRecord,
    states,
)
from switchboard.storage import Database, SqlContractStore, repo
from switchboard.storage.orm import Connector
from switchboard.tracing import Span, new_trace_id, utcnow

# --------------------------------------------------------------------------
# esquema do MVP (v0.1): "agents" eram servidores MCP


def _legacy_metadata() -> MetaData:
    md = MetaData()
    Table(
        "llm_models",
        md,
        Column("id", Integer, primary_key=True),
        Column("name", String(63), unique=True),
        Column("preset", String(40)),
        Column("provider", String(20)),
        Column("model", String(200)),
        Column("base_url", String(500)),
        Column("api_key", Text),
        Column("api_key_header", String(100)),
        Column("extra_headers", JSON),
        Column("temperature", Float),
        Column("max_tokens", Integer),
        Column("timeout_s", Float),
        Column("json_mode", Boolean),
        Column("created_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True)),
    )
    Table(
        "agents",
        md,
        Column("id", Integer, primary_key=True),
        Column("name", String(63), unique=True),
        Column("description", Text),
        Column("url", String(500)),
        Column("transport", String(20)),
        Column("auth_token", Text),
        Column("allowed_tools", JSON),
        Column("enabled", Boolean),
        Column("timeout_s", Float),
        Column("created_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True)),
    )
    Table(
        "router_profiles",
        md,
        Column("id", Integer, primary_key=True),
        Column("name", String(63), unique=True),
        Column("description", Text),
        Column("model_id", ForeignKey("llm_models.id")),
        Column("system_prompt", Text),
        Column("top_k", Integer),
        Column("min_score", Float),
        Column("synthesize", Boolean),
        Column("allow_clarify", Boolean),
        Column("enabled", Boolean),
        Column("created_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True)),
    )
    Table(
        "profile_agents",
        md,
        Column(
            "profile_id", ForeignKey("router_profiles.id", ondelete="CASCADE"), primary_key=True
        ),
        Column("agent_id", ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
    )
    Table(
        "traces",
        md,
        Column("id", String(32), primary_key=True),
        Column("created_at", DateTime(timezone=True)),
        Column("profile", String(63)),
        Column("question", Text),
        Column("answer", Text),
        Column("route", String(20)),
        Column("model", String(250)),
        Column("agent", String(63)),
        Column("tool", String(200)),
        Column("arguments", JSON),
        Column("reason", Text),
        Column("sources", JSON),
        Column("steps", JSON),
        Column("warnings", JSON),
        Column("latency_ms", Float),
        Column("input_tokens", Integer),
        Column("output_tokens", Integer),
        Column("error", Text),
    )
    return md


def test_mvp_database_is_migrated(empty_db_url):
    legacy = Database(empty_db_url)  # só para normalizar a URL
    engine = create_engine(legacy.url)
    now = utcnow().isoformat()
    with engine.begin() as conn:
        _legacy_metadata().create_all(conn)
        conn.execute(
            text(
                "INSERT INTO llm_models (id, name, provider, model, api_key_header, extra_headers, timeout_s, json_mode) "
                "VALUES (1, 'offline', 'offline', '', 'Authorization', '{}', 60, true)"
            )
        )
        for i, name in ((1, "credito"), (2, "chamados")):
            conn.execute(
                text(
                    "INSERT INTO agents (id, name, description, url, transport, allowed_tools, enabled, timeout_s, created_at, updated_at) "
                    "VALUES (:i, :n, 'MCP legado', :u, 'streamable-http', '[\"x\"]', true, 30, :t, :t)"
                ),
                {"i": i, "n": name, "u": f"http://{name}/mcp", "t": now},
            )
        conn.execute(
            text(
                "INSERT INTO router_profiles (id, name, description, model_id, system_prompt, top_k, min_score, synthesize, allow_clarify, enabled) "
                "VALUES (1, 'default', '', 1, 'p', 4, 0.2, true, true, true)"
            )
        )
        conn.execute(
            text("INSERT INTO profile_agents (profile_id, agent_id) VALUES (1, 1), (1, 2)")
        )
        conn.execute(
            text(
                "INSERT INTO traces (id, created_at, profile, question, answer, route, model, reason, sources, steps, warnings, latency_ms, input_tokens, output_tokens) "
                "VALUES ('t1', :t, 'default', 'q', 'a', 'delegated', 'm', '', '[]', '[]', '[]', 1, 0, 0)"
            ),
            {"t": now},
        )
    engine.dispose()

    db = Database(empty_db_url)
    db.init()
    db.init()  # idempotente
    try:
        tables = set(inspect(db.engine).get_table_names())
        assert "agents" not in tables and "profile_agents" not in tables
        assert {"mcp_connectors", "a2a_agents", "contracts", "contract_events", "spans"} <= tables
        with db.session() as s:
            names = sorted(c.name for c in s.scalars(select(Connector)))
            assert names == ["chamados", "credito"]
            resolved = repo.resolve_profile(s, "default")
            assert sorted(c.name for c in resolved.connectors) == ["chamados", "credito"]
            assert resolved.connectors[0].allowed_tools == ["x"] and resolved.agents == []
            assert resolved.spec.wait_s == 8.0 and resolved.spec.decision_threshold == 0.6
            [trace] = repo.query_traces(s)
            assert repo.trace_to_dict(trace)["status"] == "completed"
            version = s.execute(
                text("SELECT value FROM switchboard_meta WHERE key = 'schema_version'")
            ).scalar()
            assert version == "2"
    finally:
        db.dispose()


# --------------------------------------------------------------------------
# SqlContractStore


async def test_sql_store_conditional_updates_and_leases(db):
    store = SqlContractStore(db)
    run_id = new_trace_id()
    await store.create_run(
        RunRecord(
            id=run_id, profile="p", question="q", answer="aguarde", extra={"route": "delegated"}
        )
    )
    assert await store.update_run(run_id, expect={"pending"}, status="consolidating")
    assert not await store.update_run(
        run_id, expect={"pending"}, status="consolidating"
    )  # só um consolida
    assert (await store.get_run(run_id)).status == "consolidating"
    assert [r.id for r in await store.open_runs()] == [run_id]

    from switchboard.contracts import ContractEvent, ContractRecord

    contract = ContractRecord(
        id="ctr_x",
        run_id=run_id,
        profile="p",
        agent="risco",
        skill="avaliar",
        kind="completo",
        deadline_at=utcnow() + timedelta(minutes=5),
        next_check_at=utcnow() - timedelta(seconds=1),
    )
    await store.add_contract(contract, [ContractEvent("ctr_x", "state", "router", states.PROPOSED)])
    loaded = await store.get_contract("ctr_x")
    assert loaded.deadline_at.tzinfo is not None  # SQLite devolve sem fuso: normalizado
    stale = await store.get_contract("ctr_x")
    loaded.state = states.ACTIVE
    assert await store.save_contract(
        loaded, [ContractEvent("ctr_x", "state", "poll", states.ACTIVE)]
    )
    stale.state = states.FAILED
    assert not await store.save_contract(stale, [])  # versão antiga perde
    assert (await store.get_contract("ctr_x")).state == states.ACTIVE
    events = await store.contract_events("ctr_x")
    assert [e.state for e in events] == [states.PROPOSED, states.ACTIVE]

    now = utcnow()
    assert [c.id for c in await store.due_contracts(now)] == ["ctr_x"]
    assert await store.claim("ctr_x", now, now + timedelta(seconds=30))
    assert not await store.claim("ctr_x", now, now + timedelta(seconds=30))
    assert await store.due_contracts(now) == []
    await store.release("ctr_x")
    assert [c.id for c in await store.due_contracts(now)] == ["ctr_x"]

    span = Span(
        "s" * 16,
        run_id,
        None,
        "consolidacao",
        "consolidacao",
        utcnow(),
        utcnow(),
        "ok",
        {"llm": False},
    )
    await store.add_spans([span])
    assert [s.name for s in await store.run_spans(run_id)] == ["consolidacao"]


async def test_manager_with_sql_store_end_to_end(db, network, risk_agent):
    store = SqlContractStore(db)
    directory = network.directory()
    spec = AgentSpec(name="risco", url=risk_agent.url)
    manager = ContractManager(
        store, client=directory.client, poll_min_s=0.01, agent_resolver=lambda n: _spec(spec)
    )
    info = await directory.describe(spec)
    run_id = new_trace_id()
    await manager.begin_run(
        RunRecord(id=run_id, profile="p", question="q", extra={"route": "delegated"})
    )
    contract = await manager.open(
        OpenRequest(
            run_id,
            "p",
            spec,
            info,
            info.skill("avaliar_risco"),
            {"cliente": "Ana", "valor": 10.0},
            "avalie",
            60,
        )
    )
    assert contract.state == states.ACTIVE and contract.reply_mode == "poll"
    risk_agent.complete(contract.remote_task_id, {"risco": "baixo", "score": 810.0}, "Risco baixo.")
    await manager.check(await store.get_contract(contract.id))
    run = await manager.wait_run(run_id, 2)
    assert run.status == "completed" and "Risco baixo." in run.answer
    with db.session() as s:
        details = repo.run_details(s, run_id)
    [c] = details["contracts"]
    assert c["state"] == states.COMPLETED and c["output"] == {"risco": "baixo", "score": 810}
    assert [e["state"] for e in c["events"] if e["kind"] == "state"] == [
        states.PROPOSED,
        states.ACTIVE,
        states.COMPLETED,
    ]
    assert any(sp["kind"] == "contrato" and sp["status"] == "ok" for sp in details["spans"])
    await directory.aclose()


async def _spec(spec):
    return spec
