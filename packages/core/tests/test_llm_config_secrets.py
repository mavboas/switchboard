from __future__ import annotations

import json

import httpx
import pytest

from switchboard.config import ModelSpec, load_yaml
from switchboard.embeddings import OpenAICompatibleEmbedder, build_embedder
from switchboard.errors import ConfigError, LLMError, SecretError
from switchboard.llm import (
    AnthropicChat,
    Message,
    OfflineChat,
    OpenAICompatibleChat,
    build_chat_model,
)
from switchboard.llm.anthropic import to_anthropic_messages
from switchboard.secrets import SecretBox, resolve_env


def _openai_reply(content: str, model: str = "m") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


async def test_openai_compat_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_reply('{"ok": true}'))

    chat = OpenAICompatibleChat(
        model="gpt-x",
        base_url="https://api.example/v1/",
        api_key="sk-123",
        transport=httpx.MockTransport(handler),
    )
    result = await chat.chat([Message("system", "s"), Message("user", "u")], json_mode=True)
    assert seen["url"] == "https://api.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-123"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "u"}
    assert (result.text, result.usage.input_tokens, result.usage.output_tokens) == (
        '{"ok": true}',
        12,
        3,
    )
    await chat.aclose()


async def test_openai_compat_adapts_rejected_parameters():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."
                    }
                },
            )
        if "temperature" in body:
            return httpx.Response(
                400, json={"error": {"message": "temperature does not support 0.2"}}
            )
        return httpx.Response(200, json=_openai_reply("oi"))

    chat = OpenAICompatibleChat(model="o", transport=httpx.MockTransport(handler))
    assert (await chat.chat([Message("user", "u")])).text == "oi"
    assert "max_completion_tokens" in bodies[-1] and "temperature" not in bodies[-1]
    # o ajuste é lembrado: a próxima chamada já sai certa
    await chat.chat([Message("user", "u")])
    assert len(bodies) == 4


async def test_openai_compat_errors_become_llm_error():
    chat = OpenAICompatibleChat(
        model="m",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})
        ),
    )
    with pytest.raises(LLMError, match="HTTP 401 - bad key"):
        await chat.chat([Message("user", "u")])

    def boom(request):
        raise httpx.ConnectError("recusada")

    chat = OpenAICompatibleChat(model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError, match="falha de rede"):
        await chat.chat([Message("user", "u")])


async def test_azure_style_api_key_header():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json=_openai_reply("ok"))

    spec = ModelSpec(
        name="azure",
        model="dep",
        api_key="k",
        api_key_header="api-key",
        base_url="https://r/openai/v1",
    )
    chat = build_chat_model(spec, transport=httpx.MockTransport(handler))
    await chat.chat([Message("user", "u")])
    assert seen["api-key"] == "k" and "authorization" not in seen


def test_to_anthropic_messages_alternation():
    system, convo = to_anthropic_messages(
        [
            Message("system", "a"),
            Message("assistant", "oi"),
            Message("user", "x"),
            Message("user", "y"),
        ]
    )
    assert system == "a"
    assert [m["role"] for m in convo] == ["user", "assistant", "user"]
    assert convo[-1]["content"] == "x\n\ny"


async def test_anthropic_request_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-x",
                "content": [{"type": "text", "text": "olá"}],
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        )

    chat = AnthropicChat(
        model="claude-x",
        base_url="https://api.anthropic.com/v1",
        api_key="ak",
        transport=httpx.MockTransport(handler),
    )
    result = await chat.chat([Message("system", "regras"), Message("user", "oi")], json_mode=True)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "ak"
    assert "JSON" in seen["body"]["system"] and seen["body"]["system"].startswith("regras")
    assert (result.text, result.usage.input_tokens) == ("olá", 7)


async def test_offline_chat_and_factory():
    model = build_chat_model(ModelSpec(name="off", provider="offline"))
    assert isinstance(model, OfflineChat) and model.offline
    assert "offline" in (await model.chat([Message("user", "x")])).text.lower()


async def test_openai_embeddings_batching():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    emb = OpenAICompatibleEmbedder(model="e", batch_size=2, transport=httpx.MockTransport(handler))
    vectors = await emb.embed(["a", "b", "c"])
    assert calls == [2, 1]
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]


def test_build_embedder_requires_openai_connection():
    from switchboard.config import EmbedderSpec

    spec = EmbedderSpec(kind="model", connection="c", model="e")
    with pytest.raises(ConfigError):
        build_embedder(spec, ModelSpec(name="c", provider="anthropic", model="x"))


def test_secret_box_roundtrip_and_env(monkeypatch):
    box = SecretBox("chave-de-teste")
    sealed = box.seal("sk-secreta")
    assert sealed.startswith("enc:") and "sk-secreta" not in sealed
    assert box.open(sealed) == "sk-secreta"
    assert box.seal("env:MINHA_API_KEY") == "env:MINHA_API_KEY"
    monkeypatch.setenv("MINHA_API_KEY", "valor")
    assert box.open("env:MINHA_API_KEY") == "valor"
    assert SecretBox.describe(sealed) == "•••••• (cifrada)"
    assert box.seal("  ") is None
    with pytest.raises(SecretError):
        SecretBox(None).seal("sk-sem-chave")
    with pytest.raises(SecretError):
        SecretBox("outra-chave").open(sealed)
    assert resolve_env("env:NAO_EXISTE_X") is None


def test_env_references_are_restricted(monkeypatch):
    """Quem edita a configuração não pode ler variáveis arbitrárias do processo."""
    monkeypatch.setenv("SWITCHBOARD_SECRET_KEY", "mestra")
    monkeypatch.setenv("DATABASE_PASSWORD", "segredo")
    box = SecretBox("k")
    for name in (
        "SWITCHBOARD_SECRET_KEY",
        "SWITCHBOARD_DATABASE_URL",
        "DATABASE_PASSWORD",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "PATH",
        "X-Y",
    ):
        with pytest.raises(SecretError, match="não está liberada"):
            box.seal(f"env:{name}")
        with pytest.raises(SecretError):
            box.open(f"env:{name}")  # mesmo que alguém grave direto no banco
    assert box.seal("env:MCP_CRM_TOKEN") == "env:MCP_CRM_TOKEN"
    assert box.seal("env:A2A_RISCO_TOKEN") == "env:A2A_RISCO_TOKEN"  # token de agente A2A
    assert box.seal("env:OPENAI_API_KEY") == "env:OPENAI_API_KEY"
    custom = SecretBox("k", "MEU_SEGREDO,ACME_*")
    assert custom.seal("env:ACME_CHAVE") == "env:ACME_CHAVE"
    with pytest.raises(SecretError):
        custom.seal("env:OPENAI_API_KEY")  # a lista customizada substitui a padrão
    # modo YAML (arquivo do operador) continua sem restrição
    assert resolve_env("env:DATABASE_PASSWORD") == "segredo"


def test_master_key_file_is_created_once(tmp_path):
    from switchboard.secrets import load_master_key

    path = tmp_path / "sub" / "secret.key"
    first = load_master_key(None, str(path))
    assert first and len(first) >= 40 and path.read_text() == first
    assert load_master_key("", str(path)) == first  # segundo processo lê a mesma chave
    assert load_master_key("explicita", str(path)) == "explicita"
    assert load_master_key(None, None) is None
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_load_yaml_with_env_interpolation_and_reference_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SB_MODELO", "gpt-teste")
    good = tmp_path / "ok.yaml"
    good.write_text(
        """
models:
  - name: principal
    model: ${SB_MODELO}
    api_key: env:OPENAI_API_KEY
    base_url: ${SB_URL:-https://api.openai.com/v1}
profiles:
  - name: default
    model: principal
""",
        encoding="utf-8",
    )
    spec = load_yaml(good)
    assert spec.models[0].model == "gpt-teste"
    assert spec.models[0].base_url == "https://api.openai.com/v1"

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "models: [{name: m, provider: offline}]\nprofiles: [{name: p, model: m, agents: [fantasma]}]\n"
    )
    with pytest.raises(ConfigError, match="agente 'fantasma' não existe"):
        load_yaml(bad)

    invalid_name = tmp_path / "name.yaml"
    invalid_name.write_text(
        "models: [{name: 'Com Espaço', provider: offline}]\nprofiles: [{model: x}]\n"
    )
    with pytest.raises(ConfigError, match="letras minúsculas"):
        load_yaml(invalid_name)


def test_load_yaml_ignores_placeholders_in_comments(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "# ${NAO_DEFINIDA} só aparece em comentário\nmodels: [{name: m, provider: offline}]\nprofiles: [{model: m}]\n",
        encoding="utf-8",
    )
    assert load_yaml(config).profiles[0].model == "m"


async def test_openai_compat_empty_answer_at_token_limit_is_an_error():
    reply = {
        "model": "m",
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {},
    }
    chat = OpenAICompatibleChat(
        model="m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply))
    )
    with pytest.raises(LLMError, match="limite de tokens"):
        await chat.chat([Message("user", "u")])


async def test_closed_client_becomes_llm_error():
    chat = OpenAICompatibleChat(
        model="m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    )
    await chat.aclose()
    with pytest.raises(LLMError):
        await chat.chat([Message("user", "u")])


async def test_openai_compat_drops_max_tokens_on_context_errors():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            msg = "This model's maximum context length is 4096 tokens. However, you requested 4500 tokens."
            return httpx.Response(400, json={"error": {"message": msg}})
        return httpx.Response(200, json=_openai_reply("ok"))

    chat = OpenAICompatibleChat(model="m", transport=httpx.MockTransport(handler))
    assert (await chat.chat([Message("user", "u")])).text == "ok"
    assert "max_tokens" not in bodies[-1]
