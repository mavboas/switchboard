"""Atalhos de configuração para provedores comuns.

Um preset só preenche valores padrão (tipo de provedor, URL base, cabeçalho da
chave). Os nomes de modelo são sugestões de exemplo; confira o catálogo atual
do seu provedor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    provider: str
    base_url: str
    api_key_header: str = "Authorization"
    model_hint: str = ""
    key_hint: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PRESETS: dict[str, Preset] = {
    p.key: p
    for p in [
        Preset(
            "openai",
            "OpenAI",
            "openai",
            "https://api.openai.com/v1",
            model_hint="gpt-5-mini",
            key_hint="env:OPENAI_API_KEY",
        ),
        Preset(
            "anthropic",
            "Anthropic (Claude)",
            "anthropic",
            "https://api.anthropic.com",
            model_hint="claude-sonnet-4-5",
            key_hint="env:ANTHROPIC_API_KEY",
        ),
        Preset(
            "azure-openai",
            "Azure OpenAI (API v1)",
            "openai",
            "https://SEU-RECURSO.openai.azure.com/openai/v1",
            api_key_header="api-key",
            model_hint="nome-do-deployment",
            key_hint="env:AZURE_OPENAI_API_KEY",
            notes="Troque SEU-RECURSO pelo nome do recurso; o campo modelo é o nome do deployment.",
        ),
        Preset(
            "gemini",
            "Google Gemini (compatível OpenAI)",
            "openai",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_hint="gemini-2.5-flash",
            key_hint="env:GEMINI_API_KEY",
        ),
        Preset(
            "ollama",
            "Ollama (local)",
            "openai",
            "http://localhost:11434/v1",
            model_hint="qwen3:8b",
            notes="Sem chave. Dentro do Docker use http://host.docker.internal:11434/v1.",
        ),
        Preset(
            "openrouter",
            "OpenRouter",
            "openai",
            "https://openrouter.ai/api/v1",
            model_hint="openai/gpt-5-mini",
            key_hint="env:OPENROUTER_API_KEY",
        ),
        Preset(
            "groq",
            "Groq",
            "openai",
            "https://api.groq.com/openai/v1",
            model_hint="llama-3.3-70b-versatile",
            key_hint="env:GROQ_API_KEY",
        ),
        Preset(
            "compat",
            "Outro compatível com OpenAI (vLLM, LM Studio…)",
            "openai",
            "http://localhost:8000/v1",
            model_hint="nome-do-modelo",
        ),
        Preset(
            "offline",
            "Offline (heurístico, sem LLM)",
            "offline",
            "",
            notes="Decide por palavras-chave e responde de forma extrativa. Bom para demo e testes.",
        ),
    ]
}
