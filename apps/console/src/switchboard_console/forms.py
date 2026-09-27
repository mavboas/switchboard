"""Conversão dos campos de formulário HTML para os dicionários do repositório."""

from __future__ import annotations

from typing import Any

from starlette.datastructures import FormData


def text(form: FormData, name: str) -> str:
    value = form.get(name)
    return value.strip() if isinstance(value, str) else ""


def optional_text(form: FormData, name: str) -> str | None:
    return text(form, name) or None


def checkbox(form: FormData, name: str) -> bool:
    return form.get(name) in ("on", "true", "1", "sim")


def number(form: FormData, name: str, cast=float) -> Any:
    raw = text(form, name).replace(",", ".")
    if not raw:
        return None
    try:
        return cast(float(raw)) if cast is int else cast(raw)
    except ValueError:
        return raw  # deixa a validação da spec apontar o erro


def id_list(form: FormData, name: str) -> list[int]:
    return [int(v) for v in form.getlist(name) if isinstance(v, str) and v.isdigit()]


def headers(form: FormData, name: str) -> dict[str, str]:
    """Linhas ``Nome: valor`` viram um dicionário de cabeçalhos."""
    out: dict[str, str] = {}
    for line in text(form, name).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip():
                out[key.strip()] = value.strip()
    return out


def headers_text(values: dict[str, str] | None) -> str:
    return "\n".join(f"{k}: {v}" for k, v in (values or {}).items())


def model_data(form: FormData) -> dict[str, Any]:
    return {
        "name": text(form, "name"),
        "preset": optional_text(form, "preset"),
        "provider": text(form, "provider") or "openai",
        "model": text(form, "model"),
        "base_url": optional_text(form, "base_url"),
        "api_key": optional_text(form, "api_key"),
        "clear_api_key": checkbox(form, "clear_api_key"),
        "api_key_header": text(form, "api_key_header") or "Authorization",
        "extra_headers": headers(form, "extra_headers"),
        "temperature": number(form, "temperature"),
        "max_tokens": number(form, "max_tokens", int),
        "timeout_s": number(form, "timeout_s") or 60.0,
        "json_mode": checkbox(form, "json_mode"),
    }


def agent_data(form: FormData) -> dict[str, Any]:
    return {
        "name": text(form, "name"),
        "description": text(form, "description"),
        "url": text(form, "url"),
        "transport": text(form, "transport") or "streamable-http",
        "auth_token": optional_text(form, "auth_token"),
        "clear_auth_token": checkbox(form, "clear_auth_token"),
        "allowed_tools": text(form, "allowed_tools"),
        "enabled": checkbox(form, "enabled"),
        "timeout_s": number(form, "timeout_s") or 30.0,
    }


def knowledge_data(form: FormData) -> dict[str, Any]:
    return {
        "name": text(form, "name"),
        "description": text(form, "description"),
        "embedder_kind": text(form, "embedder_kind") or "hashing",
        "embedder_dim": number(form, "embedder_dim", int) or 512,
        "embedding_model_id": number(form, "embedding_model_id", int),
        "embedding_model": optional_text(form, "embedding_model"),
        "chunk_size": number(form, "chunk_size", int) or 800,
        "chunk_overlap": number(form, "chunk_overlap", int),
    }


def profile_data(form: FormData) -> dict[str, Any]:
    return {
        "name": text(form, "name"),
        "description": text(form, "description"),
        "model_id": number(form, "model_id", int),
        "system_prompt": text(form, "system_prompt"),
        "agent_ids": id_list(form, "agent_ids"),
        "kb_ids": id_list(form, "kb_ids"),
        "top_k": number(form, "top_k", int) or 4,
        "min_score": number(form, "min_score"),
        "synthesize": checkbox(form, "synthesize"),
        "allow_clarify": checkbox(form, "allow_clarify"),
        "enabled": checkbox(form, "enabled"),
    }
