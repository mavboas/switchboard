"""Termos do contrato fechado entre o Switchboard e um agente A2A.

A extensão A2A ``urn:switchboard:a2a:contract:v1`` tem duas metades:

* **no Agent Card** o agente declara, por skill, os schemas de entrada e
  saída e (opcionalmente) a duração máxima::

      "capabilities": {"extensions": [{
          "uri": "urn:switchboard:a2a:contract:v1",
          "params": {"skills": {"analisar_proposta": {
              "input_schema": {...}, "output_schema": {...}, "max_duration_s": 900}}}}]}

* **em cada mensagem** o roteador envia os termos do contrato em
  ``message.metadata[URI]``: id, skill, hash dos dois schemas, prazo e quem
  chama. O agente confere os hashes contra os schemas que ele mesmo publica:
  se divergirem (o agente mudou o schema depois da descoberta), rejeita.

O hash é o SHA-256 da forma canônica do schema (chaves ordenadas, sem
espaços, números inteiros sem ``.0``) — o protobuf ``Struct`` do SDK
transforma ``480`` em ``480.0``, e os dois lados precisam chegar ao mesmo
valor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

CONTRACT_EXTENSION_URI = "urn:switchboard:a2a:contract:v1"
CONTRACT_VERSION = 1


def normalize_numbers(value: Any) -> Any:
    """Converte floats inteiros (``24.0``) em int, recursivamente."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): normalize_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_numbers(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_numbers(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def schema_hash(schema: Mapping[str, Any] | None) -> str | None:
    if schema is None:
        return None
    return "sha256:" + hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()


def schema_problems(schema: Any) -> list[str]:
    """Problemas no próprio schema (não é objeto ou não é um JSON Schema válido)."""
    if not isinstance(schema, Mapping):
        return ["o schema precisa ser um objeto JSON"]
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError as exc:
        return [f"schema inválido: {exc.message}"]
    return []


def validate(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Erros de validação de ``instance`` contra ``schema`` (lista vazia = válido)."""
    validator = Draft202012Validator(dict(schema))
    errors = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        where = ".".join(str(p) for p in error.absolute_path)
        errors.append(f"{where}: {error.message}" if where else error.message)
    return errors[:20]


def _types(schema: Mapping[str, Any]) -> set[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind}
    if isinstance(kind, list):
        return {str(k) for k in kind}
    return set()


def coerce_to_schema(value: Any, schema: Mapping[str, Any] | None) -> Any:
    """Ajustes seguros antes de validar: ``24.0`` vira ``24`` onde o schema pede inteiro.

    Dados que passam pelo protobuf ``Struct`` chegam com todo número como
    float; aqui eles voltam a ser inteiros quando o schema diz ``integer``.
    """
    if not isinstance(schema, Mapping):
        return value
    types = _types(schema)
    if isinstance(value, float) and value.is_integer() and "integer" in types:
        return int(value)
    if isinstance(value, Mapping):
        props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        extra = schema.get("additionalProperties")
        out = {}
        for key, item in value.items():
            sub = (
                props.get(key) if key in props else (extra if isinstance(extra, Mapping) else None)
            )
            out[key] = coerce_to_schema(item, sub)
        return out
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return [coerce_to_schema(item, schema["items"]) for item in value]
    return value


@dataclass(frozen=True)
class SkillTerms:
    """O que o agente declara para uma skill no Agent Card."""

    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    max_duration_s: float | None = None

    @property
    def input_hash(self) -> str | None:
        return schema_hash(self.input_schema)

    @property
    def output_hash(self) -> str | None:
        return schema_hash(self.output_schema)

    @property
    def complete(self) -> bool:
        return self.input_schema is not None and self.output_schema is not None

    def to_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.input_schema is not None:
            out["input_schema"] = self.input_schema
        if self.output_schema is not None:
            out["output_schema"] = self.output_schema
        if self.max_duration_s is not None:
            out["max_duration_s"] = self.max_duration_s
        return out

    @classmethod
    def parse(cls, params: Any) -> tuple[SkillTerms | None, list[str]]:
        if not isinstance(params, Mapping):
            return None, ["termos da skill precisam ser um objeto"]
        problems: list[str] = []
        schemas: dict[str, dict[str, Any] | None] = {}
        for key in ("input_schema", "output_schema"):
            raw = params.get(key)
            if raw is None:
                schemas[key] = None
                continue
            found = schema_problems(raw)
            if found:
                problems.extend(f"{key}: {p}" for p in found)
                schemas[key] = None
            else:
                schemas[key] = normalize_numbers(dict(raw))
        duration = params.get("max_duration_s")
        max_duration = (
            float(duration) if isinstance(duration, (int, float)) and duration > 0 else None
        )
        terms = cls(schemas["input_schema"], schemas["output_schema"], max_duration)
        if terms.input_schema is None and terms.output_schema is None and not problems:
            problems.append("sem input_schema nem output_schema")
        return (terms if not problems else None), problems


def skills_from_extension(params: Any) -> tuple[dict[str, SkillTerms], dict[str, list[str]]]:
    """Termos por skill declarados em ``params`` da extensão (e problemas por skill)."""
    found: dict[str, SkillTerms] = {}
    problems: dict[str, list[str]] = {}
    skills = params.get("skills") if isinstance(params, Mapping) else None
    if not isinstance(skills, Mapping):
        return found, problems
    for skill_id, raw in skills.items():
        terms, issues = SkillTerms.parse(raw)
        if terms is not None:
            found[str(skill_id)] = terms
        if issues:
            problems[str(skill_id)] = issues
    return found, problems
