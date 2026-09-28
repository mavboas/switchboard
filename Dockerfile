# Imagem única para os serviços Python do monorepo. Escolha o serviço com
#   --build-arg PACKAGE=switchboard-console | switchboard-router | switchboard-example-agents
# e o comando no `docker run`/compose (switchboard-console, switchboard-router,
# switchboard-agent credito | chamados | analise-credito | risco). Só o pacote
# escolhido e suas dependências entram no venv final.

ARG PYTHON_IMAGE=python:3.12-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.17

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder
ARG PACKAGE=switchboard-console
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /src

# 1) só as dependências de terceiros (camada reaproveitada entre builds)
COPY pyproject.toml uv.lock .python-version ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY packages/agentkit/pyproject.toml packages/agentkit/README.md packages/agentkit/
COPY apps/router/pyproject.toml apps/router/
COPY apps/console/pyproject.toml apps/console/
COPY examples/agents/pyproject.toml examples/agents/
RUN uv sync --frozen --no-dev --no-install-workspace --package "${PACKAGE}"

# 2) o código do monorepo, instalado sem modo editável
COPY packages packages
COPY apps apps
COPY examples/agents examples/agents
RUN uv sync --frozen --no-dev --no-editable --package "${PACKAGE}"

FROM ${PYTHON_IMAGE}
ARG PACKAGE=switchboard-console
LABEL org.opencontainers.image.title="${PACKAGE}" \
      org.opencontainers.image.source="https://github.com/mavboas/switchboard" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV PATH="/app/.venv/bin:${PATH}" \
    SWITCHBOARD_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SWITCHBOARD_DEMO_KNOWLEDGE_DIR=/app/examples/knowledge
WORKDIR /app
RUN useradd --create-home --uid 10001 switchboard \
    && mkdir -p /app/data /var/lib/switchboard \
    && chown switchboard /app/data /var/lib/switchboard
COPY --from=builder /app/.venv /app/.venv
COPY examples/knowledge /app/examples/knowledge
USER switchboard
CMD ["switchboard-console"]
