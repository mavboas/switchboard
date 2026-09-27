"""Sobe o console: ``switchboard-console`` (lê SWITCHBOARD_* do ambiente ou do .env)."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .settings import Settings


def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level
    )


if __name__ == "__main__":  # pragma: no cover
    main()
