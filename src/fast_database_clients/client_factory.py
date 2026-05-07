#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2026-04-09
# ---------------------------------------------------------------------------
"""
Factory for creating database clients from application config.

Reads ``[databases.<name>]`` sections from the application config and
instantiates the correct client class based on ``type``.
"""
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections import deque
from typing import Dict

from fast_database_clients.fast_database_client import DatabaseClientBase
from fast_database_clients.fast_influxdb_client import FastInfluxDBClient
from fast_database_clients.fast_influxdb3_client import FastInfluxDB3Client

logger = logging.getLogger(__name__)

# Maps config type strings to client classes
_CLIENT_TYPES = {
    "influxdb_v2": FastInfluxDBClient,
    "influxdb_v3": FastInfluxDB3Client,
}

MAX_BUFFER_LENGTH = 1_000_000


def create_database_client(
    name: str,
    config: dict,
    buffer_maxlen: int = MAX_BUFFER_LENGTH,
) -> DatabaseClientBase:
    """
    Create a single database client from a config dict.

    The config dict must contain a ``type`` key (``influxdb_v2`` or ``influxdb_v3``).
    Remaining keys are passed to the client's ``from_params()`` factory.
    """
    config = dict(config)  # shallow copy so we don't mutate the original
    db_type = config.pop("type", None)
    if db_type is None:
        raise ValueError(f"Database config '{name}' is missing required 'type' field")

    cls = _CLIENT_TYPES.get(db_type)
    if cls is None:
        raise ValueError(
            f"Unknown database type '{db_type}' for '{name}'. "
            f"Supported types: {list(_CLIENT_TYPES.keys())}"
        )

    # Each client gets its own buffer (the router pushes metrics into it)
    buffer = deque(maxlen=buffer_maxlen)
    config["buffer"] = buffer
    config["name"] = name

    logger.info(
        "Creating database client '%s' (type=%s)",
        name,
        db_type,
        extra={"database_name": name, "database_type": db_type, "event": "client_create"},
    )

    return cls.from_params(**config)


def create_database_clients(
    databases_config: dict,
    buffer_maxlen: int = MAX_BUFFER_LENGTH,
) -> Dict[str, DatabaseClientBase]:
    """
    Create all database clients from the ``[databases]`` section of the
    application config.

    Returns a ``{name: client}`` mapping.
    """
    if not databases_config:
        raise ValueError("No databases configured in [databases] section")

    clients: Dict[str, DatabaseClientBase] = {}

    for name, config in databases_config.items():
        clients[name] = create_database_client(name, config, buffer_maxlen)

    logger.info(
        "Created %d database clients: %s",
        len(clients),
        list(clients.keys()),
    )

    return clients
