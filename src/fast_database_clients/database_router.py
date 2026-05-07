#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2026-04-09
# ---------------------------------------------------------------------------
"""
DatabaseRouter -- Fan-out router that reads metrics from a shared buffer
and distributes them to per-database-client buffers based on routing context.
"""
# ---------------------------------------------------------------------------

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fast_database_clients.fast_database_client import DatabaseClientBase

logger = logging.getLogger(__name__)

BATCH_SIZE = 5_000
POLL_INTERVAL = 0.1  # seconds


class DatabaseRouter:
    """
    Reads metrics from a shared input buffer and routes each metric to the
    appropriate database client buffers based on ``context.databases``.

    If a metric has no routing context (or an empty one), it is sent to all
    databases listed in *default_databases*.
    """

    def __init__(
        self,
        input_buffer,
        clients: Dict[str, DatabaseClientBase],
        default_databases: List[str],
        batch_size: int = BATCH_SIZE,
        poll_interval: float = POLL_INTERVAL,
    ):
        self.input_buffer = input_buffer
        self.clients = clients
        self.default_databases = default_databases
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._route_loop, daemon=True, name="DatabaseRouter"
        )
        self._thread.start()
        logger.info(
            "DatabaseRouter started",
            extra={
                "clients": list(self.clients.keys()),
                "default_databases": self.default_databases,
                "event": "router_started",
            },
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("DatabaseRouter stopped")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_loop(self) -> None:
        while not self._stop_event.is_set():
            buf_len = len(self.input_buffer)
            if buf_len > 0:
                read_count = min(self.batch_size, buf_len)

                # Snapshot + drain (same pattern as DatabaseClientBase)
                metrics = tuple(itertools.islice(iter(self.input_buffer), read_count))
                popped = 0
                while popped < read_count:
                    try:
                        self.input_buffer.popleft()
                        popped += 1
                    except IndexError:
                        break

                if metrics:
                    self._distribute(metrics)
            else:
                time.sleep(self.poll_interval)

    def _distribute(self, metrics: tuple) -> None:
        """Group metrics by target database and push to per-client buffers."""
        routed: Dict[str, list] = defaultdict(list)

        for metric in metrics:
            targets = self._resolve_targets(metric)
            for target in targets:
                routed[target].append(metric)

        for db_name, batch in routed.items():
            client = self.clients.get(db_name)
            if client is not None:
                client.buffer.extend(batch)
                logger.debug(
                    "Routed %d metrics to '%s'",
                    len(batch),
                    db_name,
                    extra={"database": db_name, "metrics_count": len(batch), "event": "router_distribute"},
                )
            else:
                logger.error(
                    "Unknown database target '%s' -- %d metrics dropped",
                    db_name,
                    len(batch),
                    extra={"database": db_name, "metrics_count": len(batch), "event": "router_unknown_target"},
                )

    def _resolve_targets(self, metric: Any) -> List[str]:
        """Determine which databases a metric should be written to."""
        if isinstance(metric, dict):
            context = metric.get("context")
            if context and isinstance(context, dict):
                databases = context.get("databases")
                if databases:
                    return databases
        return self.default_databases
