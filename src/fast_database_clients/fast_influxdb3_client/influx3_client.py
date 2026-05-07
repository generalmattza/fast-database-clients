#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2026-04-09
# ---------------------------------------------------------------------------
"""
FastInfluxDB3Client -- InfluxDB v3 writer using influxdb3-python.

Subclasses DatabaseClientBase to provide the same buffer/write-thread pattern
as FastInfluxDBClient (v2), but targeting the InfluxDB v3 API.
"""
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Iterable, Union, Tuple
import logging

from influxdb_client_3 import InfluxDBClient3
from influxdb_client_3 import Point

from fast_database_clients.fast_database_client import DatabaseClientBase
from fast_database_clients.action_outcome import ActionOutcome, ActionOutcomeMessage

DEFAULT_WRITE_PRECISION = "ms"
WRITE_BATCH_SIZE = 5_000

logger = logging.getLogger(__name__)


def _metric_to_point(
    data: Union[dict, object],
    write_precision: str = DEFAULT_WRITE_PRECISION,
) -> Point:
    """Convert a metric dict (or dataclass) to an influxdb_client_3.Point."""
    if not isinstance(data, dict):
        if is_dataclass(data):
            data = asdict(data)
        else:
            raise ValueError("data must be a dict or a dataclass")

    # Strip transient metadata that should not reach the database
    data.pop("context", None)
    data.pop("write_precision", None)
    data.pop("expansion_pattern", None)

    measurement = data.get("measurement", "default")
    fields = data.get("fields", {})
    tags = data.get("tags", {})

    point = Point(measurement)
    for k, v in tags.items():
        point = point.tag(k, v)
    for k, v in fields.items():
        point = point.field(k, v)

    return point


def _chunks(iterable, size):
    """Yield successive size-length chunks from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class FastInfluxDB3Client(DatabaseClientBase):
    """
    InfluxDB v3 writer backed by influxdb3-python.

    Provides the same lifecycle as FastInfluxDBClient (v2):
      - from_params() factory
      - buffer + write thread via DatabaseClientBase
      - write() that converts metrics to Points and sends in batches
    """

    @classmethod
    def from_params(
        cls,
        url: str,
        token: str,
        database: str,
        buffer=None,
        write_interval: float = 1.0,
        default_write_precision: str = DEFAULT_WRITE_PRECISION,
        write_batch_size: int = WRITE_BATCH_SIZE,
        timeout: int = 10_000,
        name: str = "",
        query_port: int = None,
        **kwargs,
    ) -> "FastInfluxDB3Client":
        instance = cls(
            buffer=buffer,
            write_interval=write_interval,
            write_batch_size=write_batch_size,
            name=name,
        )
        # Pass the full URL as host -- InfluxDBClient3 parses scheme, hostname, port
        client_kwargs = dict(
            token=token,
            host=url,
            database=database,
        )
        if query_port is not None:
            client_kwargs["query_port_overwrite"] = query_port
        instance._client = InfluxDBClient3(**client_kwargs)
        instance.database = database
        instance.default_write_precision = default_write_precision
        instance.write_batch_size = write_batch_size
        instance._url = url
        return instance

    def write(
        self,
        metrics: Union[dict, Iterable[dict]],
        database: str = None,
        write_precision: str = None,
    ) -> None:
        database = database or self.database
        write_precision = write_precision or self.default_write_precision

        if isinstance(metrics, dict):
            metrics = [metrics]

        points = (
            _metric_to_point(m, write_precision=write_precision) for m in metrics
        )

        for batch in _chunks(points, self.write_batch_size):
            number_of_metrics = len(batch)
            measurements = set()
            for p in batch:
                if hasattr(p, '_name'):
                    measurements.add(p._name)

            log_action_outcome = ActionOutcomeMessage(
                action=f"Sending {number_of_metrics} metrics to influxdb3",
                action_verbose=f"Sending {number_of_metrics} metrics to influxdb3 database '{database}' at {self._url}",
            )
            outcome = ActionOutcome.SUCCESS
            try:
                self._client.write(record=batch, write_precision=write_precision)
            except Exception as e:
                outcome = ActionOutcome.FAILED
                logger.error(
                    "Failed to write metrics to InfluxDB v3",
                    extra={
                        "database": self.name,
                        "metrics_count": number_of_metrics,
                        "error": str(e),
                        "target_database": database,
                        "event": "influxdb3_write_failed",
                    },
                )
            finally:
                logger.info(
                    **log_action_outcome(
                        outcome=outcome,
                        database=self.name,
                        metrics_count=number_of_metrics,
                        measurements=sorted(measurements),
                        target_database=database,
                        event="influxdb3_write",
                    )
                )

    def ping(self) -> bool:
        """Health check -- attempt a trivial query."""
        try:
            # influxdb3-python has no dedicated ping; a lightweight query suffices
            self._client.query("SELECT 1")
            return True
        except Exception as e:
            logger.warning("InfluxDB v3 ping failed: %s", e)
            return False

    def query(self, query: str, **kwargs):
        return self._client.query(query=query, **kwargs)

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
        super().close()

    def __repr__(self):
        return f"FastInfluxDB3Client(url={self._url}, database={self.database})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
