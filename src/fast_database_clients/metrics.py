"""Prometheus metrics shared across the database client and router.

Defined in a single module so that the router and the per-client write loop
import the same Counter/Histogram/Gauge instances (the prometheus_client
default registry rejects duplicate names if the same metric is constructed
twice in different modules).
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Per-client write-path metrics (FastDatabaseClient.write_periodically)
# ---------------------------------------------------------------------------

database_writes_total = Counter(
    "database_writes_total",
    "Number of database write attempts, partitioned by outcome",
    labelnames=("database", "outcome"),  # outcome: success|failure
)

database_write_duration_seconds = Histogram(
    "database_write_duration_seconds",
    "Wall-clock duration of a single database write call",
    labelnames=("database",),
)

database_write_batch_metrics = Histogram(
    "database_write_batch_metrics",
    "Number of metrics included in a single database write batch",
    labelnames=("database",),
    # batch sizes can range from 1 to write_batch_size (default 5_000)
    buckets=(1, 10, 100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
)

database_buffer_occupancy = Gauge(
    "database_buffer_occupancy",
    "Current item count in a database client's outbound buffer",
    labelnames=("database",),
)

# ---------------------------------------------------------------------------
# Router metrics (DatabaseRouter._route_loop / _distribute)
# ---------------------------------------------------------------------------

database_router_metrics_routed_total = Counter(
    "database_router_metrics_routed_total",
    "Number of metrics fanned out by the router to a downstream database",
    labelnames=("database",),
)

database_router_metrics_dropped_total = Counter(
    "database_router_metrics_dropped_total",
    "Number of metrics the router could not deliver (e.g. unknown target)",
    labelnames=("database", "reason"),
)

database_router_batch_size = Histogram(
    "database_router_batch_size",
    "Number of metrics drained from the input buffer per router iteration",
    buckets=(1, 10, 100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
)

database_router_loop_iterations_total = Counter(
    "database_router_loop_iterations_total",
    "Number of times the router's drain-and-distribute loop has run",
)
