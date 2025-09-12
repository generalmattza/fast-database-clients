import threading
import queue
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import WriteOptions, WriteType
from influxdb_client.client.write.point import Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.query_api import QueryApi

from fast_database_clients.fast_database_client import DatabaseClientBase

logger = logging.getLogger(__name__)


@dataclass
class InfluxDBWriterConfig:
    url: str
    token: str
    org: str
    bucket: str
    write_precision: str = "ms"
    batch_size: int = 5000  # Reverted to 5000 based on best performance
    max_retries: int = 3
    retry_delay: float = 1.0


class ThreadSafeInfluxDBWriter:
    """Thread-safe InfluxDB writer with connection pooling."""

    def __init__(self, config: InfluxDBWriterConfig):
        self.config = config
        self._local = threading.local()
        self.stats = {"total_writes": 0, "failed_writes": 0, "total_points": 0}
        self._stats_lock = threading.Lock()

    def _get_client(self):
        """Get thread-local InfluxDB client."""
        if not hasattr(self._local, "client"):
            self._local.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                enable_gzip=True,  # Enable compression
            )
            self._local.write_api = self._local.client.write_api(
                write_options=WriteOptions(
                    write_type=WriteType.synchronous
                )  # Reverted to synchronous
            )
        return self._local.client, self._local.write_api

    def write_batch(self, batch: List[Any], thread_id: Optional[str] = None) -> bool:
        """Write a batch of records to InfluxDB with retry logic."""
        if not batch:
            return True

        client, write_api = self._get_client()
        thread_id = thread_id or threading.current_thread().name

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"Thread {thread_id}: Writing batch of {len(batch)} records (attempt {attempt + 1})"
                )

                write_api.write(
                    bucket=self.config.bucket,
                    record=batch,
                    write_precision=self.config.write_precision,
                )

                with self._stats_lock:
                    self.stats["total_writes"] += 1
                    self.stats["total_points"] += len(batch)

                logger.debug(
                    f"Thread {thread_id}: Successfully wrote {len(batch)} records"
                )
                return True

            except InfluxDBError as e:
                logger.warning(
                    f"Thread {thread_id}: Write attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2**attempt))
                else:
                    logger.error(
                        f"Thread {thread_id}: Failed to write batch after {self.config.max_retries} attempts"
                    )
                    with self._stats_lock:
                        self.stats["failed_writes"] += 1
                    return False
            except Exception as e:
                logger.error(f"Thread {thread_id}: Unexpected error: {e}")
                with self._stats_lock:
                    self.stats["failed_writes"] += 1
                return False

        return False

    def flush(self):
        """Flush any pending writes."""
        if hasattr(self._local, "write_api"):
            try:
                self._local.write_api.flush()
                logger.debug("Flushed pending writes")
            except Exception as e:
                logger.error(f"Failed to flush writes: {e}")

    def close(self):
        """Close all thread-local clients."""
        self.flush()
        if hasattr(self._local, "write_api"):
            self._local.write_api.close()
        if hasattr(self._local, "client"):
            self._local.client.close()
        logger.debug("ThreadSafeInfluxDBWriter closed")


class MultiThreadedInfluxDBClient(DatabaseClientBase):
    """High-performance multithreaded InfluxDB client with proper work distribution."""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        buffer: Optional[deque] = None,
        num_workers: int = 4,
        batch_size: int = 5000,
        write_precision: str = "ms",
        max_queue_size: int = 100,
        timeout: float = 30.0,
        write_interval: float = 0.5,
    ):
        self.config = InfluxDBWriterConfig(
            url=url,
            token=token,
            org=org,
            bucket=bucket,
            write_precision=write_precision,
            batch_size=batch_size,
        )

        self.num_workers = max(1, num_workers)
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self.work_queue = queue.Queue(maxsize=max_queue_size)
        self.writer = ThreadSafeInfluxDBWriter(self.config)
        self.executor = None
        self._shutdown = False

        super().__init__(
            buffer=buffer,
            write_interval=write_interval,
            write_batch_size=batch_size,
        )

        logger.info(
            f"Initialized MultiThreadedInfluxDBClient with {self.num_workers} workers"
        )

    def convert(self, data: Dict[str, Any]) -> Point:
        """Convert dict to InfluxDB Point."""
        if isinstance(data, Point):
            return data
        try:
            point = Point.from_dict(data)
            return point
        except Exception as e:
            logger.error(f"Failed to convert data to Point: {e}")
            raise

    def _create_batches(self, data: Iterable[Any]) -> List[List[Any]]:
        """Create batches from data efficiently."""
        batches = []
        current_batch = []

        for item in data:
            current_batch.append(item)
            if len(current_batch) >= self.config.batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    def write_sync(
        self, metrics: Union[Iterable[Point], Iterable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Write metrics synchronously using thread pool."""
        if isinstance(metrics, (list, tuple)):
            data = list(metrics)
        else:
            data = list(metrics)

        logger.info(f"write_sync called with {len(data)} metrics")

        if not data:
            logger.info("No data to write, returning early from write_sync")
            return {
                "total_batches": 0,
                "successful_batches": 0,
                "failed_batches": 0,
                "total_points": 0,
                "duration_seconds": 0.0,
                "points_per_second": 0.0,
                "success": True,
                "writer_stats": self.writer.stats.copy(),
            }

        if data and isinstance(data[0], dict):
            data = [self.convert(item) for item in data]

        batches = self._create_batches(data)
        total_points = len(data)

        logger.info(
            f"Writing {total_points} points in {len(batches)} batches using {self.num_workers} workers"
        )

        start_time = time.perf_counter()
        successful_batches = 0
        failed_batches = 0

        with ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix="InfluxWriter"
        ) as executor:
            future_to_batch = {
                executor.submit(
                    self.writer.write_batch, batch, f"worker-{i % self.num_workers}"
                ): batch
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                try:
                    success = future.result(timeout=self.timeout)
                    if success:
                        successful_batches += 1
                    else:
                        failed_batches += 1
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    failed_batches += 1

        duration = time.perf_counter() - start_time

        stats = {
            "total_batches": len(batches),
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_points": total_points,
            "duration_seconds": duration,
            "points_per_second": total_points / duration if duration > 0 else 0,
            "success": failed_batches == 0,
            "writer_stats": self.writer.stats.copy(),
        }

        logger.info(
            f"Write completed: {successful_batches}/{len(batches)} batches successful, "
            f"{stats['points_per_second']:.0f} points/sec"
        )

        return stats

    def write(self, data: Tuple[Any, ...], **kwargs) -> bool:
        """Override base write method for multithreaded batch writing."""
        if not data:
            return True

        points = tuple(
            self.convert(item) if isinstance(item, dict) else item for item in data
        )
        num_points = len(points)

        if num_points == 0:
            return True

        if num_points <= self.num_workers:
            return self.writer.write_batch(list(points))

        sub_batch_size = max(1, (num_points + self.num_workers - 1) // self.num_workers)
        subbatches = [
            points[i : i + sub_batch_size] for i in range(0, num_points, sub_batch_size)
        ]

        successful_subs = 0
        total_subs = len(subbatches)

        with ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix="InfluxSubWriter"
        ) as executor:
            futures = {
                executor.submit(self.writer.write_batch, list(sub), f"sub-{j}"): sub
                for j, sub in enumerate(subbatches)
            }

            for future in as_completed(futures):
                try:
                    if future.result(timeout=self.timeout):
                        successful_subs += 1
                except Exception as e:
                    logger.error(f"Sub-batch future failed: {e}")

        success = successful_subs == total_subs
        if not success:
            logger.warning(
                f"Write batch: {successful_subs}/{total_subs} sub-batches succeeded for {num_points} points"
            )

        return success

    def write_from_buffer(self) -> Dict[str, Any]:
        """Write all data from the internal buffer synchronously."""
        if not self.buffer:
            return {"total_batches": 0, "total_points": 0, "success": True}

        data = list(self.buffer)
        self.buffer.clear()

        return self.write_sync(data)

    def add_to_buffer(self, metrics: Union[Dict[str, Any], Iterable[Dict[str, Any]]]):
        """Add metrics to the internal buffer as Points."""
        if isinstance(metrics, dict):
            self.buffer.append(self.convert(metrics))
        else:
            self.buffer.extend(self.convert(item) for item in metrics)

    def ping(self) -> bool:
        """Ping the InfluxDB server."""
        client = InfluxDBClient(
            url=self.config.url,
            token=self.config.token,
            org=self.config.org,
        )
        try:
            health = client.health()
            return health.status == "pass"
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False
        finally:
            client.close()

    def query(self, query: str, **kwargs) -> Any:
        """Execute a query on InfluxDB."""
        client = InfluxDBClient(
            url=self.config.url,
            token=self.config.token,
            org=self.config.org,
        )
        try:
            query_api: QueryApi = client.query_api()
            result = query_api.query(query=query, org=self.config.org, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
        finally:
            client.close()

    def close(self):
        """Clean up resources."""
        super().close()
        self.writer.flush()
        self.writer.close()
        logger.info("MultiThreadedInfluxDBClient closed")

    def __enter__(self):
        return self

    def __del__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def generate_metrics(count: int):
    """Generate example metrics."""
    import random
    from datetime import datetime, timedelta, timezone

    start_time = datetime.now(timezone.utc) - timedelta(hours=0.5)
    for i in range(count):
        yield {
            "measurement": "benchmark_test",
            "fields": {
                "value": random.randint(1, 100),
                "cpu_usage": random.uniform(0, 100),
                "memory_usage": random.uniform(0, 100),
            },
            "time": start_time + timedelta(milliseconds=i * 10),
            "tags": {
                "host": f"server_{i % 5}",
                "region": random.choice(["us-east", "us-west", "eu-central"]),
            },
        }


def benchmark_workers():
    """Benchmark different worker counts and batch sizes using config file."""
    from config_loader import load_configs

    config_file = "config/influx_config.toml"
    config = load_configs(config_file)

    client_config = {
        "url": config["influx2"]["url"],
        "token": config["influx2"]["token"],
        "org": config["influx2"]["org"],
        "bucket": config["database_client"]["influx"]["default_bucket"],
    }

    num_metrics = 300000
    worker_counts = [1, 2, 4, 8, 12]
    batch_sizes = [2500, 5000, 10000]

    metrics = list(generate_metrics(num_metrics))  # Pre-generate metrics
    print(f"\nBenchmarking with {num_metrics} metrics:")
    print("-" * 60)

    for batch_size in batch_sizes:
        for workers in worker_counts:
            print(f"\nTesting with {workers} workers and batch size {batch_size}...")
            client_config["batch_size"] = batch_size
            with MultiThreadedInfluxDBClient(
                num_workers=workers, **client_config
            ) as client:
                start_time = time.perf_counter()
                stats = client.write_sync(metrics)
                duration = time.perf_counter() - start_time
                print(
                    f"Workers: {workers:2d} | Batch Size: {batch_size:5d} | "
                    f"Duration: {stats['duration_seconds']:6.2f}s | "
                    f"Rate: {stats['points_per_second']:8.0f} points/sec | "
                    f"Success: {stats['success']}"
                )
                logger.info(
                    f"Stats: Total Points: {stats['total_points']}, "
                    f"Successful Batches: {stats['successful_batches']}/{stats['total_batches']}, "
                    f"Failed Batches: {stats['failed_batches']}"
                )


def example_usage():
    """Example usage matching the original pattern."""
    from config_loader import load_configs

    config_file = "config/influx_config.toml"
    config = load_configs(config_file)

    buffer = deque()
    influx_client_config = {
        "url": config["influx2"]["url"],
        "token": config["influx2"]["token"],
        "org": config["influx2"]["org"],
        "bucket": config["database_client"]["influx"]["default_bucket"],
        "batch_size": config["database_client"]["influx"]["write_batch_size"],
        "num_workers": 4,
        "buffer": buffer,
    }

    start_time = time.perf_counter()

    with MultiThreadedInfluxDBClient(**influx_client_config) as client:
        metrics = generate_metrics(100000)
        metrics_list = list(metrics)
        logger.info(f"Generated {len(metrics_list)} metrics")
        client.add_to_buffer(metrics_list)
        logger.info(f"Added {len(buffer)} metrics to buffer")
        stats = client.write_from_buffer()
        logger.info(f"Stats returned: {list(stats.keys())}")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    points_per_sec = stats.get("points_per_second", 0)
    total_points = stats.get("total_points", 0)

    logger.info(
        f"Data ingestion completed in {total_time:.2f} seconds. "
        f"Processed {total_points} points. "
        f"Rate: {points_per_sec:.0f} points/sec"
    )


if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the benchmark
    benchmark_workers()

    # Optionally run the example
    # example_usage()
