#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2023-11-19
# version ='1.0'
# ---------------------------------------------------------------------------
"""Demonstration of how to use the FastInfluxDBClient class to send metrics to InfluxDB server"""

# ---------------------------------------------------------------------------

from fast_influxdb_client import FastInfluxDBClient, InfluxMetric
import random
import time
import logging
from datetime import datetime, timezone


def setup_logging(influxdb_handler):
    # get __main__ logger
    logger = logging.getLogger("fast_influxdb_client.fast_influxdb_client")
    logger.setLevel(logging.DEBUG)

    # setup logging to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # setup logging to influxdb
    influxdb_handler.setLevel(logging.INFO)

    # setup logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    influxdb_handler.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(influxdb_handler)

    return logger


def main():
    bucket = "metrics2"
    config_file = "config.toml"
    # Create new client
    client = FastInfluxDBClient.from_config_file(config_file=config_file)
    print(f"{client=}")
    client.default_bucket = bucket
    # client.create_bucket(bucket)
    logger = setup_logging(client.get_logging_handler())
    client.update_bucket(bucket, retention_duration="10d")

    # Generate some random data, and send to influxdb server
    while 1:
        data = random.random()
        data2 = random.randint(0, 100)
        data3 = random.choice([True, False])

        metric = InfluxMetric(
            measurement="py_metric1",
            fields={"data1": data, "data2": data2, "data3": data3},
            time=datetime.now(timezone.utc),
        )

        client.write_metric(metric)
        # logger.info(f"Sent metric: {metric}")
        time.sleep(1)


if __name__ == "__main__":
    main()
