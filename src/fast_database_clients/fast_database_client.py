#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2024-01-23
# Copyright © 2024 Davidson Engineering Ltd.
# ---------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Protocol, TypeVar, runtime_checkable, Optional, Tuple
import itertools
import threading
import time
import logging

logger = logging.getLogger(__name__)

MAX_BUFFER_LENGTH = 65_536
WRITE_BATCH_SIZE = 5_000

T = TypeVar("T")

@runtime_checkable
class SupportsPopleft(Protocol[T]):
    """Minimal queue-like protocol this class relies on."""
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterable[T]: ...
    def popleft(self) -> T: ...

class DatabaseClientBase(ABC):
    def __init__(
        self,
        buffer: Optional[SupportsPopleft[T]] = None,
        write_interval: float = 0.5,
        write_batch_size: int = WRITE_BATCH_SIZE,
        **kwargs,
    ) -> None:
        self._kwargs = kwargs
        self._client = None
        self._last_write_time = time.time()
        self.write_interval = float(write_interval)
        self.write_batch_size = int(write_batch_size)
        self.buffer: SupportsPopleft[T] = (
            buffer if buffer is not None else deque(maxlen=MAX_BUFFER_LENGTH)
        )
        self._stop_event = threading.Event()
        self.write_thread: Optional[threading.Thread] = None

    @abstractmethod
    def ping(self) -> bool: ...

    @abstractmethod
    def write(self, data: Iterable[T], **kwargs) -> None: ...

    @abstractmethod
    def query(self, query: str, **kwargs): ...

    def convert(self, data: T):
        return data

    def close(self) -> None:
        """Shutdown the client."""
        self.stop()
        # If a thread is running, join it safely
        wt = getattr(self, "write_thread", None)
        if isinstance(wt, threading.Thread) and wt.is_alive():
            wt.join(timeout=5.0)

    def write_periodically(self) -> None:
        try:
            self.__enter__()  # Open resources
            while not self._stop_event.is_set():
                time_condition = (time.time() - self._last_write_time) > self.write_interval
                buf_len = len(self.buffer)
                if buf_len and (buf_len >= self.write_batch_size or time_condition):
                    now = time.time()
                    batch_size = min(self.write_batch_size, buf_len)

                    # Snapshot the items to write without mutating the buffer during iteration
                    metrics: Tuple[T, ...] = tuple(itertools.islice(iter(self.buffer), batch_size))

                    # Remove up to batch_size items; guard against concurrent underrun
                    popped = 0
                    while popped < batch_size:
                        try:
                            self.buffer.popleft()
                            popped += 1
                        except IndexError:
                            # Another thread may have drained the buffer; this is fine.
                            logger.debug(
                                "Buffer underrun while popping (popped=%d, target=%d).",
                                popped, batch_size
                            )
                            break

                    if metrics:
                        try:
                            self.write(metrics)
                        except Exception as e:
                            logger.error("Write operation failed: %s", e, exc_info=True)

                    self._last_write_time = now
                else:
                    time.sleep(min(self.write_interval, 0.1))
        finally:
            self.__exit__(None, None, None)  # Ensure cleanup

    def start(self) -> None:
        if self.write_thread and self.write_thread.is_alive():
            return
        self._stop_event.clear()
        self.write_thread = threading.Thread(target=self.write_periodically, daemon=True)
        self.write_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def __del__(self) -> None:
        # Best-effort cleanup; avoid raising in GC
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "DatabaseClientBase[T]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
