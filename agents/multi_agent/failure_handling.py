"""
Production failure handling for the multi-agent system.

Implements:
  - CircuitBreaker: stops calling a failing agent to prevent cascade failures
  - RetryPolicy: exponential backoff with jitter
  - Timeout: per-agent deadline enforcement
  - GracefulDegradation: fallback chain when primary agents fail

Circuit breaker states:
  CLOSED  -> normal operation, calls pass through
  OPEN    -> agent is failing, calls blocked, fallback used
  HALF_OPEN -> probe call to check if agent recovered

Interview talking point:
  "The circuit breaker prevents a slow or failing agent from blocking the
   entire pipeline. When the verifier is down, the system degrades gracefully
   to returning unverified output with a confidence penalty, rather than
   timing out the user request."
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """
    Circuit breaker pattern for agent calls.

    Tracks consecutive failures. After failure_threshold failures, opens
    the circuit for timeout_seconds. Then allows one probe call (HALF_OPEN).
    If the probe succeeds, closes the circuit. If it fails, re-opens.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time > self.config.timeout_seconds
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit '%s' -> HALF_OPEN (probing)", self.name)
        return self._state

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def allow_call(self) -> bool:
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit '%s' -> CLOSED (recovered)", self.name)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit '%s' -> OPEN (probe failed)", self.name)
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' -> OPEN (%d failures)", self.name, self._failure_count
            )

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure_ago_s": (
                round(time.time() - self._last_failure_time, 1)
                if self._last_failure_time else None
            ),
        }


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryPolicy:
    """
    Exponential backoff with jitter.

    Jitter prevents thundering herd: if 100 agents all fail at once and
    retry at the same interval, they'll all hit the downstream service
    simultaneously. Adding random jitter spreads the load.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def execute(self, fn: Callable[[], T], context: str = "") -> T:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = fn()
                if attempt > 1:
                    logger.info("Retry succeeded on attempt %d for: %s", attempt, context)
                return result
            except Exception as e:
                last_error = e
                if attempt == self.config.max_attempts:
                    break

                delay = min(
                    self.config.base_delay_seconds * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay_seconds,
                )
                if self.config.jitter:
                    delay *= (0.5 + random.random() * 0.5)

                logger.warning(
                    "Attempt %d/%d failed for '%s': %s. Retrying in %.2fs.",
                    attempt, self.config.max_attempts, context, e, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"All {self.config.max_attempts} attempts failed for '{context}'. "
            f"Last error: {last_error}"
        ) from last_error

    def delay_for_attempt(self, attempt: int) -> float:
        delay = min(
            self.config.base_delay_seconds * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay_seconds,
        )
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        return delay


class TimeoutGuard:
    """
    Deadline enforcement for agent calls.
    Uses wall-clock time (not thread-based) for compatibility with async code.
    """

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self._start: Optional[float] = None

    def __enter__(self) -> "TimeoutGuard":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        pass

    def check(self) -> None:
        if self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"Agent exceeded deadline: {elapsed:.2f}s > {self.timeout_seconds}s"
            )

    def remaining(self) -> float:
        if self._start is None:
            return self.timeout_seconds
        return max(0.0, self.timeout_seconds - (time.perf_counter() - self._start))


class GracefulDegradation:
    """
    Fallback chain: try primary, then fallbacks in order.

    Example:
        chain = GracefulDegradation([
            ("gpt4_agent", gpt4_agent.run),
            ("gpt4o_mini_agent", mini_agent.run),
            ("cached_response", cache.get),
        ])
        result = chain.execute(task)

    The first successful result is returned. Each fallback is logged with
    a confidence penalty so downstream consumers know quality may be reduced.
    """

    def __init__(self, chain: list[tuple[str, Callable]]):
        self.chain = chain

    def execute(self, *args, **kwargs) -> tuple[Any, str, float]:
        """
        Returns (result, provider_name, confidence_multiplier).
        confidence_multiplier < 1.0 signals degraded quality.
        """
        for i, (name, fn) in enumerate(self.chain):
            try:
                result = fn(*args, **kwargs)
                confidence_multiplier = 1.0 - (i * 0.15)
                if i > 0:
                    logger.warning(
                        "Degraded to fallback '%s' (level %d, confidence penalty %.0f%%)",
                        name, i, i * 15,
                    )
                return result, name, max(confidence_multiplier, 0.1)
            except Exception as e:
                logger.warning("Provider '%s' failed: %s. Trying next.", name, e)

        raise RuntimeError("All providers in degradation chain failed.")
