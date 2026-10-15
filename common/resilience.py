"""Graceful degradation (Phase 7): what the service does when a dependency fails.

A RAG request depends on several remote calls (embedding API, chat API, moderation API). Production systems assume
each will fail sometimes (rate limit, outage, expired key, billing) and decide IN ADVANCE what to serve instead:

  dependency            fallback                                         what the user gets
  query embedding       BM25 only (skip the dense side)                  slightly worse retrieval, still an answer
  planner (chat_json)   rule-based plan (extract_filters, one sub-question per named company)
  generation (chat)     extractive answer: the best passage's sentences, cited, marked "degraded"
  moderation            fail OPEN (log it): low-harm domain; a medical/child-safety app would fail CLOSED

A CircuitBreaker sits in front of each dependency. After `threshold` consecutive failures it OPENS: calls are not
attempted for `cooldown_s` (they would fail anyway, each after its own timeout and retries), the fallback is used
immediately. After the cooldown one trial call is let through (HALF-OPEN): success closes the breaker, failure
re-opens it. Without a breaker, an outage makes EVERY request wait for N retries x timeout before falling back.

This module exists because of a real event: during Phase 6 the OpenAI account returned 429 billing_not_active, and
by Phase 7 the key had expired (401). The service has to keep answering anyway.
"""
import threading
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    name: str
    threshold: int = 3
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if time.monotonic() - self.opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        """Should we try the real call? closed: yes; open: no; half-open: yes (one trial)."""
        return self.state != "open"

    def success(self) -> None:
        with self._lock:
            self.failures, self.opened_at, self.last_error = 0, None, None

    def failure(self, err: Exception) -> None:
        with self._lock:
            self.failures += 1
            self.last_error = f"{type(err).__name__}: {str(err)[:200]}"
            if self.failures >= self.threshold or self.opened_at is not None:  # a failed half-open trial re-opens at once
                self.opened_at = time.monotonic()

    def call(self, fn, fallback, *args, **kw):
        """fn(*args, **kw), or fallback(*args, **kw) if the breaker is open or fn raises. Returns (result, degraded)."""
        if not self.allow():
            return fallback(*args, **kw), True
        try:
            out = fn(*args, **kw)
        except Exception as e:  # noqa: BLE001: any failure of a dependency means "use the fallback"
            self.failure(e)
            return fallback(*args, **kw), True
        self.success()
        return out, False

    def status(self) -> dict:
        return {"state": self.state, "failures": self.failures, "last_error": self.last_error}


# one breaker per dependency, shared by all requests in the process
BREAKERS = {name: CircuitBreaker(name) for name in ("embed", "plan", "generate", "moderation")}
