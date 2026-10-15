"""Rate limits and spend limits per user (Phase 7), checked BEFORE any work is done.

  TokenBucket       requests per minute. Each user has a bucket of `capacity` tokens that refills at `rate` per
                    second; a request takes one token or gets 429 + Retry-After. Allows short bursts (up to capacity)
                    while bounding the average rate. (Fixed windows let 2x the limit through at a window boundary;
                    sliding logs are exact but store every timestamp.)
  SpendLimiter      dollars per user per day. The cost of a request is known only AFTER the LLM call, so the check
                    uses a PRE-CALL ESTIMATE: prompt tokens counted with tiktoken (exact for the prompt) + max_tokens
                    for the output (an upper bound). Phase 6's Budget charged after each call and overshot its limit
                    by one call; this estimate is what closes that gap. After the call, the estimate is replaced by
                    the actual cost.
In memory and per process here. With several workers or replicas every process would have its own buckets (a user
gets N x the limit), so production keeps them in Redis (atomic INCR/EXPIRE or a Lua token-bucket script) or at the
API gateway.
"""
import threading
import time
from dataclasses import dataclass, field

from common.chunking import count_tokens
from common.trace import cost_usd


def estimate_cost(model: str, messages: list[dict], max_output_tokens: int) -> tuple[int, float]:
    """(prompt tokens, upper-bound cost in USD). ~4 tokens of chat-format overhead per message."""
    tokens = sum(count_tokens(m["content"]) + 4 for m in messages) + 3
    return tokens, cost_usd(model, tokens, max_output_tokens) or 0.0


@dataclass
class TokenBucket:
    rate: float      # tokens added per second
    capacity: float  # burst size
    tokens: float = None
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = self.capacity if self.tokens is None else self.tokens

    def take(self) -> float:
        """0.0 if allowed, else the seconds until a token is available (the Retry-After)."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


class RateLimiter:
    def __init__(self):
        self.buckets: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def check(self, user: str, rpm: float) -> float:
        with self.lock:
            b = self.buckets.setdefault(user, TokenBucket(rate=rpm / 60, capacity=max(1.0, rpm / 6)))  # burst = 10 s
            return b.take()


class SpendLimiter:
    def __init__(self):
        self.spent: dict[tuple[str, str], float] = {}  # (user, day) -> USD
        self.lock = threading.Lock()

    def reserve(self, user: str, limit_usd: float, estimate: float) -> bool:
        """Reserve the estimated cost; False (refuse) if it would push today's spend over the limit."""
        key = (user, time.strftime("%Y-%m-%d"))
        with self.lock:
            if self.spent.get(key, 0.0) + estimate > limit_usd:
                return False
            self.spent[key] = self.spent.get(key, 0.0) + estimate
            return True

    def settle(self, user: str, estimate: float, actual: float) -> None:
        """Swap the reservation for the actual cost (usually much lower: max_tokens is an upper bound)."""
        key = (user, time.strftime("%Y-%m-%d"))
        with self.lock:
            self.spent[key] = self.spent.get(key, 0.0) - estimate + actual

    def today(self, user: str) -> float:
        return self.spent.get((user, time.strftime("%Y-%m-%d")), 0.0)
