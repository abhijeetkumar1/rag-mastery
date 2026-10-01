"""Agent guardrails (Phase 6): an agent decides its own control flow, so it can loop, wander or overspend. Limits are
enforced in CODE, outside the LLM, and every stop is recorded with its reason.

  max_steps        LLM turns (each reasoning step = one LLM call)
  max_tool_calls   tool executions across the run
  max_tokens       prompt + completion tokens across all LLM calls (the context grows every turn: cost is ~quadratic)
  max_cost_usd     the same budget in dollars (common.trace.cost_usd)
  allowed_tools    allowlist: a tool not on it is never executed, even if the LLM (or an injected passage) asks
  loop detection   an identical call (same tool, same normalized args), or a near-duplicate search (same tool and
                   filters, query words ≥ 80% overlapping), is not executed again, and the agent is told to change
                   approach. A step whose calls are ALL repeats is a "stuck" step; after `max_stuck_steps` of them the
                   run stops with stop_reason="loop". (v1 counted repeated CALLS: one batched step that re-issued 5
                   searches counted as 5 and stopped the run before the agent could react to the hint.)
When a limit is hit the agent is not just killed: the graph routes to a "finalize" node that must answer from the
evidence gathered so far (or refuse), so the user still gets the best available answer.
"""
import json
import re
from dataclasses import dataclass, field

from common.trace import cost_usd


@dataclass
class AgentLimits:
    max_steps: int = 8
    max_tool_calls: int = 12
    max_tokens: int = 60_000
    max_cost_usd: float = 0.02
    allowed_tools: frozenset = frozenset({"search_filings", "search_tables", "company_relations", "calculate"})
    max_stuck_steps: int = 1
    similar_query: float = 0.8


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


@dataclass
class Budget:
    limits: AgentLimits
    model: str
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    repeats: int = 0
    stuck_steps: int = 0
    last_cost: float = 0.0  # cost of the most recent LLM call (for its trace span)
    blocked: list = field(default_factory=list)  # (tool, reason) of every call the guard refused
    _seen: list = field(default_factory=list)

    def charge(self, usage: dict | None) -> None:
        """One LLM call. usage = AIMessage.usage_metadata ({input_tokens, output_tokens, ...})."""
        self.steps += 1
        if usage:
            self.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            self.last_cost = cost_usd(self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)) or 0.0
            self.cost += self.last_cost

    def admit(self, name: str, args: dict) -> str | None:
        """Check one tool call BEFORE executing it. Returns None (allowed) or the reason it is refused."""
        if name not in self.limits.allowed_tools:
            reason = f"tool '{name}' is not allowed"
        elif self.tool_calls >= self.limits.max_tool_calls:
            reason = "tool-call budget exhausted"
        else:
            key = (name, json.dumps({k: v for k, v in args.items() if k != "query"}, sort_keys=True))
            q = _words(str(args.get("query", args.get("expression", ""))))
            dup = next((1 for k, w in self._seen if k == key and (q == w or (q and w and len(q & w) / len(q | w) >= self.limits.similar_query))), None)
            if dup:
                self.repeats += 1
                reason = ("repeated call (same tool and arguments as before). Use the passages you already have, or change "
                          "approach: different words, another filing year, or the other search tool (search_filings "
                          "for facts stated in prose, search_tables for figures in financial statements)")
            else:
                self._seen.append((key, q))
                self.tool_calls += 1
                return None
        self.blocked.append((name, reason))
        return reason

    def end_step(self, n_calls: int, n_repeats: int) -> None:
        """Called by the tools node after a step's calls: a step made only of repeats means the agent is stuck."""
        if n_calls and n_repeats == n_calls:
            self.stuck_steps += 1

    def exceeded(self) -> str | None:
        """The reason to stop the run now, or None."""
        lim = self.limits
        if self.stuck_steps > lim.max_stuck_steps:
            return "loop"
        if self.steps >= lim.max_steps:
            return "max_steps"
        if self.tokens >= lim.max_tokens:
            return "max_tokens"
        if self.cost >= lim.max_cost_usd:
            return "max_cost"
        return None

    def summary(self) -> dict:
        return {"steps": self.steps, "tool_calls": self.tool_calls, "tokens": self.tokens, "cost_usd": round(self.cost, 6),
                "repeats": self.repeats, "stuck_steps": self.stuck_steps, "blocked": self.blocked}
