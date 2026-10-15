"""Per-user access control for retrieval (Phase 7): a user only ever sees chunks they are entitled to.

In a multi-tenant RAG system (e.g. each customer's own documents, or desks with different research entitlements) the
access check belongs in RETRIEVAL, as a metadata filter applied inside the vector DB and the BM25 index:
  pre-filter (what we do)   the search only considers allowed chunks: nothing else can reach the prompt
  post-filter               search everything, drop disallowed hits afterwards: leaks via counts and timing, and
                            can return fewer than k results (all top hits were someone else's)
  prompt instruction        "don't use documents from X": never an access control. The LLM still reads them.
We do the pre-filter AND assert after retrieval (defense in depth): a filter bug then shows up as a logged
violation with the chunk dropped, not as a leaked answer.

Users here have a ticker allowlist (stand-in for tenant / document ACLs). The scope is part of every cache key.
"""
import json
from dataclasses import dataclass

from common.query import COMPANIES

ALL = tuple(sorted(COMPANIES))


@dataclass(frozen=True)
class User:
    id: str
    tickers: tuple = ALL
    rpm: float = 30           # requests per minute
    daily_usd: float = 1.0    # spend limit per day

    @property
    def scope(self) -> tuple:
        return tuple(sorted(self.tickers))


def load_users(path) -> dict[str, User]:
    """{api_key: User} from a JSON file: [{"api_key", "id", "tickers", "rpm", "daily_usd"}]. Demo only: real keys
    are hashed at rest and issued by an identity provider; users come from a JWT's claims."""
    out = {}
    for u in json.loads(path.read_text()):
        out[u["api_key"]] = User(u["id"], tuple(u.get("tickers") or ALL), u.get("rpm", 30), u.get("daily_usd", 1.0))
    return out


def scoped_filter(f: dict | None, user: User) -> dict | None:
    """Intersect a retrieval filter with the user's entitlement. Returns None if the result is EMPTY: the request
    asked only for companies the user may not see (the caller refuses, it must not search unfiltered)."""
    f = dict(f or {})
    allowed = set(user.tickers)
    want = set(f.get("ticker", allowed)) & allowed
    if not want:
        return None
    if want != set(ALL):
        f["ticker"] = sorted(want)
    else:
        f.pop("ticker", None)
    return f


def violations(hits: list[dict], user: User) -> list[str]:
    """Ids of retrieved chunks the user may NOT see (should always be empty)."""
    return [h["id"] for h in hits if h["meta"]["ticker"] not in user.tickers]
