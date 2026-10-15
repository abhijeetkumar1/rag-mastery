"""FastAPI service over phase7_production/service.py.

Run: uv run uvicorn phase7_production.app:app --port 8000                     # or: uv run python -m phase7_production.01_serve
     LLM_BACKEND=stub uv run uvicorn phase7_production.app:app --port 8000     # no OpenAI calls at all

Endpoints
  GET  /health          liveness: the process is up (cheap, never touches dependencies)
  GET  /ready           readiness: index loaded, models warm, breaker states (a load balancer only routes here when ready)
  POST /ask             JSON in, JSON out (the events collected)
  POST /ask/stream      Server-Sent Events: meta, step, sources, token, guardrails, done | error
  POST /feedback        thumbs up/down on a trace_id
  GET  /metrics         Prometheus text format
Auth: header X-API-Key -> a User (phase7_production/users.json): ticker allowlist, requests/min, $/day.
"""
import json
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common import telemetry
from common.access import User
from phase7_production.service import MAX_QUESTION_CHARS, RAGService

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the index and warm the reranker ONCE, before the first request (Phase 2: a cold reranker = 12 s first request)
    t = time.perf_counter()
    svc = RAGService()
    svc.warmup()
    STATE.update(svc=svc, ready=True, startup_s=round(time.perf_counter() - t, 2))
    yield
    telemetry.flush()


app = FastAPI(title="10-K RAG", version="7.0", lifespan=lifespan)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    mode: Literal["auto", "pipeline", "agent"] = "auto"


class FeedbackIn(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000)


def svc() -> RAGService:
    if not STATE.get("ready"):
        raise HTTPException(503, "starting up")
    return STATE["svc"]


def current_user(x_api_key: str | None = Header(default=None), s: RAGService = Depends(svc)) -> User:
    u = s.user(x_api_key)
    if u is None:
        raise HTTPException(401, "missing or invalid X-API-Key")
    return u


def rate_limited(user: User = Depends(current_user), s: RAGService = Depends(svc)) -> User:
    wait = s.admit(user)
    if wait:
        telemetry.REQUESTS.labels("none", "rate_limited").inc()
        raise HTTPException(429, f"rate limit: {user.rpm:g} requests/minute", headers={"Retry-After": str(int(wait) + 1)})
    return user


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(s: RAGService = Depends(svc)):
    return {"ready": True, "startup_s": STATE["startup_s"], **s.health()}


@app.post("/ask")
def ask(body: AskIn, user: User = Depends(rate_limited), s: RAGService = Depends(svc)):
    # a plain `def` endpoint: FastAPI runs it in a worker thread, so the blocking pipeline doesn't block the event loop
    out = s.ask(body.question, user, body.mode)
    if "error" in out:
        return JSONResponse(out, status_code=out["error"]["http"] if out["error"]["http"] >= 400 else 200)
    return out


@app.post("/ask/stream")
async def ask_stream(body: AskIn, request: Request, user: User = Depends(rate_limited), s: RAGService = Depends(svc)):
    gen = s.stream(body.question, user, body.mode)  # a sync generator: sse-starlette iterates it in a thread pool

    def events():
        try:
            for ev in gen:
                yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        finally:
            gen.close()  # client gone: closing the generator closes the OpenAI stream and ends the trace

    # ping keeps proxies from closing an idle connection while retrieval/agent steps run
    return EventSourceResponse(events(), ping=10)


@app.post("/feedback")
def feedback(body: FeedbackIn, user: User = Depends(current_user), s: RAGService = Depends(svc)):
    s.feedback(body.trace_id, user, body.rating, body.comment)
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(telemetry.REGISTRY), media_type=CONTENT_TYPE_LATEST)
