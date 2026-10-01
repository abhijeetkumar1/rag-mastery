"""Contextual Retrieval (Anthropic, 2024): an LLM writes 1-2 sentences that situate each chunk in its filing, and the
sentences are prepended to the chunk BEFORE embedding and BM25 indexing. Output: data/processed/chunks_llmctx350.jsonl

Run: uv run python -m phase5_advanced_indexing.03_contextual            # ~3,000 gpt-4o-mini calls first time (~$0.6), cached
     uv run python -m phase5_advanced_indexing.03_contextual --limit 20 # try it on 20 chunks

Input: chunks_structctx350.jsonl (02_chunk). Each chunk's prompt: the deterministic header, the previous chunk
(the text just before it in the section) and the start of the next one. Calls run 3 at a time (I/O bound; 8 hit the 200k tokens/minute limit) and are
cached by chat_json, so a rerun costs nothing and gives the same contexts.

This is the INDEX-TIME cost of the technique: one LLM call per chunk, paid again whenever chunking changes.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import RateLimitError

from common.chunking import _enc, count_tokens
from common.config import CHAT_MODEL, DATA_DIR
from common.contextual import llm_context
from common.llm import client
from common.trace import cost_usd

PROC = DATA_DIR / "processed"


def clip(text: str, n: int, tail: bool = False) -> str:
    ids = _enc.encode(text)
    return _enc.decode(ids[-n:] if tail else ids[:n]) if len(ids) > n else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)  # 8 exceeded gpt-4o-mini's 200k tokens/minute
    args = ap.parse_args()
    rows = [json.loads(line) for line in (PROC / "chunks_structctx350.jsonl").read_text().splitlines()]
    todo = rows[: args.limit] if args.limit else rows
    client()  # create the shared OpenAI client before the threads do

    def neighbours(i: int) -> tuple[str, str]:
        same = lambda j: 0 <= j < len(rows) and rows[j]["id"].rsplit("_", 1)[0] == rows[i]["id"].rsplit("_", 1)[0]
        before = clip(rows[i - 1]["body"], 400, tail=True) if same(i - 1) else "(start of section)"
        after = clip(rows[i + 1]["body"], 200) if same(i + 1) else "(end of section)"
        return before, after

    def work(i: int):
        before, after = neighbours(i)
        r = rows[i]
        for attempt in range(6):  # on top of the client's own retries: the first full run hit the 200k TPM limit
            try:
                return llm_context(f"{r['ctx_header']} section title: {r['section']}", before, r["body"], after)
            except RateLimitError:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"rate limited too often on {r['id']}")

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(work, range(len(todo))))
    tin = sum(u["input_tokens"] for _, u in results)
    tout = sum(u["output_tokens"] for _, u in results)
    cached = sum(u["cached"] for _, u in results)
    print(f"{len(todo)} contexts in {time.time() - t0:.0f}s ({cached} cached), {tin} in / {tout} out tokens "
          f"= ${cost_usd(CHAT_MODEL, tin, tout) or 0:.3f} with {CHAT_MODEL}")

    out = []
    for r, (ctx, _) in zip(todo, results):
        out.append({**r, "llm_context": ctx, "text": f"{r['ctx_header']}\n{ctx}\n{r['body']}"})
    if not args.limit:
        (PROC / "chunks_llmctx350.jsonl").write_text("\n".join(json.dumps(r) for r in out))
        print(f"wrote chunks_llmctx350.jsonl ({len(out)} chunks)")
    ctx_tokens = [count_tokens(r["llm_context"]) for r in out]
    print(f"context: mean {sum(ctx_tokens) / len(ctx_tokens):.0f} tokens per chunk\n")
    for r in [x for x in out if x["kind"] == "table"][:3] + [x for x in out if x["kind"] == "text"][:2]:
        print(f"--- {r['id']} ({r['kind']})\n  {r['llm_context']}\n  body: {r['body'][:150]!r}")


if __name__ == "__main__":
    main()
