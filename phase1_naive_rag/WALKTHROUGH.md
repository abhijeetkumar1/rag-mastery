# Phase 1 walkthrough: naive RAG

Read this top to bottom and run each command as you reach it. The theory, the experiments and the interview Q&A are in `NOTES.md`. This file covers **the architecture and how every step works**. The numbers are from the current index (`text-embedding-3-small`, recursive chunks of 350 tokens).

## Architecture

### Two pipelines

```
                         OFFLINE (indexing): run once per corpus / config change
┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐
│ 00 download    │──►│ 01 parse       │──►│ 02 chunk       │──►│ 03 embed + index    │
│ EDGAR → HTML   │   │ HTML → clean   │   │ sections →     │   │ chunks → vectors →  │
│ + manifest     │   │ text by Item   │   │ 2,344 chunks   │   │ Chroma (HNSW)       │
└────────────────┘   └────────────────┘   └────────────────┘   └─────────────────────┘
 data/raw/            data/processed/      data/processed/      .cache/embeddings/
 <T>_FY<Y>.html       <T>_FY<Y>.json       chunks_*.jsonl       data/chroma/

                         ONLINE (query): runs for every question
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ question │──►│ embed_query  │──►│ Chroma top-k │──►│ build prompt │──►│ chat() → answer  │
└──────────┘   │ (same model) │   │ (k=5, cosine)│   │ [1]..[k] +   │   │ with [n] cites + │
               └──────────────┘   └──────────────┘   │ source header│   │ refusal path     │
                                                     └──────────────┘   └──────────────────┘
```

**Design principle:** every step reads the previous step's files and writes its own. This gives you:
- **Inspectability.** Open any JSON or JSONL file and see exactly what a step produced. Every bug in this phase was found this way.
- **Cheap reruns.** Change the chunker and you rerun only 02 → 03; downloading and parsing aren't repeated.
- **Replaceable parts.** Any step can be swapped (a different parser, vector DB or LLM) without touching the others.

### Components

| Layer | Module | Key functions | Tech |
|---|---|---|---|
| Config | `common/config.py` | env loading: `EMBED_PROVIDER`, `EMBED_MODEL`, `CHAT_MODEL`, `SEC_USER_AGENT` | python-dotenv |
| Acquisition | `phase1_naive_rag/00_download_10k.py` | `get`, `ticker_to_cik`, `latest_10ks` | httpx, SEC EDGAR APIs |
| Parsing | `phase1_naive_rag/01_parse.py` | `html_to_text`, `split_sections` | BeautifulSoup + lxml, regex |
| Chunking | `common/chunking.py` + `02_chunk.py` | `recursive_chunks`, `fixed_token_chunks`, `chunk_corpus` | tiktoken (`o200k_base`) |
| Embedding | `common/llm.py` | `embed` (cached), `embed_query` (query prefix) | OpenAI API or sentence-transformers |
| Vector store | `common/store.py` | `collection_name`, `get_collection`, `add_chunks`, `search` | Chroma (PersistentClient, HNSW) |
| Generation | `common/llm.py` + `04_ask.py` | `chat`, `format_context`, `ask` | OpenAI chat (`gpt-4o-mini`, temp 0) |

### Data model: how a record changes shape through the pipeline

```
manifest entry     {ticker, fiscal_year, file, url, accession, filing_date, period_end, primary_doc}
        │  (01) + sections
        ▼
processed doc      {ticker, fiscal_year, filing_date, period_end, url,
                    sections: [{item: "Item 1A", title: "Risk Factors", text: "..."}]}
        │  (02) one row per chunk, metadata copied down
        ▼
chunk row          {id: "AAPL_FY2025_Item8_049", text, ticker, fiscal_year, item, section, url}
        │  (03) + vector
        ▼
Chroma record      id | embedding (1536 floats) | document (= text) | metadata {ticker, fiscal_year, item, section, url}
        │  (04) search result
        ▼
hit                {id, text, meta, score = 1 − cosine distance}
```

The metadata **starts in the manifest and is copied down** to every chunk. It isn't used for retrieval yet. It powers citations now, and filters in Phase 2.

### Corpus at a glance

| | |
|---|---|
| Filings | 10: AAPL, MSFT, NVDA, TSLA, AMZN × 2 latest fiscal years |
| Raw HTML | 31 MB (1.5–8.6 MB per filing) |
| Clean text | 207K–399K characters per filing |
| Chunks | 2,344 recursive (fixed-size: 2,187), mean 286 tokens, 0.67M tokens total |
| Vectors | 1536-d (OpenAI) or 384-d (bge-small); one collection per config |

---

## Step 0: Download (`00_download_10k.py`)

**Goal:** get the source documents plus the metadata that describes them.

```bash
uv run python -m phase1_naive_rag.00_download_10k
```

```
company_tickers.json ──► {"AAPL": 320193, ...}                  ticker_to_cik()   :28
submissions/CIK##########.json ──► filings.recent (parallel lists)  latest_10ks()  :33
    keep form == "10-K" (skip 10-K/A amendments), first 2
for each filing:                                                   main()           :50
    fy  = int(period_end[:4])
    url = /Archives/edgar/data/{cik}/{accession-no-dashes}/{primaryDocument}
    save data/raw/{TICKER}_FY{fy}.html   (skip if it already exists)
write data/raw/manifest.json
```

1. **`get()`** (`:21`) wraps every request. It sleeps `DELAY_S = 0.2` s (about 5 requests per second, under SEC's limit of 10). The client sends `SEC_USER_AGENT` as the User-Agent header (`:56`); SEC blocks anonymous clients. `raise_for_status()` makes an HTTP error stop the script instead of saving an error page as a "10-K".
2. **`ticker_to_cik()`** (`:28`): SEC identifies companies by **CIK** number, not ticker.
3. **`latest_10ks()`** (`:33`): the submissions JSON stores filings as parallel lists (`form[i]`, `accessionNumber[i]`, `filingDate[i]`, `reportDate[i]`, `primaryDocument[i]`). Only an exact `"10-K"` counts, so amendments are skipped.
4. **Fiscal year** (`:62`) = the year the reporting period ends. That matches each company's own naming. NVIDIA's year ending January 2026 is its FY2026, and Microsoft's year ending June 2026 is FY2026.
5. **Idempotent** (`:66`): an existing file is skipped. The manifest is always rewritten.

**Output:** 10 HTML files plus `manifest.json`.

**How to check it worked:** `grep -c "ANNUAL REPORT PURSUANT TO SECTION 13" data/raw/*.html` should give ≥ 1 for every file.

**Things to know:**
- **"Latest year" means different calendar periods per company.** AAPL's latest is FY2025, because its FY2026 10-K isn't filed until late October. MSFT's latest is FY2026.
- **Only the primary document is downloaded.** Exhibits (contracts, certifications) are skipped.

**Interview angle:** in production this is the *connector* layer. The hard parts are incremental sync (only re-process changed documents), deletes (remove their chunks from the index), and carrying **access permissions** through to retrieval time.

---

## Step 1: Parse (`01_parse.py`)

**Goal:** turn about 1.5–8.6 MB of HTML into 200–400K characters of clean text, split into SEC "Item" sections.

```bash
uv run python -m phase1_naive_rag.01_parse
```

### 1a. `html_to_text()` (`:39`), six stages

| # | Lines | What | Why |
|---|---|---|---|
| 1 | `:41-44` | Delete `<ix:header>`, `<script>`, `<style>`, and anything `display:none` | The inline-XBRL header holds thousands of hidden machine-readable facts. If kept, they'd be chunked and retrieved as noise |
| 2 | `:46-49` | Each `<tr>` becomes one line, `cell \| cell \| cell`. Cells that are only `$`, `%` or `)` are dropped | Keeps `Total net sales \| 416,161 \| 391,035 \| 383,285` together. With default extraction, each number ends up on its own line with no label |
| 3 | `:50-54` | Paragraph breaks go only around **block** tags (`p, div, li, h1–h6, table`), and `<br>` becomes a newline | Inline `<span>`s are joined without a break, which rebuilds MSFT's `ITEM 1. B` + `USINESS` |
| 4 | `:56-58` | `get_text("")`, non-breaking spaces → spaces, collapse whitespace, trim around newlines | Normalization |
| 5 | `:59` | Drop lines that fully match `NOISE_RE` (`:32`) | Page clutter: page numbers, "Table of Contents" (up to 81 per filing), bare `PART II` / `Item 8`, `____` rules, `Apple Inc. \| 2025 Form 10-K \| 48` footers |
| 6 | `:60-61` | 3+ newlines → 2; join lone `•` bullets back onto their text | MSFT puts about 115 bullets on their own lines |

### 1b. `split_sections()` (`:65`), a line-by-line state machine

```
current = Cover
for line in text:
    cells = line.split("|")
    if last cell is a number        → TOC row  "Item 1. | Business | 3"   → body text
    elif more than 2 cells          → table data                           → body text
    elif ITEM_RE matches (cells joined) and title ≥ 4 chars, uppercase first, line < 200 chars
                                    → NEW SECTION (Item N, title)
    else                            → body text of current section
drop sections whose body ≤ 200 chars ("Item 6. [Reserved]")
```

- **`ITEM_RE`** (`:28`) matches `Item 7A. Title`, `ITEM 1: Title`, `Item 1 — Title`, and similar forms.
- **Joining cells** before matching handles Amazon, which writes real headings as 2-cell table rows (`Item 1. | Business`).
- **The title rules** reject running headers (`Item 1` with no title) and cross-references inside sentences ("Item 7 of this report..." has a lowercase next word).

**Output:** `data/processed/AAPL_FY2025.json` = `{ticker, fiscal_year, filing_date, period_end, url, sections:[{item, title, text}]}`

**How to check it worked:** the script prints each file's sections.
```
AAPL_FY2025.html   html= 1.5MB -> text= 210K chars  sections: Cover, 1, 1A, 1C, 2, 3, 5, 7, 7A, 8, 9A, 9B, 10, 15, 16
```
- A file listing **only `Cover`** means section detection failed on it. That's how the Amazon bug was found.
- To find leftover clutter, count the most frequent short lines in each file. That's how "Table of Contents" and the Apple footer were found.

**Things to know:**
- **NVIDIA has no Item 8.** Its financial statements are in Item 15. That's a real quirk of its filing, not a parser bug.
- **Noise isn't neutral.** The Apple footer raised a *wrong-year* table from 0.587 to 0.719 and ranked it #1. See NOTES.md "Noise isn't neutral".
- **Parsing fails without errors.** Nothing crashes; the output is just wrong. Always inspect it.

---

## Step 2: Chunk (`02_chunk.py` + `common/chunking.py`)

**Goal:** cut each section into pieces of at most `size` tokens, each carrying metadata.

```bash
uv run python -m phase1_naive_rag.02_chunk                                     # recursive 350/50 (default)
uv run python -m phase1_naive_rag.02_chunk --chunker fixed --size 200 --overlap 20
```

### 2a. Token counting (`chunking.py:10-14`)

Sizes are counted in **tokens**, because every model limit is in tokens. tiktoken's `o200k_base` (the GPT-4o tokenizer) is the general-purpose counter.

### 2b. Strategy A: `fixed_token_chunks()` (`chunking.py:17`)

```
ids = encode(text)                          e.g. 1,000 tokens
windows: [0:350] [300:650] [600:950] [900:1000]      step = size − overlap = 300
decode each window back to text
```
Every chunk has exactly `size` tokens, but the cuts ignore sentences and table rows.

### 2c. Strategy B: `recursive_chunks()` (`chunking.py:40`), the default

**Split: `_split()` (`:26`).**
```
_split(text, seps=["\n\n", "\n", ". ", " "]):
    if tokens(text) ≤ size: return [text]
    if no seps left:        return hard token cut
    for part in text.split(seps[0]):
        recurse on part + seps[0] with seps[1:]     # keeps the separator, so joining the parts gives back the original text
```
Result: pieces that are each ≤ size and break at the most natural boundary available (paragraph, then line or table row, then sentence, then word).

**Pack (`:45-63`).**
```
cur = []
for piece in pieces:
    if cur + piece > size:
        emit chunk(cur)
        carry = trailing whole pieces of cur totalling ≤ overlap (50) tokens
        drop carried pieces from the front while carry + piece > size      # trim guard, :57
        cur = carry
    cur += piece
emit last chunk
```
- **Overlap** carries only **whole pieces**. If the previous chunk ended with a 200-token paragraph, nothing is carried. The effective overlap varies from 0 to 50 tokens.
- **The trim guard** fixed a bug where the carried overlap plus the next piece produced 388-token chunks.

### 2d. `chunk_corpus()` (`02_chunk.py:19`)

```
for each processed file (sorted):
    for each section:
        for i, text in enumerate(chunker(section.text)):      # chunks never cross an Item boundary
            row = {id: f"{ticker}_FY{year}_{Item}_{i:03d}", text, ticker, fiscal_year, item, section[:120], url}
```
- **Structure-aware + recursive:** structure first (Items), then natural boundaries inside each Item.
- **IDs are deterministic,** so a rerun produces the same IDs. That makes results easy to compare between runs, and in production it enables in-place updates and deletes.

### 2e. `embedder_max_tokens()` (`02_chunk.py:38`)

This only runs for HF models. The embedding model has its **own tokenizer** and limit (bge-small: 512), and **longer text is silently truncated**. The check counts every chunk with bge's tokenizer; the result was 0 truncated. For OpenAI it's skipped, because the 8,191-token limit is far above 350.

**Output:** `data/processed/chunks_recursive350.jsonl`, one JSON object per line.

**How to check it worked:**
```
recursive size=350 overlap=50: 2344 chunks, tokens mean=286 median=309 min=17 max=350, total=0.67M tokens
fixed     size=350 overlap=50: 2187 chunks, tokens mean=337 median=350 min=35 max=350, total=0.74M tokens
```
- `max ≤ size` must hold.
- The script also prints one boundary (the end of chunk 40 and the start of chunk 41). With recursive chunking it should fall at a paragraph end.
- Fixed-size chunking stores more total tokens for fewer chunks, because each window repeats 50 tokens of overlap.

**Things to know:**
- **Chunk size is the main trade-off.** Small chunks give precise vectors with little context; large ones carry more context with diluted vectors. The prompt cost is k × size.
- **Mixed-topic chunks:** packing to 350 tokens joins unrelated neighbours. Tesla's Musk risk shared a chunk with credit risk (score 0.717), and Apple's net-sales table shares one with the start of the auditor's report.

---

## Step 3: Embed and index (`03_index.py` + `common/store.py` + `common/llm.py`)

**Goal:** turn each chunk into a vector and store everything in a persistent, searchable index.

```bash
uv run python -m phase1_naive_rag.03_index                            # indexes chunks_recursive350
uv run python -m phase1_naive_rag.03_index --chunker fixed --size 350 # flags must match what 02 produced
```

### 3a. Flow of `main()` (`03_index.py:22`)

```
rows  = read chunks_<chunker><size>.jsonl
name  = collection_name(chunker, size)          → "10k_text-embedding-3-small_recursive350"   store.py:14
col   = get_collection(name, reset=True)         → drop + recreate, hnsw:space=cosine          store.py:20
add_chunks(col, ids, texts, metadatas)           → embed all, then col.add() in batches of 1000 store.py:28
3 test queries → search(col, q, k=3) → print
```

- **The collection name encodes the model and the chunking settings.** Each configuration gets its own collection, and vectors from different models never share one. Mixing them wouldn't raise an error; the search results would just be meaningless.
- **`reset=True`** means a rebuild never leaves stale chunks from a previous run behind.
- **`hnsw:space = cosine`:** our vectors are unit-length, so this gives the same ranking as dot product (Phase 0).
- **Chroma = storage + ANN index + filters.** We compute the embeddings ourselves (we don't use Chroma's embedding functions), so we control batching and caching, and the DB can be swapped (pgvector or Qdrant in Phase 7) without touching the embedding code.

### 3b. `embed()` (`llm.py:50`), the cached embedder

```
for each text:  key = sha256("text-embedding-3-small::" + text)
                cache hit? load : add to misses
misses → batches of 256 → OpenAI embeddings API (or local bge) → write to .cache/embeddings/<key>.json
return (2344, 1536) float32
```
- After the parser fixes, re-indexing took **5–9 s instead of about 21 s**. Only chunks whose text changed missed the cache. This is hash(model + text) caching, the standard production pattern.
- **Disk usage:** the cache is about 130 MB, because each vector is stored as a JSON text file. In production you'd store float32 binary (about 4× smaller) in a key-value store.

**Output:** a Chroma collection in `data/chroma/` (SQLite plus HNSW files) with 2,344 records.

**How to check it worked:** it prints `indexed 2344 chunks into '10k_text-embedding-3-small_recursive350' ...`. The count must equal the number of JSONL lines, and the test queries should return the right company and section (the NVIDIA query → NVDA Item 1A).

---

## Step 4: Ask (`04_ask.py`)

**Goal:** answer a question grounded in retrieved chunks, with citations.

```bash
uv run python -m phase1_naive_rag.04_ask                                  # 4 demo questions
uv run python -m phase1_naive_rag.04_ask "What was Amazon's AWS operating income in 2025?"
uv run python -m phase1_naive_rag.04_ask -k 10 --show-context "..."       # more hits, show chunk text
```
`--chunker/--size` must match an index you've built. If the collection is empty, the script stops with a clear message (`:63`).

### 4a. Embed the query: `embed_query()` (`llm.py:89`)

```
query_prefix(model)  (llm.py:83)
  BGE en v1.5 → "Represent this sentence for searching relevant passages: " + query
  OpenAI      → query unchanged (symmetric model)
embed([...])[0] → (1536,) vector, SAME model as the index
```
- **Asymmetric models** (BGE, E5) embed questions and passages differently; the instruction goes on the **query side only**. Forgetting it lowers quality without any error.
- **Same model as the index:** a query embedded with another model is in a different vector space, and the search returns nonsense.

### 4b. Retrieve: `search()` (`store.py:35`)

```
col.query(query_embeddings=[q], n_results=k, where=None)     # HNSW approximate k-NN
→ ids, documents, metadatas, distances
score = 1 − distance      (Chroma returns cosine DISTANCE)
→ [{id, text, meta, score}, ...] ordered best first
```
`where=` supports metadata filters (e.g. `{"ticker": "AMZN"}`). It isn't used in naive RAG; it's the first fix in Phase 2.

### 4c. Build the prompt: `format_context()` (`04_ask.py:36`) and `ask()` (`:44`)

```
system: SYSTEM (04_ask.py:17)
user:   Context:

        [1] AAPL 10-K FY2025, Item 8: Financial Statements and Supplementary Data
        2025 | 2024 | 2023
        Net sales: ...
        Total net sales | 416,161 | 391,035 | 383,285
        ---
        [2] AAPL 10-K FY2024, Item 8: ...
        ...
        Question: What was Apple's total net sales in fiscal 2025?
```
The **source header** on each passage matters. Risk-factor text says "we", never "Tesla", so without the header the LLM can't tell which company a passage is about.

The `SYSTEM` rules:
1. Use **only** the passages.
2. Cite every claim as `[n]`.
3. If the answer isn't there, say "I don't know based on the provided filings". This is the refusal path.
4. State units (tables are in millions).
5. Be concise.

### 4d. Generate: `chat()` (`llm.py:93`)

gpt-4o-mini at **temperature 0**, so the same input gives a repeatable answer. That matters for debugging and evaluation.

**Cost per question:** about 1,700 prompt tokens with k=5 and 350-token chunks. It grows roughly as k × chunk size.

### 4e. Report (`04_ask.py:66-76`)

```
answer text
cited = {n for each "[n]" in answer}                      (regex, :69)
for each hit: " *[n] score  TICKER FY item  (chunk_id)"   * = cited
```
Comparing **retrieved against cited** is your first debugging tool. If the right chunk was retrieved but not cited, look at generation. If it wasn't retrieved at all, look at retrieval.

### Current results

| Question | Result | Why |
|---|---|---|
| Apple FY2025 total net sales | ✅ $416,161M [1] | The right table is #1 after the footer fix. Before it, a 2023 table was #1 and the LLM rescued the answer from #2 |
| NVIDIA export-control risk | ✅ cites [1][2][3] | FY2025 and FY2026 risk factors are near-duplicates, so the answer blends both years |
| MSFT vs AMZN revenue growth | ❌ "I don't know" | **5/5 hits are MSFT.** One query vector lands near one company, and top-k can't guarantee coverage of both |
| Google ad revenue (not in corpus) | ✅ refuses | Top score 0.50 against 0.65–0.75 for real hits. The retriever still returns 5 chunks, and the prompt's refusal rule is what saves it |

---

## What to rerun after a change

| You changed... | Rerun |
|---|---|
| Tickers or years (`00`) | 00 → 01 → 02 → 03 |
| Parser rules (`01_parse.py`) | 01 → 02 → 03 (the cache re-embeds only changed chunks) |
| Chunk strategy, size or overlap | 02 `--chunker X --size N` → 03 with the same flags → 04 with the same flags |
| Embedding model or provider (`.env`) | 03 (a new collection is created automatically) → 04 |
| Prompt, `k` or chat model | 04 only |

## Debugging order when an answer is wrong

1. **Was the fact extracted?** Search `data/processed/<file>.json`. If it's missing, it's a **parse** problem.
2. **Is it in one clean chunk?** Search `chunks_*.jsonl`. If it's split or mixed with another topic, it's a **chunking** problem.
3. **Was it retrieved?** Use `--show-context -k 20`. If it's retrieved but ranked low, it's a **ranking** problem (Phases 2–3).
4. **Retrieved, but the answer is still wrong?** Then it's a **generation** problem: the prompt, context order, or missing context in the chunk (Phase 5).

Check retrieval before generation. This is the standard answer to "your RAG gives wrong answers, how do you debug it?"

## Where naive RAG breaks, and which phase fixes it

| Failure seen here | Fix | Phase |
|---|---|---|
| Comparison question gets only one company | Metadata filters, query decomposition, multi-query | 2, 3 |
| Exact terms (tickers, "Item 1A", product names) | BM25 + dense hybrid, RRF | 2 |
| Wrong-year table ranked #1; near-duplicate FY chunks | Cross-encoder reranking, MMR diversity | 2 |
| Vague questions ("Microsoft's revenue?") | Query rewriting, HyDE | 3 |
| "We" chunks and tables without context; mixed-topic chunks | Contextual retrieval, parent-child chunks | 5 |
| "Does it actually work?" | Golden set, recall@k, faithfulness | 4 |

## Self-check before moving on
- [ ] Explain why the Apple footer made a 2023 table rank above the correct 2025 table.
- [ ] Why must chunks never cross an Item boundary? What would break if they did?
- [ ] What happens if you index with bge-small and query with text-embedding-3-small?
- [ ] Why do the MSFT vs AMZN results fail the same way with both embedding models?
- [ ] Walk through the four-step debugging order for "the answer is wrong".
- [ ] `fiscal_year` metadata is the *filing* year. Why wouldn't filtering to FY2025 exclude the 2023 table?
