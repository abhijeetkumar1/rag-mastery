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
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ question │──►│ embed_query  │──►│ Chroma top-k │──►│ build prompt │──►│ generate         │──►│ guardrails   │──► answer
└──────────┘   │ (same model) │   │ (k=5, cosine)│   │ [1]..[k] +   │   │ gpt-4o-mini,     │   │ citations +  │    + warnings
               └──────────────┘   └──────────────┘   │ source header│   │ [n] cites +      │   │ numeric      │
                                                     └──────────────┘   │ refusal path     │   │ grounding    │
                                                                        └──────────────────┘   └──────────────┘
               ├──── span ────┤   ├──── span ────┤                      ├────── span ──────┤   ├─── span ────┤
               └──────────────────────── one Trace per question → data/traces/YYYY-MM-DD.jsonl ─────────────┘
                                                                                     read by 06_traces (timeline + p50/p95, cost, refusal rate)
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
| Vector store | `common/store.py` | `collection_name`, `get_collection`, `add_chunks`, `search`, `search_by_vector` | Chroma (PersistentClient, HNSW) |
| Generation | `common/llm.py` + `04_ask.py` | `chat_completion`, `chat`, `format_context`, `ask` | OpenAI chat (`gpt-4o-mini`, temp 0) |
| Guardrails | `common/guardrails.py` + `05_guardrails_demo.py` | `check_answer`, `check_citations`, `check_numeric_grounding` | Plain Python + regex (no LLM call) |
| Observability | `common/trace.py` + `06_traces.py` | `Trace`, `span()`, `cost_usd`, `load_traces` | JSONL traces, NumPy percentiles |

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

### 📘 Deep dive: SEC EDGAR and the 10-K

**EDGAR** is the SEC's public filing system. Every US-listed company files its reports there, free to download. Three identifiers matter:

| Identifier | Example | Meaning |
|---|---|---|
| **CIK** (Central Index Key) | `320193` (Apple) | A permanent company ID. Tickers can change; the CIK doesn't. URLs use it padded to 10 digits: `CIK0000320193` |
| **Accession number** | `0000320193-25-000079` | A unique ID for each filing: `<filer CIK>-<year>-<sequence>`. The folder URL uses it without dashes |
| **Form type** | `10-K`, `10-K/A`, `10-Q`, `8-K` | Annual report, amended annual report, quarterly report, material event |

The two JSON APIs we use:
- `sec.gov/files/company_tickers.json`: ticker → CIK for every listed company.
- `data.sec.gov/submissions/CIK##########.json`: a company's filing history. `filings.recent` holds about the last 1,000 filings as parallel lists.

**Fair-access policy:** at most 10 requests per second, and a `User-Agent` with a name and email is mandatory. Clients that break either rule get blocked by IP.

**What a 10-K contains.** Its structure is set by regulation, which is why splitting by Item works:

| Item | Content | Typical size here | RAG relevance |
|---|---|---|---|
| 1 Business | What the company does, segments, competition | 13–54K chars | "What does X do?" |
| **1A Risk Factors** | Every risk the company must disclose | **59–115K chars** (the biggest prose section) | Risk questions. Near-identical from year to year |
| 1B / 1C | Unresolved SEC comments / cybersecurity | small | |
| 2, 3, 4 | Properties, legal proceedings, mine safety | small | |
| 5 | Stock market info, buybacks | small | |
| 6 | [Reserved] (removed by the SEC in 2021) | empty, dropped | |
| **7 MD&A** | Management's Discussion & Analysis: *why* the numbers moved | 15–55K chars | "Why did revenue grow?" Narrative plus tables |
| 7A | Market risk (interest rates, FX) | small | |
| **8 Financial Statements** | Income statement, balance sheet, cash flows, notes | 60–173K chars | Exact numbers. **NVDA puts these under Item 15** |
| 9, 9A–9C | Accountant changes, internal controls | small | |
| 10–14 | Directors, executive pay, ownership | **nearly empty** | Usually "incorporated by reference" to the separate proxy statement (DEF 14A), so the text isn't in the 10-K |
| 15, 16 | Exhibits index, summary | varies | |

**Interview angle:** knowing your documents' structure is the cheapest retrieval improvement there is. It gives you section-aware chunking, section metadata for filters, and the knowledge that exec-pay questions can't be answered from 10-Ks alone.

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

### 📘 Deep dive: inline XBRL, BeautifulSoup and lxml

**Inline XBRL (iXBRL).** Since 2019–2021 the SEC has required financial filings to be **machine-readable HTML**. It's a single XHTML file that browsers display normally but that also carries tagged data:
```html
<ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
                contextRef="c-1" unitRef="usd" decimals="-6" scale="6">416,161</ix:nonFraction>
```
- **Tagged numbers:** every number in the financial statements is wrapped like this (AAPL FY2025 has 969 `ix:nonFraction` tags), with a standard **us-gaap concept name**, a period (via `contextRef`), a unit and a scale (`6` = millions).
- **`<ix:header>`:** a hidden block holding contexts, units and hidden facts. We delete it (its contents don't belong in retrieval).
- **Why MSFT's files are about 4× bigger:** more tags, more `<span>`s and more inline styles. The text itself is similar in size.
- **Worth knowing:** for exact numeric questions, the XBRL facts are **better than text retrieval**. They're structured, labeled and period-tagged. SEC even serves them as JSON (`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`). A production financial assistant would route numeric questions to structured data and prose questions to RAG. That's the text-to-SQL / structured-retrieval idea in Phase 5 and Phase 6 routing.

**BeautifulSoup + lxml.**
- **lxml** is a fast C library that parses HTML or XML into a tree.
- **BeautifulSoup** is a friendlier Python API on top: `find_all(tag)`, CSS `select()`, `get_text()`, and tree edits (`decompose()` removes a node, `replace_with()` swaps it, `insert_before/after` adds text).
- The `XMLParsedAsHTMLWarning` we suppress happens because iXBRL is technically XHTML. Parsing it as HTML with lxml is fine for text extraction; it just loses the XML namespace precision, which we don't need.
- **Why edit the tree before `get_text()`:** text extraction is lossy. Once everything is flat text, you can't tell a table cell from a paragraph or an inline span from a block. So all structure decisions (row → `a | b | c`, block → blank line) happen **on the tree**, and the flattening happens last.

**Production alternatives:**
- **Unstructured, Docling, LlamaParse, Azure Document Intelligence:** they handle PDFs, scans and tables, returning typed elements (Title, NarrativeText, Table).
- **Hand-written parsers like ours:** they win when you know the format well, as with 10-Ks. Generic parsers win across many formats.

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

### 📘 Deep dive: tokens and tiktoken

**What a token is.** LLMs and embedding models don't read characters or words. They read **tokens**: sub-word pieces from a fixed vocabulary, learned with **BPE** (Byte-Pair Encoding).
- **How BPE builds its vocabulary:** start from bytes, then repeatedly merge the most frequent adjacent pair into a new token, until the vocabulary reaches its target size.
- **Common words** end up as 1 token (` revenue`), and rare strings split into several (`E-4471` → `E`, `-`, `447`, `1`). That's one reason exact IDs embed poorly (Phase 0).
- **Rule of thumb** for English: 1 token ≈ 4 characters ≈ 0.75 words. Tables with lots of numbers and `|` use more tokens per character.

**tiktoken** is OpenAI's fast BPE tokenizer library, with a Rust core.
- **`o200k_base`** (200k vocabulary) is the encoding for GPT-4o and gpt-4o-mini. `cl100k_base` is the older one (GPT-4, text-embedding-3).
- Counting with `o200k_base` is a good estimate for **prompt cost**. It's *not* exactly what the embedder sees: text-embedding-3 uses `cl100k_base`, and bge uses a BERT WordPiece vocabulary of about 30k tokens. That's why `02_chunk.py` re-counts with **bge's own tokenizer** before trusting the 512 limit.

**Why sizes are in tokens, not characters:**
- **Model limits** (bge 512, OpenAI embeddings 8,191, the chat context window) are in tokens.
- **Prices** are per token.
- **Characters per token vary.** A 350-token chunk of prose is about 1,500 characters; a 350-token numeric table is less.

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

### 📘 Deep dive: embedding models

An embedding model is a neural network (a Transformer encoder) that maps text to a fixed-length vector. It's trained with **contrastive learning**:
- **Positive pairs** (question ↔ answer passage, paraphrase ↔ paraphrase) are pulled together.
- **In-batch negatives** (other passages in the same batch) are pushed apart.
- **Pooling:** the per-token outputs are combined into one vector, either by taking the special `[CLS]` token's output (BGE) or by averaging all token outputs (mean pooling).

| | `text-embedding-3-small` (current) | `BAAI/bge-small-en-v1.5` (local option) |
|---|---|---|
| Where it runs | OpenAI API | Your laptop (CPU/MPS) via sentence-transformers |
| Dimensions | 1536 (can be reduced with `dimensions=`, Matryoshka) | 384 |
| Max input | 8,191 tokens | 512 tokens (**silently truncates**) |
| Query/doc symmetry | Symmetric (no prefix) | **Asymmetric:** query instruction prefix |
| Normalized output | Yes | Yes, with `normalize_embeddings=True` |
| Cost | $0.02 per 1M tokens (our corpus ≈ $0.013) | Free; about 38 s to embed the corpus on a laptop |
| Data leaves your machine | Yes | No |
| Scores on our data | real hits 0.65–0.75, off-corpus 0.50 | real hits 0.72–0.83, off-corpus 0.69 |

**sentence-transformers** is the Hugging Face library that wraps a Transformer, its tokenizer and the pooling step into `model.encode(texts)`. It downloads models from the Hugging Face Hub into `~/.cache/huggingface` on first use.

**How to choose (interview answer):** start from the retrieval tab of the **MTEB leaderboard**, then **evaluate on your own data** (Phase 4). Also weigh dimensions (storage and speed), max input tokens, language coverage, latency, price, and whether data may leave your network.

### 📘 Deep dive: Chroma

**What it is:** an open-source vector database. It can run embedded in your Python process (what we use), as a client/server, or as a hosted service. What it provides:
1. **Storage** for records `{id, embedding, document, metadata}`.
2. **ANN search** over the embeddings (HNSW, next deep dive).
3. **Filtering:** `where=` on metadata (`{"ticker": "AMZN"}`, `{"fiscal_year": {"$gte": 2025}}`, `$and`/`$or`) and `where_document=` on text (`{"$contains": "iPhone"}`).

**Our usage** (`common/store.py`):

| Call | What it does |
|---|---|
| `chromadb.PersistentClient(path="data/chroma")` | Embedded mode, saved to disk. No server needed |
| `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})` | A **collection** is like a table: one embedding space, one index |
| `col.add(ids, embeddings, documents, metadatas)` | Insert. We pass **our own** vectors |
| `col.query(query_embeddings=[q], n_results=k, where=...)` | ANN search. Returns ids, documents, metadatas, **distances** |
| `col.get(ids=[...])` | Fetch by ID (no similarity search) |
| `client.delete_collection(name)` | Drop the collection (our `reset=True` path) |

**What's on disk.** This is from inspecting your `data/chroma/`, chromadb 1.5.9:
```
data/chroma/
├── chroma.sqlite3                          80 MB   system database (SQLite)
│   ├── collections, segments               which collections exist and their config
│   ├── embeddings + embedding_metadata     ids + metadata (4,710 records × 6 fields, both collections)
│   ├── embedding_fulltext_search*          SQLite FTS5 full-text index over documents → powers where_document
│   └── embeddings_queue                    write-ahead log: every add() lands here first (vectors as BLOBs)
└── <segment-uuid>/                          one folder per collection's VECTOR segment (the HNSW index)
    ├── header.bin                           index parameters
    ├── data_level0.bin        12.5 MB       layer-0 graph: each node's vector + its neighbour list
    ├── link_lists.bin                       neighbour lists for the upper layers
    ├── length.bin
    └── index_metadata.pickle                id ↔ internal label mapping
```

Each collection has **two segments**:
- A **metadata segment** (SQLite): documents, metadata, full-text index.
- A **vector segment** (HNSW files): the vectors and the graph.

A query combines the two: filter via SQLite, then search the vectors via HNSW.

**Chroma internals that matter:**
- **Writes are batched.** Everything goes into `embeddings_queue` first, and is flushed into the HNSW files every `sync_threshold = 1000` records. Your `data_level0.bin` is exactly 2,000 × 6,284 bytes: **2,000 of the 2,344 vectors are in the persisted index. The last ~344 added (the TSLA FY2025 chunks, for example) are still in the queue.** (6,284 bytes = a 1,536-float vector of 6,144 bytes + 32 neighbour IDs + overhead.) Queries still see all 2,344, because Chroma replays the queue into memory on load. Verified: queued vectors come back as their own top-1. Real databases work the same way: write-ahead log first, index flush later.
- **The default embedding function trap.** The collection config shows `embedding_function: default`. When you don't pass one, Chroma attaches its built-in model (all-MiniLM-L6-v2, 384-d). We always pass `embeddings=` and `query_embeddings=`, so it's never used. But if you called `col.query(query_texts=["..."])`, Chroma would embed the text with MiniLM (384-d) and search an index of 1536-d OpenAI vectors. That fails with a dimension error, or with the bge index (also 384-d) it **silently returns meaningless results**. Always query with vectors from the same model as the index.
- **Orphan files.** `delete_collection` removed the collections from SQLite but left 2 old segment folders on disk (about 24 MB). Deleting all of `data/chroma/` and re-running `03_index` cleans them up.
- **Scale:** embedded Chroma is ideal for prototypes and up to about a few million vectors on one machine. Beyond that, or for multi-tenant production, you'd look at pgvector (vectors inside Postgres, with SQL joins and transactions), Qdrant, Weaviate, Milvus, or a managed service (Phase 7).

### 📘 Deep dive: HNSW (Hierarchical Navigable Small World)

**The problem:** exact search compares the query with every vector, O(N). Phase 0 measured 142 ms at 100k vectors and 6.7 s at 300k. HNSW finds *almost* the same top-k while touching only a few hundred vectors, roughly O(log N).

**Idea 1: a navigable proximity graph.** Link every vector to its nearest neighbours. To search, start somewhere and **greedily hop** to whichever neighbour is closer to the query, until no neighbour is closer. It's like navigating a city by always taking the street that heads toward your destination.

**Idea 2: layers, like a skip list.** A single graph needs many short hops to cross the space, and greedy search can stall. So HNSW adds sparse "express" layers:

```
Layer 2:  A ─────────────────────── F                     ~9 nodes here     long jumps
          │                          │
Layer 1:  A ──────── C ──────────── F ──────── H           ~146 nodes        medium jumps
          │          │               │          │
Layer 0:  A ── B ── C ── D ── E ── F ── G ── H ── I ── J   all 2,344 nodes   local links
```
- Every vector lives in layer 0.
- Each vector is promoted to the next layer up with probability about 1/M. With **M = 16**, about 1/16 of the nodes reach layer 1 and about 1/256 reach layer 2.
- For your 2,344 vectors that gives roughly 3 layers: ~146 nodes in layer 1 and ~9 in layer 2.

**Search:**
1. Enter at the top layer's entry point and greedily hop toward the query.
2. When no neighbour is closer, **drop one layer** and continue from the same node.
3. At layer 0, run a *beam* search that keeps the best **`ef_search`** candidates, not just one.
4. Return the best k of them.

**Insert:**
1. Draw a random top level for the new node.
2. Search down to that level.
3. At each layer, connect the node to up to M neighbours, chosen with a heuristic that prefers **diverse directions** over simply the closest. Diversity keeps the graph navigable. Layer 0 allows 2M links.

**The knobs** (your collection's actual values):

| Parameter | Chroma name | Yours | Effect of increasing |
|---|---|---|---|
| M | `max_neighbors` | 16 | Better recall, more memory, slower inserts |
| ef_construction | `ef_construction` | 100 | Better graph quality, slower build |
| ef_search | `ef_search` | 100 | **Better recall, slower queries.** This is the knob you tune at query time |

With `ef_search = 100` over only 2,344 vectors, each search looks at a large share of the collection, so recall is effectively 100%.

**Trade-offs:**
- ✅ Best-in-class recall/latency (typically 95–99% recall@10 at millisecond latency on millions of vectors).
- ✅ No training step, and it supports incremental inserts.
- ❌ Memory-heavy: vectors plus links must be in RAM (10M × 1536 dims ≈ 61 GB before the links). The fixes are quantization, or IVF-PQ / DiskANN.
- ❌ Deletes leave tombstones, and the graph degrades until it's rebuilt.
- ❌ **Filtered search:** a very selective filter ("only AMZN FY2025", 5% of the data) removes most of the graph, and greedy search can get stuck. Databases handle this by filtering during the graph walk, or by switching to brute force for small filtered sets. That matters in Phase 2.

**Other index types:**

| Index | Idea | Use when |
|---|---|---|
| Flat | Brute force, exact | Fewer than ~100k vectors (like ours: about 1 ms) |
| **HNSW** | Layered graph | The default, if you have the RAM |
| IVF | k-means clusters; search the `nprobe` nearest clusters | Less memory; a training step is acceptable |
| IVF-PQ | IVF plus vectors compressed to about 64 bytes | Billions of vectors |
| DiskANN | Graph stored on SSD | Very large scale on one machine |

**How to measure it:** recall@k = |HNSW top-k ∩ exact top-k| / k over a set of queries. Raise `ef_search` until recall reaches about 98% within your latency budget.

**At our scale** brute force takes about 1 ms, so HNSW doesn't help yet. It pays off past roughly 100k vectors. Saying that in an interview shows you understand the trade-off rather than following defaults.

### 📘 Deep dive: cosine distance vs similarity

- **Cosine similarity** = cos θ, in [−1, 1]. Higher means more similar.
- **Cosine distance** = 1 − cos θ, in [0, 2]. Lower means more similar. Chroma returns this.
- `store.search()` converts it back: `score = 1 − distance`.
- **Chroma's `hnsw:space` options:**
  - `cosine`: distance = 1 − cos.
  - `l2`: squared Euclidean. **This is Chroma's default.**
  - `ip`: distance = 1 − dot product.
- For unit vectors, all three rank results identically (Phase 0). We set `cosine` explicitly so the scores are interpretable.
- **Gotcha:** forget `hnsw:space` and you get L2, and your "scores" become squared distances where lower is better. Code that sorts by "score" descending then silently returns the **worst** matches first.

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
`--chunker/--size` must match an index you've built. If the collection is empty, the script stops with a clear message (`:89`).

`ask()` (`04_ask.py:48`) runs the stages below. Each is wrapped in a **trace span** (`with tr.span(...)`), which Step 6 explains.

### 4a. Embed the query: `embed_query()` (`llm.py:89`)

```
query_prefix(model)  (llm.py:83)
  BGE en v1.5 → "Represent this sentence for searching relevant passages: " + query
  OpenAI      → query unchanged (symmetric model)
embed([...])[0] → (1536,) vector, SAME model as the index
```
- **Asymmetric models** (BGE, E5) embed questions and passages differently; the instruction goes on the **query side only**. Forgetting it lowers quality without any error.
- **Same model as the index:** a query embedded with another model is in a different vector space, and the search returns nonsense.

### 4b. Retrieve: `search_by_vector()` (`store.py:39`)

```
col.query(query_embeddings=[q], n_results=k, where=None)     # HNSW approximate k-NN
→ ids, documents, metadatas, distances
score = 1 − distance      (Chroma returns cosine DISTANCE)
→ [{id, text, meta, score}, ...] ordered best first
```
`where=` supports metadata filters (e.g. `{"ticker": "AMZN"}`). It isn't used in naive RAG; it's the first fix in Phase 2.

`search(col, query)` (`store.py:35`) is just `embed_query` + `search_by_vector` in one call. `ask()` calls the two separately so each gets its own span. That's how the traces show that embedding the query (an API call) costs about 60× more time than the HNSW search itself.

### 4c. Build the prompt: `format_context()` (`04_ask.py:40`)

```
system: SYSTEM (04_ask.py:21)
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

### 4d. Generate: `chat_completion()` (`llm.py:93`)

gpt-4o-mini at **temperature 0**, so the same input gives a repeatable answer. That matters for debugging and evaluation. `ask()` uses `chat_completion()`, which returns the **full** response, instead of `chat()` (`llm.py:98`, text only), because the response's `usage` field (`prompt_tokens`, `completion_tokens`) is what the trace uses to compute cost.

**Cost per question:** about 1,700 prompt tokens with k=5 and 350-token chunks. It grows roughly as k × chunk size.

### 📘 Deep dive: the Chat Completions API, and prompting for grounded answers

**The API.** `client.chat.completions.create(model, messages, temperature)`:
- **`messages`** is a list of `{role, content}`:
  - `system`: the rules (our `SYSTEM`).
  - `user`: the context plus the question.
  - `assistant`: previous model turns, if any.
- The model is **stateless**. Everything it knows about this request comes from `messages`, which is exactly why RAG works by putting retrieved text into the prompt.
- **`temperature`** controls sampling randomness. 0 means (nearly) always take the most likely token, so answers are repeatable, which is what you want for factual Q&A and evaluation. Higher values mean more varied output.
- **Tokens and cost:** you pay for input tokens (≈1,700 here) plus output tokens. Input size grows with k × chunk size. gpt-4o-mini is cheap and fast; bigger models read long contexts more reliably.
- **The context window** (128k for gpt-4o-mini) is a hard limit, but quality drops well before it. Models attend worse to the middle of long contexts ("lost in the middle"), so more chunks isn't automatically better.

**Why each prompt rule exists:**

| Rule | Guards against |
|---|---|
| "Use ONLY the numbered context passages" | The model answering from its training data (possibly outdated, or not from the filing) |
| "Cite every claim with [n]" | Unverifiable answers. Citations let the user, and your eval, check the source |
| "If not in the passages, say I don't know" | **Hallucination under missing context.** The retriever *always* returns k chunks, even for questions the corpus can't answer (Google: 0.50) |
| "State units" | 10-K tables are in millions. "$416,161" without units is wrong by 10⁶ |
| Source header per passage | Chunks say "we", not "Tesla". Without the header the model can't attribute facts |

**Limits of prompt-only grounding:**
- Citations are **claims, not proof**. The model can cite [3] for something [3] doesn't say.
- Refusal is **probabilistic**. A weaker model or a tempting partial match can still produce a confident wrong answer.
- **Step 5 adds the first code-level check**: output guardrails that verify citations and numbers. Phase 4 measures faithfulness properly, and Phases 6–7 harden it further.

### 4e. Report (`04_ask.py:93-116`)

```
answer text
for each hit: " *[n] score  TICKER FY item  (chunk_id)"   * = cited (from the guardrail report's "cited" list)
guardrails: ✅ passed | refusal | ⚠ invalid citations / no citations / ungrounded numbers
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

## Step 5: Output guardrails (`common/guardrails.py` + `05_guardrails_demo.py`)

**Goal:** check every generated answer *before* the user sees it, cheaply and deterministically.

```bash
uv run python -m phase1_naive_rag.05_guardrails_demo      # 10 hand-written cases, no API calls
```
In `04_ask.py` the checks run on every answer inside the `guardrails` span (`:69-71`), and the result is printed under each answer.

### 5a. The checks: `check_answer(answer, passages)` (`guardrails.py:103`)

```
report.refused            = is_refusal(answer)                      "I don't know based on the provided filings"
report.cited              = sorted {n for "[n]" in answer}
report.invalid_citations  = [n for n in cited if n ∉ 1..k]         citing [7] when only 5 passages existed
report.uncited_answer     = no citations AND not a refusal          claims with no evidence
report.ungrounded_numbers = numbers(answer) − numbers(cited passages)
report.passed             = none of the three problems above
```

**`check_citations()`** (`:45`) catches two failures:
- **Invalid indices:** the model cites a passage that doesn't exist.
- **Uncited answers:** the model ignored the "cite every claim" rule.

**`check_numeric_grounding()`** (`:54`) is the finance-specific check. A hallucinated revenue figure is the most damaging error a filings assistant can make. It works in three steps:
1. **`extract_numbers()`** (`:31`) finds numbers with `NUM_RE` (`:16`). The pattern allows `$`, thousands commas, decimals and `%`, and skips numbers glued to letters or hyphens (`FY2025`, `10-K`, `E-4471`).
2. **`_norm()`** (`:23`) puts them in a comparable form: `$416,161` → `416161`, `6.40%` → `6.4`.
3. **Compare** the answer's numbers with the **cited** passages' numbers. If the answer cited nothing, it compares with all retrieved passages.

**One asymmetry matters:** on the *answer* side, years (`2025`) and bare single digits are skipped as noise. On the *passage* side **every** number is kept.
- **Why it matters:** the table cell `| 6 |` is the evidence for an answer's "grew 6%" (the parser dropped the separate `%` cell).
- **The bug that taught this:** the first version skipped single digits on both sides, and flagged a correct "6%" answer as ungrounded. The trace kept a record of it.

**Later improvement:** Phase 3 adds `check_answer(..., tolerant=True)` (`explain_derived()`, `guardrails.py:61`), which accepts unit conversions and growth rates computed from numbers on one table row. It fixes the two false positives below (8/10), and a tamper test confirms it still catches wrong numbers. Strict mode stays the default here.

**Policy: warn, don't block.** `04_ask` prints `⚠` warnings and still shows the answer. Blocking needs high **precision**, and 5b shows this check doesn't have it.

### 5b. Testing the guardrail itself: `05_guardrails_demo.py`

A guardrail is a **classifier** ("is this answer safe to show?"), so it has false positives and false negatives like any other. The demo runs 10 hand-written answers against Apple's real FY2025 net-sales passage:

| Case | Expected | Got | |
|---|---|---|---|
| Grounded answer `$416,161 million [1]` | pass | pass | ✅ |
| Hallucinated `$420,500 million [1]` | flag | flag `['420500']` | ✅ |
| Cites `[7]` with 2 passages | flag | flag | ✅ |
| No citations | flag | flag | ✅ |
| Right number, **wrong passage** cited `[2]` | flag | flag `['416161']` | ✅ |
| Refusal | pass | pass | ✅ |
| Correctly **derived** "grew 6.4%" | pass | **flag** | ❌ false positive |
| **Unit conversion** "$416.2 billion" | pass | **flag** | ❌ false positive |
| Real number, **wrong label** ("China ... $416,161M") | flag | **pass** | ❌ false negative |
| Real number, **wrong year** ($391,035M is FY2024) | flag | **pass** | ❌ false negative |

**Result: 6/10.** The four misses show exactly what string matching can't do: verify **arithmetic**, **unit conversions**, or whether a number is attached to the **right label and year**. That needs a *semantic* check, such as an LLM-as-judge or an NLI faithfulness model (Phase 4), at the cost of an extra model call per answer.

### 📘 Deep dive: guardrails

**What they are.** Guardrails are checks around an LLM system that enforce policy at runtime, independently of the model's instructions. Prompt rules ("cite everything") are *requests*; guardrails are *enforcement*.

**Where they sit** (the interview framework):

| Stage | Examples | In this project |
|---|---|---|
| **Input** | Scope/topic classifier, prompt-injection detection, PII detection, rate limits, auth | Phase 3 (scope, "investment advice" detection), Phase 7 (PII, rate limits) |
| **Retrieval** | Per-user access filters (metadata `where=`), treating retrieved text as untrusted, a relevance floor | Phase 2 (relevance floor), Phase 5 (indirect prompt injection) |
| **Output** | Citation validity, groundedness/faithfulness, numeric verification, moderation, PII redaction, format validation | **Phase 1: citations + numeric grounding.** Phase 4: faithfulness judge. Phase 7: moderation, PII |
| **Agent/tool** | Max steps, tool allowlist, budget, loop detection, human approval for side effects | Phase 6 |

**Kinds of checks, cheapest first:**
1. **Rules and regex** (ours): microseconds, deterministic and explainable, but shallow. Our check runs in about 0.4 ms (from the traces).
2. **Small classifiers:** moderation endpoints, prompt-injection classifiers, NLI models for entailment. Milliseconds, and they need calibration.
3. **LLM-as-judge:** a second model call ("is every claim supported by the passages?"). The most semantic option, but it adds latency and cost, and it can be wrong too.

**Actions a failed check can take:**
- **Block:** refuse or return a fallback.
- **Repair:** regenerate with feedback, or strip the unsupported sentence.
- **Warn:** show the answer with a flag (ours).
- **Log only:** a shadow mode, used to measure a new guardrail before enforcing it.

Choose the action by the check's **precision**. Blocking with a noisy check destroys good answers, and users learn to ignore constant warnings.

**Frameworks** (Phase 7):
- **NeMo Guardrails:** dialogue rails written in the Colang language.
- **Guardrails AI:** validators on structured outputs.
- **Llama Guard:** a safety classifier model.
- **Moderation endpoints:** e.g. OpenAI's.

They package these patterns. The logic you write for your domain, like numeric grounding for financial filings, is usually what matters most.

**Interview angle:** *"Guardrails at input, retrieval and output. Cheap deterministic checks first, semantic checks where it pays off. Every guardrail is itself a classifier with false positives and false negatives, so I test it with labeled cases and roll it out in log-only mode before it blocks anything."*

---

## Step 6: Observability, traces and metrics (`common/trace.py` + `06_traces.py`)

**Goal:** for any question, be able to see exactly what happened: which chunks, what scores, how long each stage took, how many tokens, what it cost, and whether the guardrails passed. And from all questions together: how the system is doing overall.

```bash
uv run python -m phase1_naive_rag.06_traces              # aggregate metrics + timelines of the last 5 questions
uv run python -m phase1_naive_rag.06_traces --last 1     # timeline of the most recent question
uv run python -m phase1_naive_rag.06_traces --json 1     # the raw trace record
```

### 6a. The tracer: `common/trace.py`

```python
tr = Trace("ask", question=..., k=5, collection=..., embed_model=..., chat_model=...)   # trace.py:33
with tr.span("vector_search", k=5) as sp:        # trace.py:43: times the block
    hits = search_by_vector(...)
    sp["hits"] = [{"id": ..., "score": ...}]     # record results INTO the span
tr.set(answer=..., refused=..., guardrails_passed=...)
tr.end()                                         # trace.py:59: appends one JSON line to data/traces/YYYY-MM-DD.jsonl
```
- **`span()`** is a context manager. It records `start_ms` (relative to the trace start) and `duration_ms`, and yields the span's `attrs` dict so the caller can attach data.
- **Exceptions:** if the block raises, the span records `error` and re-raises.
- **`ask()` calls `tr.end()` in a `finally`** (`04_ask.py:76`), so **failed requests are traced too**. Those are the ones you most need to see.
- **`cost_usd()`** (`trace.py:26`) turns token counts into dollars using `PRICES` (`:20`), a hand-maintained table of list prices. Unknown or local models return `None`.
- **`load_traces()`** (`:77`) reads all the JSONL files back.

One trace record (real, from your run):
```json
{"trace_id": "1ebc18cb9b60", "name": "ask", "started_at": "2026-09-26T17:22:08.552+00:00", "duration_ms": 1167.5,
 "attrs": {"question": "By what percentage did Apple's total net sales grow in fiscal 2025?", "k": 5,
           "collection": "10k_text-embedding-3-small_recursive350", "embed_model": "text-embedding-3-small",
           "chat_model": "gpt-4o-mini", "answer": "Apple's total net sales grew by 6% ... [3].",
           "refused": false, "guardrails_passed": true},
 "spans": [{"name": "embed_query",   "start_ms": 0.0,  "duration_ms": 0.5,    "attrs": {}},
           {"name": "vector_search", "start_ms": 0.6,  "duration_ms": 3.5,    "attrs": {"k": 5, "hits": [{"id": "AAPL_FY2025_Item8_049", "score": 0.6714}, ...]}},
           {"name": "generate",      "start_ms": 4.0,  "duration_ms": 1163.0, "attrs": {"model": "gpt-4o-mini", "input_tokens": 1786, "output_tokens": 38, "cost_usd": 0.00029}},
           {"name": "guardrails",    "start_ms": 1167.0, "duration_ms": 0.4,  "attrs": {"refused": false, "cited": [3], "invalid_citations": [], "uncited_answer": false, "ungrounded_numbers": [], "passed": true}}]}
```

### 6b. The viewer: `06_traces.py`

**`summary()`** (`:47`) aggregates metrics across all traces. Real output after 6 questions:
```
== 6 requests ==
latency   p50=1247 ms  p95=3226 ms  max=3526 ms
  embed_query    mean=  392.9 ms   share of total=22.9%
  vector_search  mean=    6.4 ms   share of total= 0.4%
  generate       mean= 1319.6 ms   share of total=76.7%
  guardrails     mean=    0.4 ms   share of total= 0.0%
tokens    mean in=1791  mean out=39
cost      total=$0.0018  per request=$0.00029  -> $29 per 100k requests
quality   refusal rate=33%  guardrail warnings=17%  top-1 retrieval score mean=0.654
```

**`timeline()`** (`:19`) shows one request as a waterfall. This is the same question asked twice:
```
total=3526 ms  [guardrails ⚠]                       ← first run (before the guardrail fix)
  embed_query     2352.5 ms  █████████████████████████████████      ← API call (cache miss)
  vector_search      9.5 ms                                   █
  generate        1163.3 ms                                   ████████████████
  guardrails         0.3 ms                                                   █
                 ungrounded: ['6']                                 ← the false positive, preserved in history

total=1168 ms  [guardrails ✅]                       ← second run
  embed_query        0.5 ms  █                                     ← same question → embedding cache hit
  vector_search      3.5 ms  █
  generate        1163.0 ms  █████████████████████████████████████████████████
```

**What the data tells you:**
- **Generation is 77% of the latency, and vector search is under 1%.** If you need speed, look at the LLM first: streaming (Phase 7), a smaller model, fewer or smaller chunks, response caching. Don't start by tuning HNSW.
- **Query embedding costs a network round-trip on a cache miss** (400 ms on average, with a 2.3 s outlier). A local embedder or a query-embedding cache removes it. The cache hit took 0.5 ms.
- **p95 is about 2.6× p50.** The tail comes from the network and the API, not from our code. p95/p99 is what users feel, so always report percentiles, not just averages.
- **Cost: about $0.0003 per question**, which is about $29 per 100k questions. Input tokens dominate (1,791 in vs 39 out), so k × chunk size drives cost.
- **The refusal rate and top-1 score** are quality signals you can watch over time. A rising refusal rate or a falling top-1 score after a re-index is an early warning (**drift**).

### 📘 Deep dive: observability for LLM/RAG systems

**The three signals** (from general software observability):

| Signal | What | Here |
|---|---|---|
| **Traces** | The path of one request through the system, as a tree of timed **spans** | `data/traces/*.jsonl`, one record per question |
| **Metrics** | Numbers aggregated over time: latency percentiles, rates, costs | `06_traces.py summary()` |
| **Logs** | Discrete events and messages | Our span attributes play this role |

**Why LLM apps need more than normal web tracing.** A request can be "successful" (HTTP 200, fast) and still **wrong**. So an LLM trace also records the *content*:
- the prompt and inputs,
- the retrieved chunk IDs and scores,
- the output, token counts and cost,
- the model and index versions,
- the guardrail and eval results.

That's what makes failures **attributable**: in the trace you can see whether retrieval missed the chunk or generation misused it (the debugging order below).

**Data model:** trace → spans → attributes. It's the same in **OpenTelemetry** (the vendor-neutral standard; spans have a `trace_id`, a `span_id`, a `parent_span_id` and attributes) and in LLM platforms:
- **Langfuse:** open source and self-hostable.
- **LangSmith:** LangChain's platform, native to LangGraph.
- **Arize Phoenix:** open source, built on OpenTelemetry.

Our spans are flat, because the pipeline is a straight line. Agents (Phase 6) produce **nested** spans (agent step → tool call → retrieval → LLM), which is where a real platform pays off.

**What to record (the RAG checklist):**
- **Versions:** embed model, chat model, collection/index name, prompt version.
- **Retrieval:** query (and rewritten queries), top-k IDs and scores, filters used.
- **Generation:** input/output tokens, cost, latency, temperature.
- **Quality:** refusal, guardrail results, and later eval scores and user feedback (👍/👎).

**Production concerns** (Phase 7):
- **PII:** traces contain user questions and answers, so apply redaction, retention limits and access control.
- **Sampling:** keep 100% of errors, sample successes.
- **Cost of tracing itself:** export asynchronously so it doesn't add latency.
- **Online evaluation:** run an LLM judge on a sample of traced requests.
- **Dashboards and alerts:** p95 latency, cost per request, refusal rate, retrieval-score drift.

**Interview angle:** *"I trace every stage with the model and index versions, the retrieved IDs and scores, tokens, cost and guardrail results. With that, a bad answer can be attributed to parsing, retrieval, ranking or generation. On top of the traces I track p50/p95 latency, cost per request, refusal rate and score drift, and sample traces for online evaluation."*

---

## What to rerun after a change

| You changed... | Rerun |
|---|---|
| Tickers or years (`00`) | 00 → 01 → 02 → 03 |
| Parser rules (`01_parse.py`) | 01 → 02 → 03 (the cache re-embeds only changed chunks) |
| Chunk strategy, size or overlap | 02 `--chunker X --size N` → 03 with the same flags → 04 with the same flags |
| Embedding model or provider (`.env`) | 03 (a new collection is created automatically) → 04 |
| Prompt, `k` or chat model | 04 only (compare before and after in `06_traces`) |
| Guardrail rules (`common/guardrails.py`) | 05 (all verdicts should still be as expected) → 04 |
| Nothing: you just want to inspect | `06_traces` (reads existing traces, no API calls) |

## Debugging order when an answer is wrong

0. **Open the trace** (`06_traces --last 1`): retrieved IDs and scores, tokens, and guardrail results. It usually tells you which of the steps below to check.
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

## Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| **10-K** | The annual report every US-listed company files with the SEC |
| **CIK / accession number** | Permanent company ID / unique filing ID on EDGAR |
| **iXBRL** | HTML with machine-readable tags around every financial number |
| **Item** | A regulated section of a 10-K (1A Risk Factors, 7 MD&A, 8 Financials, ...) |
| **Token / BPE** | Sub-word unit that models read / the algorithm that learns the vocabulary |
| **Chunk** | A retrievable piece of a document (here ≤ 350 tokens, inside one Item) |
| **Overlap** | Tokens repeated between neighbouring chunks, so boundary facts survive |
| **Embedding** | A fixed-length vector representing a text's meaning |
| **Asymmetric embedding** | Model that embeds queries differently from documents (BGE query prefix) |
| **Collection** | Chroma's table: one embedding space, one index, one config |
| **Segment** | Chroma storage unit: a metadata segment (SQLite) + a vector segment (HNSW) per collection |
| **HNSW** | Layered proximity-graph index for approximate nearest-neighbour search |
| **M / ef_construction / ef_search** | HNSW links per node / build beam width / query beam width |
| **ANN / recall@k** | Approximate nearest neighbour / share of the true top-k that ANN returns |
| **Cosine distance** | 1 − cosine similarity. Lower is closer |
| **top-k** | The k best-scoring chunks returned by retrieval |
| **Grounding** | Making the LLM answer from supplied context rather than its training data |
| **Refusal path** | An explicit instruction and behaviour for "the answer isn't in the context" |
| **Temperature** | Sampling randomness. 0 means repeatable output |
| **Guardrail** | Runtime check enforcing a policy on inputs, retrieval or outputs, independent of the prompt |
| **Groundedness / faithfulness** | Every claim in the answer is supported by the provided context |
| **False positive / negative (guardrail)** | Flags a good answer / misses a bad one |
| **Trace / span** | One request's record / one timed stage inside it, with attributes |
| **p50 / p95** | Median / 95th-percentile latency (what most users see vs the slow tail) |
| **Drift** | Metrics (refusal rate, retrieval scores) shifting over time, e.g. after a re-index |
| **OpenTelemetry** | Vendor-neutral standard for traces, metrics and logs |

## Self-check before moving on
- [ ] Explain why the Apple footer made a 2023 table rank above the correct 2025 table.
- [ ] Why must chunks never cross an Item boundary? What would break if they did?
- [ ] What happens if you index with bge-small and query with text-embedding-3-small?
- [ ] Why do the MSFT vs AMZN results fail the same way with both embedding models?
- [ ] Walk through the four-step debugging order for "the answer is wrong".
- [ ] `fiscal_year` metadata is the *filing* year. Why wouldn't filtering to FY2025 exclude the 2023 table?
- [ ] Why is the numeric guardrail "warn" and not "block"? Use the 6/10 result.
- [ ] Why must the passage side keep single-digit numbers when the answer side skips them?
- [ ] From the traces, where would you optimize first to cut p95 latency, and why not the vector search?
- [ ] What would you add to a trace so you can tell, a month later, which index and prompt version produced an answer?
