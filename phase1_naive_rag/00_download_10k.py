"""Download recent 10-K filings from SEC EDGAR into data/raw/.

Run: uv run python -m phase1_naive_rag.00_download_10k

Only the primary 10-K document is fetched (no exhibits). A manifest.json records the
metadata (ticker, fiscal year, filing date, URL) that later phases use for filtering.
"""
import json
import time

import httpx

from common.config import DATA_DIR, SEC_USER_AGENT

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
YEARS_PER_COMPANY = 2
RAW_DIR = DATA_DIR / "raw"
DELAY_S = 0.2  # SEC fair-access limit is 10 req/s; stay well under it


def get(http: httpx.Client, url: str) -> httpx.Response:
    time.sleep(DELAY_S)
    resp = http.get(url)
    resp.raise_for_status()
    return resp


def ticker_to_cik(http: httpx.Client) -> dict[str, int]:
    rows = get(http, "https://www.sec.gov/files/company_tickers.json").json().values()
    return {r["ticker"]: r["cik_str"] for r in rows}


def latest_10ks(http: httpx.Client, cik: int, n: int) -> list[dict]:
    recent = get(http, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        if form != "10-K":  # skip amendments (10-K/A)
            continue
        out.append({
            "accession": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "period_end": recent["reportDate"][i],
            "primary_doc": recent["primaryDocument"][i],
        })
        if len(out) == n:
            break
    return out


def main() -> None:
    if not SEC_USER_AGENT:
        raise SystemExit("Set SEC_USER_AGENT='Your Name you@example.com' in .env (SEC requires it)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    with httpx.Client(headers={"User-Agent": SEC_USER_AGENT}, timeout=60, follow_redirects=True) as http:
        ciks = ticker_to_cik(http)
        for ticker in TICKERS:
            cik = ciks[ticker]
            for f in latest_10ks(http, cik, YEARS_PER_COMPANY):
                # Fiscal year = year the period ends (AAPL: Sep, MSFT: Jun, NVDA: Jan -> matches their FY naming)
                fy = int(f["period_end"][:4])
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{f['accession'].replace('-', '')}/{f['primary_doc']}")
                path = RAW_DIR / f"{ticker}_FY{fy}.html"
                if path.exists():
                    print(f"skip  {path.name} (exists)")
                else:
                    path.write_bytes(get(http, url).content)
                    print(f"saved {path.name}  {path.stat().st_size / 1e6:.1f} MB  filed {f['filing_date']}")
                manifest.append({"ticker": ticker, "fiscal_year": fy, "file": path.name, "url": url, **f})

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} filings -> {RAW_DIR}")


if __name__ == "__main__":
    main()
