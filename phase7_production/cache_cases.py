"""Labeled question pairs for the semantic-cache experiment (03_cache.py).

SAME: a paraphrase with the same correct answer (a cache hit is right).
DIFF: a near-miss with a different answer (a cache hit serves a WRONG answer). Grouped by what changed, because the
entity guard (common.cache.signature) can only catch some kinds: company, year and relative-time swaps.
Written by hand, before running the experiment.
"""
BASE = {
    "aapl_sales": "What was Apple's total net sales in fiscal 2025?",
    "tsla_margin": "What was Tesla's total gross margin in 2025?",
    "msft_emp": "How many employees did Microsoft have at the end of fiscal 2025?",
    "amzn_capex": "How much did Amazon spend on purchases of property and equipment in 2025?",
    "nvda_div": "What were NVIDIA's cash dividends per share in fiscal 2026?",
    "msft_rev": "What was Microsoft's revenue in its latest fiscal year?",
    "aapl_rd": "How much did Apple spend on research and development in fiscal 2024?",
    "tsla_deliv": "How many vehicles did Tesla deliver in 2025?",
}

SAME = [
    ("aapl_sales", "How much did Apple sell in total in FY2025?"),
    ("aapl_sales", "Apple total net sales fiscal 2025"),
    ("aapl_sales", "what were apple's total net sales for fiscal year 2025"),
    ("aapl_sales", "What were Apple's total revenues in fiscal 2025?"),
    ("aapl_sales", "Total net sales of Apple in FY 2025?"),
    ("tsla_margin", "Tesla's total gross margin for 2025?"),
    ("tsla_margin", "What gross margin did Tesla report overall in 2025?"),
    ("tsla_margin", "In 2025, what was the total gross margin at Tesla?"),
    ("msft_emp", "What was Microsoft's headcount at the end of fiscal 2025?"),
    ("msft_emp", "How many people did Microsoft employ as of June 30, 2025?"),
    ("msft_emp", "Microsoft number of employees fiscal 2025"),
    ("amzn_capex", "What was Amazon's capex in 2025?"),
    ("amzn_capex", "Amazon purchases of property and equipment 2025"),
    ("amzn_capex", "How much did Amazon invest in property and equipment purchases during 2025?"),
    ("nvda_div", "What was NVIDIA's dividend per share in fiscal 2026?"),
    ("nvda_div", "NVIDIA cash dividend per share FY2026"),
    ("msft_rev", "What was Microsoft's total revenue in the most recent fiscal year?"),
    ("msft_rev", "How much revenue did Microsoft make in its latest fiscal year?"),
    ("aapl_rd", "What were Apple's R&D expenses in fiscal 2024?"),
    ("aapl_rd", "Apple research and development spending FY2024"),
    ("aapl_rd", "How much did Apple invest in R&D during fiscal year 2024?"),
    ("tsla_deliv", "What were Tesla's vehicle deliveries in 2025?"),
    ("tsla_deliv", "Tesla total deliveries 2025"),
    ("tsla_deliv", "How many cars did Tesla deliver in 2025?"),
]

DIFF = [  # (base, near-miss, what changed)
    ("aapl_sales", "What was Apple's total net sales in fiscal 2024?", "year"),
    ("aapl_sales", "What was Amazon's total net sales in fiscal 2025?", "company"),
    ("aapl_sales", "What was Apple's net income in fiscal 2025?", "metric"),
    ("aapl_sales", "What was Apple's iPhone net sales in fiscal 2025?", "sub-component"),
    ("aapl_sales", "What was Apple's Services net sales in fiscal 2025?", "sub-component"),
    ("aapl_sales", "What was Apple's total net sales growth in fiscal 2025?", "metric"),
    ("tsla_margin", "What was Tesla's automotive gross margin in 2025?", "sub-component"),
    ("tsla_margin", "What was Tesla's total gross margin in 2024?", "year"),
    ("tsla_margin", "What was Tesla's operating margin in 2025?", "metric"),
    ("tsla_margin", "What was NVIDIA's gross margin in fiscal 2025?", "company"),
    ("msft_emp", "How many employees did Microsoft have at the end of fiscal 2026?", "year"),
    ("msft_emp", "How many employees did NVIDIA have at the end of fiscal 2025?", "company"),
    ("msft_emp", "How many Microsoft employees were in the United States at the end of fiscal 2025?", "sub-component"),
    ("amzn_capex", "How much did Amazon receive from sales of property and equipment in 2025?", "metric"),
    ("amzn_capex", "How much did Amazon spend on purchases of property and equipment in 2024?", "year"),
    ("amzn_capex", "How much did Microsoft spend on purchases of property and equipment in fiscal 2025?", "company"),
    ("amzn_capex", "How much did Amazon spend on acquisitions in 2025?", "metric"),
    ("nvda_div", "What were NVIDIA's cash dividends per share in fiscal 2025?", "year"),
    ("nvda_div", "How much did NVIDIA pay in total cash dividends in fiscal 2026?", "metric"),
    ("nvda_div", "What were Apple's cash dividends per share in fiscal 2025?", "company"),
    ("msft_rev", "What was Microsoft's revenue in the prior fiscal year?", "relative time"),
    ("msft_rev", "What was Microsoft's cloud revenue in its latest fiscal year?", "sub-component"),
    ("msft_rev", "What was Microsoft's operating income in its latest fiscal year?", "metric"),
    ("msft_rev", "What was Amazon's revenue in its latest fiscal year?", "company"),
    ("aapl_rd", "How much did Apple spend on research and development in fiscal 2025?", "year"),
    ("aapl_rd", "How much did Apple spend on selling, general and administrative expenses in fiscal 2024?", "metric"),
    ("aapl_rd", "How much did NVIDIA spend on research and development in fiscal 2025?", "company"),
    ("tsla_deliv", "How many vehicles did Tesla produce in 2025?", "metric"),
    ("tsla_deliv", "How many vehicles did Tesla deliver in 2024?", "year"),
    ("tsla_deliv", "How many Model 3 and Model Y vehicles did Tesla deliver in 2025?", "sub-component"),
]
