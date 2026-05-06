# eBay UK Sold Listings Analyser

Scrapes eBay UK sold listings for a set of candidate items and ranks them by a flip-worthiness score combining liquidity, price stability, and median price.

## Install

Requires Python 3.11+.

```bash
pip install -r requirements.txt
playwright install chromium
```

## Configure

Edit `candidates.yaml`:

```yaml
candidates:
  - name: "BMW G20 M Sport wheels"
    search_query: "BMW G20 M Sport wheels"
    category: wheels
    bundle_is_canonical: true     # treat "set of 4" / "x4" as the unit; default false
    expects_for_parts: false      # keep "for parts or not working" listings; default false
```

`name` is the label used in output. `search_query` is the literal eBay search string.
`bundle_is_canonical` is essential for wheels and tyres where a set of 4 is the natural unit.

## Run

```bash
# Full run (all candidates, up to 10 pages each)
python main.py

# Smoke test on the first candidate (1 page, prints raw rows)
python main.py --smoke

# Just one candidate by name
python main.py --only "BMW G20 M Sport wheels"

# Batch over multiple days: first 10 today, next 10 tomorrow
python main.py --limit 10
```

Add `--headless` to run without a visible browser. Default is headful, which is harder for eBay to flag.

The first run creates `ebay_state.json` (cookies + local storage). Subsequent runs reuse it so eBay treats you as a returning visitor.

## Output

- `raw_listings.csv` — every individual sold listing kept (one row per listing).
- `results.csv` — one row per candidate, sorted by `score` descending.

`results.csv` columns:

| column | meaning |
|---|---|
| `sold_count_90d` | listings sold in the last 90 days |
| `median_price`, `p25_price`, `p75_price` | price quartiles in £ |
| `price_stability` | stdev / median (lower = more predictable) |
| `score` | `log1p(count) * 1/(stability+0.1) * sqrt(median)` |
| `score_safe` | same but median capped at £500 (compare rankings) |
| `pct_new`, `pct_used`, `pct_for_parts`, `pct_unknown` | condition mix |
| `pct_auction` | share that sold via auction vs Buy It Now |
| `top5_titles` | five priciest listing titles (semicolon-joined) — outlier check |

## Anti-blocking

- Playwright + `playwright-stealth`, UA rotation, en-GB locale, Europe/London timezone.
- 2–5s random delay between page loads.
- Persistent storage state (`ebay_state.json`).
- Captcha detection (`Pardon our interruption`, splash captcha) with one retry after a longer pause; then logs a warning and moves on.
- Sequential candidates only — one ban kills the whole run.

If you start seeing zero results across multiple candidates, you've probably been rate-limited. Stop, wait an hour, and try again. The scraper does not implement IP rotation, residential proxies, or login automation — those are out of scope for this tool.

## Files

```
.
├── candidates.yaml      # input: list of search queries
├── scraper.py           # Playwright scraping per candidate
├── analyser.py          # pandas metrics + composite score
├── main.py              # orchestrator + CLI
├── requirements.txt
└── README.md
```
