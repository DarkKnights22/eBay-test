"""F30 LCI headlight deep-dive: re-scrape pair-biased query variants and classify
each listing as pair / single / unclear, plus flag aftermarket suspects. Emits
a comparison table across the 5 query variants and writes f30_lci_analysis.csv.

Usage:
    python f30_analysis.py
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from analyser import (
    HEADLIGHT_MATCH_RE,
    classify_pair_single,
    is_aftermarket_suspect,
)
from scraper import ebay_browser, scrape_candidate

logger = logging.getLogger("f30")

EXISTING_QUERY = "BMW F30 LCI LED headlights"
NEW_QUERY_NAMES = [
    "BMW F30 LCI headlights pair",
    "BMW F30 LCI LED headlights set",
    "BMW F30 F31 LCI headlights both",
    "BMW F30 LCI adaptive LED",
]
RAW_CSV = Path("raw_listings.csv")
OUT_CSV = Path("f30_lci_analysis.csv")


def _load_new_candidates() -> list[dict]:
    """Pull the 4 new F30 LCI candidates out of candidates.yaml by name."""
    with open("candidates.yaml") as f:
        data = yaml.safe_load(f)
    by_name = {c["name"]: c for c in data["candidates"]}
    return [by_name[n] for n in NEW_QUERY_NAMES]


def _scrape_new_queries(candidates: list[dict]) -> pd.DataFrame:
    """Run the existing scraper for the 4 new queries in one browser session."""
    rows: list[dict] = []
    with ebay_browser(headless=False) as ctx:
        for i, cand in enumerate(candidates, 1):
            logger.info(f"[{i}/{len(candidates)}] {cand['name']}")
            try:
                listings = scrape_candidate(ctx, cand, max_pages=10)
            except Exception as e:
                logger.error(f"  scrape failed: {e}")
                listings = []
            logger.info(f"  -> kept {len(listings)} listings")
            rows.extend(asdict(l) for l in listings)
    if not rows:
        return pd.DataFrame(columns=[
            "candidate_name", "title", "price_gbp", "sold_date",
            "condition", "is_auction", "is_bundle", "url",
        ])
    return pd.DataFrame(rows)


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df["title"].apply(classify_pair_single)
    df["aftermarket_suspect"] = df["title"].apply(is_aftermarket_suspect)
    df["headlight_match"] = df["title"].astype(str).str.contains(
        HEADLIGHT_MATCH_RE, regex=True, na=False
    )
    return df


def _per_query_summary(df: pd.DataFrame, query_order: list[str]) -> pd.DataFrame:
    rows = []
    for q in query_order:
        sub = df[df["candidate_name"] == q]
        n = len(sub)
        if n == 0:
            rows.append({
                "query": q, "n": 0, "hit_pct": 0.0,
                "pair_count": 0, "pair_median": None,
                "single_count": 0, "single_median": None,
                "unclear_count": 0, "aftermarket_count": 0,
            })
            continue
        pair = sub[sub["bucket"] == "pair"]
        single = sub[sub["bucket"] == "single"]
        unclear = sub[sub["bucket"] == "unclear"]
        rows.append({
            "query": q,
            "n": n,
            "hit_pct": round(100 * sub["headlight_match"].mean(), 1),
            "pair_count": len(pair),
            "pair_median": round(pair["price_gbp"].median(), 2) if len(pair) else None,
            "single_count": len(single),
            "single_median": round(single["price_gbp"].median(), 2) if len(single) else None,
            "unclear_count": len(unclear),
            "aftermarket_count": int(sub["aftermarket_suspect"].sum()),
        })
    return pd.DataFrame(rows)


def _print_bucket_stats(df: pd.DataFrame, query_order: list[str]) -> None:
    """For each query / bucket combo, print quartiles + 10 example titles."""
    for q in query_order:
        sub = df[df["candidate_name"] == q]
        if len(sub) == 0:
            print(f"\n=== {q} === (no listings)")
            continue
        print(f"\n=== {q} (n={len(sub)}) ===")
        for bucket in ("pair", "single", "unclear"):
            b = sub[sub["bucket"] == bucket]
            if len(b) == 0:
                continue
            prices = b["price_gbp"].astype(float)
            print(
                f"\n  -- {bucket}: n={len(b)}  "
                f"median £{prices.median():.2f}  "
                f"p25 £{prices.quantile(0.25):.2f}  "
                f"p75 £{prices.quantile(0.75):.2f}  "
                f"mean £{prices.mean():.2f} --"
            )
            for _, row in b.head(10).iterrows():
                tag = " [AFT]" if row["aftermarket_suspect"] else ""
                print(f"    £{row['price_gbp']:>7.2f}  {row['title']}{tag}")


def _verdict(df: pd.DataFrame) -> None:
    """Compute the genuine-OEM-pair median across all 5 queries (deduped by URL)."""
    headlight_only = df[df["headlight_match"]]
    pairs = headlight_only[
        (headlight_only["bucket"] == "pair") &
        (~headlight_only["aftermarket_suspect"])
    ]
    pairs_dedup = pairs.drop_duplicates(subset=["url"])

    print("\n" + "=" * 70)
    print("VERDICT — genuine-OEM-likely F30 LCI LED headlight PAIRS")
    print("=" * 70)
    print(f"  After:  headlight_match=True, bucket=pair, aftermarket_suspect=False")
    print(f"  Then:   deduped by URL across all 5 queries")
    print(f"  Sample size: {len(pairs_dedup)} listings")

    if len(pairs_dedup) == 0:
        print("\n  No clean OEM-used pair listings found. Flip thesis NOT supported.")
        return

    p = pairs_dedup["price_gbp"].astype(float)
    print(f"  Median price: £{p.median():.2f}")
    print(f"  p25 / p75:    £{p.quantile(0.25):.2f} / £{p.quantile(0.75):.2f}")
    print(f"  Mean:         £{p.mean():.2f}")
    print(f"  Min / Max:    £{p.min():.2f} / £{p.max():.2f}")

    # Quick viability heuristic: do we have enough volume and a tight enough range
    # for the flip thesis to be testable?
    spread = p.quantile(0.75) - p.quantile(0.25)
    spread_pct = spread / p.median() * 100 if p.median() else 0
    n = len(pairs_dedup)
    print(f"\n  IQR spread:   £{spread:.2f}  ({spread_pct:.0f}% of median)")

    print("\n  Plain-English read:")
    if n < 10:
        print(f"    * Only {n} clean pair listings in 90 days — not enough volume to")
        print(f"      conclude anything. Flip thesis is INCONCLUSIVE.")
    elif spread_pct > 100:
        print(f"    * {n} listings but IQR is {spread_pct:.0f}% of median — prices are")
        print(f"      all over the place (probably mixing standard + adaptive + xenon).")
        print(f"      Flip thesis WEAK without finer query splitting.")
    else:
        print(f"    * {n} clean OEM pair sales in 90 days at ~£{p.median():.0f} median,")
        print(f"      with IQR £{p.quantile(0.25):.0f}–£{p.quantile(0.75):.0f}.")
        print(f"      Flip thesis VIABLE if you can source under p25 (£{p.quantile(0.25):.0f}).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} not found — run main.py first to populate it.")
    raw = pd.read_csv(RAW_CSV)
    existing = raw[raw["candidate_name"] == EXISTING_QUERY].copy()
    logger.info(f"Loaded {len(existing)} existing rows for {EXISTING_QUERY!r}")

    new_cands = _load_new_candidates()
    new_df = _scrape_new_queries(new_cands)
    logger.info(f"Scraped {len(new_df)} listings across {len(new_cands)} new queries")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = _classify(combined)
    combined.to_csv(OUT_CSV, index=False)
    logger.info(f"Wrote {len(combined)} rows to {OUT_CSV}")

    query_order = [EXISTING_QUERY] + NEW_QUERY_NAMES
    summary = _per_query_summary(combined, query_order)

    print("\n" + "=" * 70)
    print("Per-query summary")
    print("=" * 70)
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 50):
        print(summary.to_string(index=False))

    _print_bucket_stats(combined, query_order)
    _verdict(combined)


if __name__ == "__main__":
    main()
