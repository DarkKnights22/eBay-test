"""Orchestrator: load candidates, scrape sequentially, analyse, write CSVs.

Usage:
    python main.py                                  # full run
    python main.py --limit 5                        # first 5 candidates only
    python main.py --only "BMW G20 M Sport wheels"  # one candidate
    python main.py --smoke                          # first candidate, --max-pages 1, prints raw rows
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from scraper import (
    MAX_PAGES_DEFAULT,
    Listing,
    ebay_browser,
    scrape_candidate,
)
from analyser import analyse

logger = logging.getLogger("main")


def load_candidates(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    cands = data.get("candidates", []) if data else []
    if not cands:
        raise ValueError(f"No candidates found in {path}")
    for i, c in enumerate(cands):
        if "name" not in c or "search_query" not in c:
            raise ValueError(f"Candidate {i} missing name or search_query: {c}")
    return cands


def listings_to_df(listings: list[Listing]) -> pd.DataFrame:
    if not listings:
        return pd.DataFrame(columns=[
            "candidate_name", "title", "price_gbp", "sold_date",
            "condition", "is_auction", "is_bundle", "url",
        ])
    return pd.DataFrame([asdict(l) for l in listings])


def run(
    candidates_path: Path,
    *,
    only: str | None = None,
    limit: int | None = None,
    smoke: bool = False,
    max_pages: int = MAX_PAGES_DEFAULT,
    headless: bool = False,
    raw_csv: Path = Path("raw_listings.csv"),
    results_csv: Path = Path("results.csv"),
) -> None:
    candidates = load_candidates(candidates_path)

    if only:
        candidates = [c for c in candidates if c["name"] == only]
        if not candidates:
            raise SystemExit(f"No candidate named {only!r} in {candidates_path}")
    if smoke:
        candidates = candidates[:1]
        max_pages = 1
    if limit is not None:
        candidates = candidates[:limit]

    logger.info(f"Running {len(candidates)} candidate(s); max_pages={max_pages}")

    all_listings: list[Listing] = []
    with ebay_browser(headless=headless) as ctx:
        for i, cand in enumerate(candidates, 1):
            logger.info(f"[{i}/{len(candidates)}] {cand['name']}")
            try:
                listings = scrape_candidate(ctx, cand, max_pages=max_pages)
            except Exception as e:
                logger.error(f"  scrape failed for {cand['name']!r}: {e}")
                listings = []
            logger.info(f"  -> kept {len(listings)} listings")
            all_listings.extend(listings)

    raw_df = listings_to_df(all_listings)
    raw_df.to_csv(raw_csv, index=False)
    logger.info(f"Wrote {len(raw_df)} rows to {raw_csv}")

    if smoke:
        # Show raw dataframe BEFORE summary stats so we can eyeball parsing quality.
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 80,
        ):
            print("\n=== Raw listings (first 30) ===")
            print(raw_df.head(30).to_string(index=False))

    results_df = analyse(raw_df, candidate_names=[c["name"] for c in candidates])
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Wrote {len(results_df)} rows to {results_csv}")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 60,
    ):
        print("\n=== Results (sorted by score) ===")
        print(results_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="eBay UK sold listings flip-analyser.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.yaml"))
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the candidate with this exact name.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N candidates from the YAML.")
    parser.add_argument("--smoke", action="store_true",
                        help="One candidate, one page, print raw rows before stats.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_DEFAULT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raw-csv", type=Path, default=Path("raw_listings.csv"))
    parser.add_argument("--results-csv", type=Path, default=Path("results.csv"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(
        candidates_path=args.candidates,
        only=args.only,
        limit=args.limit,
        smoke=args.smoke,
        max_pages=args.max_pages,
        headless=args.headless,
        raw_csv=args.raw_csv,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
