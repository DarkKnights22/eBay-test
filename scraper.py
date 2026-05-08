"""eBay UK sold listings scraper.

Used by main.py for the full run, or directly for smoke testing a single query:

    python scraper.py --query "BMW G20 M Sport wheels" --max-pages 1
    python scraper.py --query "BMW G20 M Sport wheels" --max-pages 1 --bundle-canonical
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote_plus

from playwright.sync_api import (
    BrowserContext,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

try:
    from playwright_stealth import stealth_sync
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

logger = logging.getLogger("scraper")

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
]

BASE_URL = "https://www.ebay.co.uk/sch/i.html"
STORAGE_STATE_PATH = Path("ebay_state.json")
MAX_PAGES_DEFAULT = 10
DAYS_WINDOW = 90

# Bundle/multi-item heuristics. Conservative: only matches obvious markers.
BUNDLE_RE = re.compile(
    r"\b(?:x\s?[2-9]|set\s+of\s+\d+|pair(?:\s+of)?|joblot|job\s+lot|bundle)\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"(?:£|GBP\s*)\s*([\d,]+(?:\.\d{1,2})?)")
SOLD_DATE_ABS_RE = re.compile(r"Sold\s+(\d{1,2}\s+\w{3}\s+\d{4})", re.IGNORECASE)
SOLD_DATE_REL_RE = re.compile(
    r"Sold\s+(\d+)\s+(day|days|hour|hours|minute|minutes)\s+ago",
    re.IGNORECASE,
)
BIDS_RE = re.compile(r"\b\d+\s+bids?\b", re.IGNORECASE)


@dataclass
class Listing:
    candidate_name: str
    title: str
    price_gbp: float
    sold_date: Optional[str]  # ISO date string or None
    condition: str
    is_auction: bool
    is_bundle: bool
    url: str


# --- Pure parsing helpers (no Playwright) -----------------------------------

def parse_price(text: str) -> Optional[float]:
    """Extract the lowest GBP price from a string. Returns None if no £/GBP value found."""
    if not text:
        return None
    matches = PRICE_RE.findall(text)
    if not matches:
        return None
    nums = [float(m.replace(",", "")) for m in matches]
    return min(nums)


def has_gbp_marker(text: str) -> bool:
    return bool(text) and bool(re.search(r"£|GBP", text, re.IGNORECASE))


def parse_sold_date(text: str, today: Optional[datetime] = None) -> Optional[datetime]:
    """Parse 'Sold DD Mmm YYYY' or 'Sold N days ago' from a string."""
    if not text:
        return None
    today = today or datetime.now()
    m = SOLD_DATE_ABS_RE.search(text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %b %Y")
        except ValueError:
            return None
    m = SOLD_DATE_REL_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("day"):
            return today - timedelta(days=n)
        if unit.startswith("hour"):
            return today - timedelta(hours=n)
        if unit.startswith("minute"):
            return today - timedelta(minutes=n)
    return None


def detect_bundle(title: str) -> bool:
    return bool(BUNDLE_RE.search(title or ""))


def detect_captcha(html: str) -> bool:
    if not html:
        return False
    h = html.lower()
    return (
        "pardon our interruption" in h
        or "are you a human" in h
        or "/splashui/captcha" in h
    )


def parse_card_fields(
    *,
    title: str,
    price_text: str,
    meta_text: str,
    condition: str,
    url: str,
    candidate_name: str,
    today: datetime,
    cutoff: datetime,
    bundle_is_canonical: bool,
    expects_for_parts: bool,
) -> tuple[Optional[Listing], str]:
    """Build a Listing from already-extracted card strings.

    Returns (Listing, "kept") on success, or (None, reason) where reason is one of:
    no_title, no_price, foreign_currency, old, bundle, for_parts.
    """
    if not title or "Shop on eBay" in title:
        return None, "no_title"

    price = parse_price(price_text)
    if price is None:
        if price_text and not has_gbp_marker(price_text):
            return None, "foreign_currency"
        return None, "no_price"

    sold_dt = parse_sold_date(meta_text, today)
    if sold_dt is not None and sold_dt < cutoff:
        return None, "old"

    is_auction = bool(BIDS_RE.search(meta_text or ""))
    is_bundle = detect_bundle(title)

    if is_bundle and not bundle_is_canonical:
        return None, "bundle"

    cond_lower = (condition or "").lower()
    if "for parts" in cond_lower and not expects_for_parts:
        return None, "for_parts"

    listing = Listing(
        candidate_name=candidate_name,
        title=title.strip(),
        price_gbp=price,
        sold_date=sold_dt.date().isoformat() if sold_dt else None,
        condition=condition.strip() if condition else "unknown",
        is_auction=is_auction,
        is_bundle=is_bundle,
        url=url,
    )
    return listing, "kept"


# --- Browser orchestration --------------------------------------------------

@contextmanager
def ebay_browser(
    headless: bool = False,
    storage_state_path: Path = STORAGE_STATE_PATH,
) -> Iterator[BrowserContext]:
    """Yield a Playwright BrowserContext with persistent state and stealth tweaks.

    Storage state is loaded from `storage_state_path` if it exists, and saved on exit.
    This makes eBay treat us as a returning visitor across runs.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx_kwargs = {
            "user_agent": random.choice(UA_LIST),
            "locale": "en-GB",
            "timezone_id": "Europe/London",
            "viewport": {"width": 1366, "height": 900},
        }
        if storage_state_path.exists():
            ctx_kwargs["storage_state"] = str(storage_state_path)
            logger.info(f"loaded storage state from {storage_state_path}")
        # Escape hatch for sandboxed environments that MITM TLS. Off by default.
        if os.environ.get("EBAY_IGNORE_HTTPS_ERRORS") == "1":
            ctx_kwargs["ignore_https_errors"] = True
            logger.warning("ignore_https_errors=True (EBAY_IGNORE_HTTPS_ERRORS=1)")
        context = browser.new_context(**ctx_kwargs)
        try:
            yield context
        finally:
            try:
                context.storage_state(path=str(storage_state_path))
                logger.info(f"saved storage state to {storage_state_path}")
            except Exception as e:
                logger.warning(f"could not save storage state: {e}")
            context.close()
            browser.close()


def _safe_inner_text(locator, timeout: int = 1000) -> str:
    try:
        if locator.count() == 0:
            return ""
        return locator.first.inner_text(timeout=timeout)
    except Exception:
        return ""


def _safe_attr(locator, attr: str, timeout: int = 1000) -> str:
    try:
        if locator.count() == 0:
            return ""
        return locator.first.get_attribute(attr, timeout=timeout) or ""
    except Exception:
        return ""


def scrape_candidate(
    context: BrowserContext,
    candidate: dict,
    max_pages: int = MAX_PAGES_DEFAULT,
) -> list[Listing]:
    """Scrape eBay UK sold listings for one candidate. Returns kept Listings."""
    name = candidate["name"]
    query = candidate["search_query"]
    bundle_canonical = bool(candidate.get("bundle_is_canonical", False))
    expects_for_parts = bool(candidate.get("expects_for_parts", False))

    today = datetime.now()
    cutoff = today - timedelta(days=DAYS_WINDOW)
    listings: list[Listing] = []

    page = context.new_page()
    if HAS_STEALTH:
        try:
            stealth_sync(page)
        except Exception as e:
            logger.debug(f"stealth_sync failed: {e}")

    try:
        for pgn in range(1, max_pages + 1):
            url = (
                f"{BASE_URL}?_nkw={quote_plus(query)}"
                f"&LH_Sold=1&LH_Complete=1&_ipg=240&_pgn={pgn}"
            )
            logger.info(f"  page {pgn}: {url}")

            ok = False
            for attempt in range(2):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(1500)  # let lazy bits render
                    if detect_captcha(page.content()):
                        logger.warning(f"  captcha detected (attempt {attempt + 1})")
                        time.sleep(random.uniform(20, 40))
                        continue
                    ok = True
                    break
                except PWTimeoutError as e:
                    logger.warning(f"  timeout: {e}")
                    time.sleep(5)
                except Exception as e:
                    logger.warning(f"  page load failed: {e}")
                    time.sleep(5)

            if not ok:
                logger.warning(f"  giving up on page {pgn}")
                break

            try:
                page.wait_for_selector("ul.srp-results, li.s-card", timeout=10000)
            except PWTimeoutError:
                logger.warning(f"  no results container on page {pgn}")
                break

            cards = page.locator("li.s-card").all()
            if not cards:
                logger.info(f"  page {pgn}: no cards, stopping")
                break

            kept = 0
            old_skipped = 0
            other_skipped: dict[str, int] = {}

            for card in cards:
                # Skip sponsored cards
                try:
                    if card.locator("text=/Sponsored/i").count() > 0:
                        other_skipped["sponsored"] = other_skipped.get("sponsored", 0) + 1
                        continue
                except Exception:
                    pass

                title = _safe_inner_text(card.locator(".s-card__title .su-styled-text"))
                if not title:
                    continue
                price_text = _safe_inner_text(card.locator(".s-card__price"))
                meta_text = _safe_inner_text(card)  # whole card text — has sold date + bid count
                condition = _safe_inner_text(card.locator(".s-card__subtitle"))
                href = _safe_attr(card.locator("a.s-card__link"), "href")

                listing, status = parse_card_fields(
                    title=title,
                    price_text=price_text,
                    meta_text=meta_text,
                    condition=condition,
                    url=href,
                    candidate_name=name,
                    today=today,
                    cutoff=cutoff,
                    bundle_is_canonical=bundle_canonical,
                    expects_for_parts=expects_for_parts,
                )
                if listing is not None:
                    listings.append(listing)
                    kept += 1
                elif status == "old":
                    old_skipped += 1
                else:
                    other_skipped[status] = other_skipped.get(status, 0) + 1

            logger.info(
                f"  page {pgn}: kept={kept}, old={old_skipped}, "
                f"other_skipped={dict(other_skipped) or '{}'}"
            )

            if kept == 0 and old_skipped == 0 and not other_skipped:
                logger.info("  empty page, stopping")
                break

            # Past 90d window: page had old listings but nothing kept.
            if old_skipped > 0 and kept == 0:
                logger.info("  page entirely outside 90d window, stopping")
                break

            time.sleep(random.uniform(2, 5))
    finally:
        page.close()

    return listings


# --- CLI smoke test ---------------------------------------------------------

def _smoke_test() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the scraper on a single query.")
    parser.add_argument("--query", required=True, help="search_query string")
    parser.add_argument("--name", default=None, help="candidate name (defaults to --query)")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--bundle-canonical", action="store_true",
                        help="treat 'set of N' / 'x4' as the canonical unit (don't filter out)")
    parser.add_argument("--expects-for-parts", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    candidate = {
        "name": args.name or args.query,
        "search_query": args.query,
        "bundle_is_canonical": args.bundle_canonical,
        "expects_for_parts": args.expects_for_parts,
    }

    with ebay_browser(headless=args.headless) as ctx:
        listings = scrape_candidate(ctx, candidate, max_pages=args.max_pages)

    print(f"\nCaptured {len(listings)} listings.\n")
    for lst in listings[:30]:
        print(json.dumps(asdict(lst), ensure_ascii=False))


if __name__ == "__main__":
    _smoke_test()
