"""Analyser: turn raw eBay sold listings into per-candidate flip-worthiness metrics."""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

# Cap applied to median_price inside the "safe" composite score so high-ticket
# items don't dominate the ranking on price alone.
SAFE_MEDIAN_CAP_GBP = 500.0

# --- Pair / single / aftermarket classification (used by f30_analysis.py) ---

_PAIR_PATTERNS = [
    r"\bpair\b", r"\bset\b", r"\bboth\b", r"\bcomplete\b",
    r"\b2x\b", r"\bx2\b",
    r"left.*right", r"right.*left",
    r"driver.*passenger", r"passenger.*driver",
    r"n\/?s.*o\/?s", r"o\/?s.*n\/?s",
]
_SINGLE_PATTERNS = [
    r"\bsingle\b", r"\bone\b", r"\bleft\b", r"\bright\b",
    r"\bdriver\s*side\b", r"\bpassenger\s*side\b",
    r"\bOSF\b", r"\bNSF\b", r"\bRHS\b", r"\bLHS\b",
    r"\bO\/S\b", r"\bN\/S\b",
    r"\bnearside\b", r"\boffside\b",
]
_AFTERMARKET_PATTERNS = [
    r"\baftermarket\b", r"\breplica\b", r"\blci.?style\b",
    r"\bdepo\b", r"\bchinese\b", r"\bbrand\s+new\b", r"\bnew\b",
]
PAIR_RE = re.compile("|".join(_PAIR_PATTERNS), re.IGNORECASE)
SINGLE_RE = re.compile("|".join(_SINGLE_PATTERNS), re.IGNORECASE)
AFTERMARKET_RE = re.compile("|".join(_AFTERMARKET_PATTERNS), re.IGNORECASE)
HEADLIGHT_MATCH_RE = re.compile(r"headlight|headlamp|head\s+light", re.IGNORECASE)


def classify_pair_single(title: str) -> str:
    """Return 'pair', 'single', or 'unclear'. Pair takes precedence on conflict."""
    if not title:
        return "unclear"
    if PAIR_RE.search(title):
        return "pair"
    if SINGLE_RE.search(title):
        return "single"
    return "unclear"


def is_aftermarket_suspect(title: str) -> bool:
    """True if title contains aftermarket/replica/new markers."""
    return bool(title) and bool(AFTERMARKET_RE.search(title))


def _condition_breakdown(conds: pd.Series) -> dict[str, float]:
    """Return % share of new / used / for_parts / unknown across a candidate's listings."""
    n = len(conds)
    if n == 0:
        return {"pct_new": 0.0, "pct_used": 0.0, "pct_for_parts": 0.0, "pct_unknown": 0.0}
    lower = conds.fillna("").str.lower()
    pct_new = (lower.str.contains("new", na=False)).sum() / n * 100
    pct_for_parts = (lower.str.contains("for parts", na=False)).sum() / n * 100
    pct_unknown = ((lower == "") | (lower == "unknown")).sum() / n * 100
    pct_used = max(0.0, 100.0 - pct_new - pct_for_parts - pct_unknown)
    return {
        "pct_new": round(pct_new, 1),
        "pct_used": round(pct_used, 1),
        "pct_for_parts": round(pct_for_parts, 1),
        "pct_unknown": round(pct_unknown, 1),
    }


def _composite_score(count: int, stability: float, median: float) -> float:
    """score = log(count) * 1/(stability + 0.1) * sqrt(median).

    log1p so count=0 gives 0 instead of -inf. stability NaN (n<2) treated as 0.
    """
    if count <= 0 or median is None or median != median:  # NaN check
        return 0.0
    s = 0.0 if (stability is None or stability != stability) else float(stability)
    return float(math.log1p(count) * (1.0 / (s + 0.1)) * math.sqrt(median))


def _composite_score_safe(count: int, stability: float, median: float) -> float:
    """Same as _composite_score but median is capped at SAFE_MEDIAN_CAP_GBP."""
    if count <= 0 or median is None or median != median:
        return 0.0
    capped = min(float(median), SAFE_MEDIAN_CAP_GBP)
    s = 0.0 if (stability is None or stability != stability) else float(stability)
    return float(math.log1p(count) * (1.0 / (s + 0.1)) * math.sqrt(capped))


def analyse(raw: pd.DataFrame, candidate_names: Iterable[str]) -> pd.DataFrame:
    """Per-candidate metrics + composite scores. Always emits one row per candidate,
    even if zero listings were captured (so the user can see which queries failed)."""
    rows = []
    for name in candidate_names:
        sub = raw[raw["candidate_name"] == name] if len(raw) else raw.iloc[0:0]
        prices = sub["price_gbp"].astype(float) if "price_gbp" in sub else pd.Series(dtype=float)

        count = int(len(sub))
        median = float(prices.median()) if count else float("nan")
        p25 = float(prices.quantile(0.25)) if count else float("nan")
        p75 = float(prices.quantile(0.75)) if count else float("nan")
        stdev = float(prices.std(ddof=0)) if count >= 2 else float("nan")
        stability = stdev / median if (count >= 2 and median and not np.isnan(median) and median > 0) else float("nan")

        cb = _condition_breakdown(sub["condition"]) if count else _condition_breakdown(pd.Series([], dtype=object))

        top5 = (
            sub.sort_values("price_gbp", ascending=False)
               .head(5)["title"]
               .tolist()
            if count else []
        )

        score = _composite_score(count, stability, median)
        score_safe = _composite_score_safe(count, stability, median)

        auction_pct = float(sub["is_auction"].mean() * 100) if count else 0.0

        rows.append({
            "candidate_name": name,
            "sold_count_90d": count,
            "median_price": round(median, 2) if count else None,
            "p25_price": round(p25, 2) if count else None,
            "p75_price": round(p75, 2) if count else None,
            "price_stability": round(stability, 3) if count >= 2 else None,
            "score": round(score, 3),
            "score_safe": round(score_safe, 3),
            "pct_auction": round(auction_pct, 1),
            **cb,
            "top5_titles": " ; ".join(top5),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
