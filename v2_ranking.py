"""Re-rank all candidates using the singles/pairs/sets classifier and a new
score_v2 formula that rewards liquidity in the *correct* unit type and damps
contaminated queries via hit_pct.

Reads existing raw_listings.csv + f30_lci_analysis.csv, scrapes only the 14
new candidates, classifies all listings, dedupes by item_id, drops accessory
listings, and emits a ranked table to terminal + f30_v2_ranking.csv.

Usage:
    python v2_ranking.py
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analyser import classify_pair_single, is_aftermarket_suspect
from scraper import ebay_browser, scrape_candidate

logger = logging.getLogger("v2")

ITEM_ID_RE = re.compile(r"/itm/(\d+)")

# Listings that match these are accessories — wrong product entirely
# (e.g. headlight bulb adapter, mirror cap stickers, bumper brackets)
ACCESSORY_RE = re.compile(
    r"adapter|holder|bulb\b|module\b|drl\b|wiring|cable|repair kit|"
    r"connector|board\b|sticker|emblem|badge|cap[\s_]*cover\b|"
    r"actuator|bracket\b|protector\b",
    re.IGNORECASE,
)

# 14 new candidates to scrape this run (already appended to candidates.yaml).
NEW_QUERY_NAMES = [
    "BMW G20 M340i exhaust",
    "BMW F30 340i exhaust",
    "BMW M Performance exhaust G20",
    "BMW G20 M Sport front splitter",
    "BMW F30 M Sport front splitter",
    "BMW G20 boot spoiler",
    "BMW F30 boot spoiler M Performance",
    "BMW G20 carbon mirror caps",
    "BMW F30 M Performance steering wheel",
    "BMW G20 M Sport steering wheel",
    "BMW F30 LCI rear lights",
    "BMW G20 LCI rear lights",
    "BMW F30 M Sport bumper",
    "BMW G20 M Sport bumper",
]

# F30 LCI variants from the previous turn — already in f30_lci_analysis.csv
F30_VARIANT_NAMES = [
    "BMW F30 LCI headlights pair",
    "BMW F30 LCI LED headlights set",
    "BMW F30 F31 LCI headlights both",
    "BMW F30 LCI adaptive LED",
]

# (hit_keyword_regex, natural_unit) for every candidate.
# natural_unit drives which bucket counts as "viable" for score_v2.
META: dict[str, tuple[str, str]] = {
    # Wheels (sets of 4 = canonical)
    "BMW G20 M Sport wheels":              (r"wheel|alloy|rim",                        "pair"),
    "BMW F30 M Sport 400M wheels":         (r"400.?M|wheel|alloy",                     "pair"),
    "BMW F80 M3 437M wheels":              (r"437.?M",                                 "pair"),
    "BMW F90 M5 706M wheels":              (r"706.?M",                                 "pair"),
    "BMW G80 M3 826M wheels":              (r"826.?M",                                 "pair"),
    "BMW E92 M3 220M wheels":              (r"220.?M",                                 "pair"),
    "BBS CH-R wheels BMW 5x120":           (r"BBS.*CH.?R|CH.?R",                       "pair"),

    # Exhausts (single)
    "Akrapovic exhaust BMW F80 M3":             (r"akrapovic",                          "single"),
    "Akrapovic exhaust BMW F87 M2 Competition": (r"akrapovic",                          "single"),
    "Eisenmann exhaust BMW E92 M3":             (r"eisenmann",                          "single"),
    "BMW M Performance exhaust F30 335i":       (r"exhaust|akrapovic|milltek|silencer|backbox|catback|downpipe", "single"),

    # Mirror caps (pair)
    "BMW M Performance carbon mirror caps F80 F82": (r"mirror",                         "pair"),
    "BMW M Performance carbon mirror caps G20 G80": (r"mirror",                         "pair"),

    # Carbon trim (single)
    "BMW M Performance carbon rear diffuser F80 M3":  (r"diffuser",                     "single"),
    "BMW M Performance carbon front splitter F80 M3": (r"splitter|lip",                 "single"),
    "BMW M Performance carbon rear spoiler F80 M3":   (r"spoiler",                      "single"),
    "BMW M Performance carbon engine cover F80":      (r"engine.?cover",                "single"),
    "BMW M4 GTS carbon front lip":                    (r"lip|splitter",                 "single"),
    "BMW F80 M3 carbon bonnet":                       (r"bonnet|hood",                  "single"),

    # Interior
    "BMW M Performance carbon steering wheel F80": (r"steering",                        "single"),
    "BMW M Performance Alcantara steering wheel":  (r"alcantara|steering",              "single"),
    "BMW G80 M3 carbon bucket seats":              (r"seat",                            "pair"),
    "BMW M Performance carbon shift paddles F80":  (r"paddle",                          "single"),

    # Suspension
    "KW V3 coilovers BMW F80 M3":                  (r"KW.*V3|coilover",                 "pair"),
    "Bilstein B16 coilovers BMW E92 M3":           (r"bilstein|coilover",               "pair"),

    # Lights
    "BMW G20 LCI laser headlights":                (r"headlight|headlamp|head\s+light", "pair"),
    "BMW F30 LCI LED headlights":                  (r"headlight|headlamp|head\s+light", "pair"),

    # Brakes
    "BMW M Performance brake calipers F30":        (r"brake|caliper",                   "pair"),

    # Electronics
    "BMW NBT Evo iD6 retrofit":                    (r"NBT|iD6|EVO",                     "single"),
    "BMW M Performance digital cluster F80":       (r"cluster|instrument",              "single"),

    # F30 LCI variants from previous turn
    "BMW F30 LCI headlights pair":                 (r"headlight|headlamp|head\s+light", "pair"),
    "BMW F30 LCI LED headlights set":              (r"headlight|headlamp|head\s+light", "pair"),
    "BMW F30 F31 LCI headlights both":             (r"headlight|headlamp|head\s+light", "pair"),
    "BMW F30 LCI adaptive LED":                    (r"headlight|headlamp|head\s+light", "pair"),

    # New 14
    "BMW G20 M340i exhaust":                       (r"exhaust|akrapovic|milltek|silencer|backbox|catback|downpipe", "single"),
    "BMW F30 340i exhaust":                        (r"exhaust|akrapovic|milltek|silencer|backbox|catback|downpipe", "single"),
    "BMW M Performance exhaust G20":               (r"exhaust|akrapovic|milltek|silencer|backbox|catback|downpipe", "single"),
    "BMW G20 M Sport front splitter":              (r"splitter|lip",                    "single"),
    "BMW F30 M Sport front splitter":              (r"splitter|lip",                    "single"),
    "BMW G20 boot spoiler":                        (r"spoiler",                         "single"),
    "BMW F30 boot spoiler M Performance":          (r"spoiler",                         "single"),
    "BMW G20 carbon mirror caps":                  (r"mirror",                          "pair"),
    "BMW F30 M Performance steering wheel":        (r"steering",                        "single"),
    "BMW G20 M Sport steering wheel":              (r"steering",                        "single"),
    "BMW F30 LCI rear lights":                     (r"rear.?light|tail.?light|brake.?light", "pair"),
    "BMW G20 LCI rear lights":                     (r"rear.?light|tail.?light|brake.?light", "pair"),
    "BMW F30 M Sport bumper":                      (r"bumper",                          "single"),
    "BMW G20 M Sport bumper":                      (r"bumper",                          "single"),
}

# Items where shipping cost is high (large/heavy/fragile items)
BIG_ITEM_KEYWORDS = ["wheel", "seat", "headlight", "rear light", "bumper",
                     "exhaust", "bonnet", "spoiler", "diffuser", "splitter"]


def _extract_item_id(url: str) -> str | None:
    m = ITEM_ID_RE.search(str(url) or "")
    return m.group(1) if m else None


def _scrape_new(candidates: list[dict]) -> pd.DataFrame:
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
    cols = ["candidate_name", "title", "price_gbp", "sold_date",
            "condition", "is_auction", "is_bundle", "url"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["item_id"] = out["url"].astype(str).apply(_extract_item_id)
    out["bucket"] = out["title"].apply(classify_pair_single)
    out["aftermarket_suspect"] = out["title"].apply(is_aftermarket_suspect)
    out["is_accessory"] = out["title"].astype(str).str.contains(
        ACCESSORY_RE, regex=True, na=False
    )

    def _hit(row):
        meta = META.get(row["candidate_name"])
        if not meta:
            return False
        return bool(re.search(meta[0], str(row["title"]), re.IGNORECASE))

    out["hit_match"] = out.apply(_hit, axis=1)
    return out


def _bucket_for_viable(name: str, natural: str,
                       pair_df: pd.DataFrame, single_df: pd.DataFrame,
                       unclear_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the listings that count as the natural unit for this candidate.

    For pair-natural items: include unclear (most unmarked listings are sets)
    EXCEPT for headlights/rear lights, where unclear is mostly single-side
    and would inflate the count — there we restrict to explicit pair only.
    For single-natural items: always include unclear (most unmarked listings
    are singles).
    """
    if natural == "pair":
        if re.search(r"headlight|rear light", name, re.IGNORECASE):
            return pair_df
        return pd.concat([pair_df, unclear_df])
    return pd.concat([single_df, unclear_df])


def _per_candidate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (_pat, natural) in META.items():
        sub = df[df["candidate_name"] == name]
        if len(sub) == 0:
            continue

        # Dedupe by item_id; drop accessories
        clean = sub.drop_duplicates(subset=["item_id"]).copy()
        clean = clean[~clean["is_accessory"]]
        n_unique = len(clean)

        hit_pct = round(100 * clean["hit_match"].mean(), 1) if n_unique else 0.0
        aft_n = int(clean["aftermarket_suspect"].sum())
        aft_pct = round(100 * aft_n / n_unique, 1) if n_unique else 0.0

        # Drop aftermarket — they're not the OEM-used target
        clean_oem = clean[~clean["aftermarket_suspect"]]

        pair_df = clean_oem[clean_oem["bucket"] == "pair"]
        single_df = clean_oem[clean_oem["bucket"] == "single"]
        unclear_df = clean_oem[clean_oem["bucket"] == "unclear"]

        viable = _bucket_for_viable(name, natural, pair_df, single_df, unclear_df)
        n_viable = len(viable)

        if n_viable >= 2:
            prices = viable["price_gbp"].astype(float)
            v_med = float(prices.median())
            v_p25 = float(prices.quantile(0.25))
            v_p75 = float(prices.quantile(0.75))
            stdev = float(prices.std(ddof=0))
            stab = stdev / v_med if v_med else float("nan")
        elif n_viable == 1:
            v_med = float(viable["price_gbp"].iloc[0])
            v_p25 = v_p75 = stab = float("nan")
        else:
            v_med = v_p25 = v_p75 = stab = float("nan")

        if n_viable > 0 and not np.isnan(stab) and not np.isnan(v_med):
            score_v2 = (
                math.log1p(n_viable)
                * (1.0 / (stab + 0.1))
                * math.sqrt(v_med)
                * (hit_pct / 100.0)
            )
        else:
            score_v2 = 0.0

        def _med(d):
            if len(d) == 0:
                return None
            return round(float(d["price_gbp"].median()), 2)

        rows.append({
            "candidate_name": name,
            "natural_unit": natural,
            "unique_items_90d": n_unique,
            "hit_pct": hit_pct,
            "aftermarket_suspect_count": aft_n,
            "aftermarket_suspect_pct": aft_pct,
            "pair_or_set_count": int(len(pair_df)),
            "pair_or_set_median": _med(pair_df),
            "single_count": int(len(single_df)),
            "single_median": _med(single_df),
            "unclear_count": int(len(unclear_df)),
            "viable_unit_count": int(n_viable),
            "viable_unit_median": round(v_med, 2) if not np.isnan(v_med) else None,
            "viable_unit_p25": round(v_p25, 2) if not np.isnan(v_p25) else None,
            "viable_unit_p75": round(v_p75, 2) if not np.isnan(v_p75) else None,
            "price_stability": round(stab, 3) if not np.isnan(stab) else None,
            "score_v2": round(score_v2, 3),
        })
    return pd.DataFrame(rows)


def _flip_economics(top: pd.Series) -> str:
    name = top["candidate_name"]
    median = float(top["viable_unit_median"])
    p25 = float(top["viable_unit_p25"]) if top["viable_unit_p25"] is not None else median * 0.6
    big = any(k in name.lower() for k in BIG_ITEM_KEYWORDS) or median >= 200
    ship = 25 if big else 8
    fee_rate = 0.125
    fee = median * fee_rate
    net = median - p25 - fee - ship
    margin_pct = (net / p25 * 100) if p25 > 0 else 0.0
    annual_units = int(top["viable_unit_count"]) * 4   # 90d -> 365d, simple x4
    annual_gross = net * annual_units
    return (
        f"\nFlip economics for #1 — {name}:\n"
        f"  Buy at p25:           £{p25:>7.2f}\n"
        f"  Sell at median:       £{median:>7.2f}\n"
        f"  eBay fee (12.5%):     £{fee:>7.2f}\n"
        f"  Shipping ({'big' if big else 'small'}):     £{ship:>7.2f}\n"
        f"  Net margin per unit:  £{net:>7.2f}  ({margin_pct:.0f}% of buy)\n"
        f"  90d viable count:     {int(top['viable_unit_count']):>5d}\n"
        f"  Annualised (×4):      ~{annual_units} units, ~£{annual_gross:.0f} gross opportunity\n"
        f"  Caveat: assumes you can actually source at p25, which is the unverified part."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cols = ["candidate_name", "title", "price_gbp", "sold_date",
            "condition", "is_auction", "is_bundle", "url"]

    raw = pd.read_csv("raw_listings.csv")
    logger.info(f"raw_listings.csv: {len(raw)} rows, "
                f"{raw['candidate_name'].nunique()} candidates")

    f30 = pd.read_csv("f30_lci_analysis.csv")
    f30_new = f30[f30["candidate_name"].isin(F30_VARIANT_NAMES)]
    logger.info(f"f30_lci_analysis.csv: {len(f30_new)} rows for 4 F30 variants")

    with open("candidates.yaml") as fp:
        cdata = yaml.safe_load(fp)
    by_name = {c["name"]: c for c in cdata["candidates"]}
    new_cands = [by_name[n] for n in NEW_QUERY_NAMES]

    new_df = _scrape_new(new_cands)
    logger.info(f"Scraped {len(new_df)} listings across {len(new_cands)} new queries")

    combined = pd.concat([raw[cols], f30_new[cols], new_df[cols]],
                         ignore_index=True)
    logger.info(f"Combined: {len(combined)} rows, "
                f"{combined['candidate_name'].nunique()} candidates")

    annotated = _annotate(combined)
    metrics = _per_candidate(annotated).sort_values(
        "score_v2", ascending=False
    ).reset_index(drop=True)
    metrics.insert(0, "rank", metrics.index + 1)

    out_csv = Path("f30_v2_ranking.csv")
    metrics.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(metrics)} rows to {out_csv}")

    print("\n" + "=" * 110)
    print("RANKED BY score_v2  (= log1p(n) * 1/(stab+0.1) * sqrt(median) * hit_pct/100)")
    print("=" * 110)
    show_cols = ["rank", "candidate_name", "natural_unit", "viable_unit_count",
                 "viable_unit_median", "hit_pct", "aftermarket_suspect_pct",
                 "score_v2"]
    with pd.option_context("display.max_colwidth", 50,
                           "display.width", 200,
                           "display.max_columns", None):
        print(metrics[show_cols].to_string(index=False))

    flagged = metrics[metrics["hit_pct"] < 70]
    if len(flagged):
        print("\n" + "-" * 110)
        print(f"{len(flagged)} candidate(s) flagged below 70% hit_pct (contaminated):")
        for _, r in flagged.iterrows():
            print(f"  - {r['candidate_name']:55s}  hit_pct={r['hit_pct']:>5.1f}%  "
                  f"score_v2={r['score_v2']:>6.2f}")

    top3 = metrics.head(3)
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    print("Top 3 by score_v2:")
    for _, r in top3.iterrows():
        print(f"  #{r['rank']} {r['candidate_name']}")
        print(f"      natural_unit={r['natural_unit']}, viable_n={r['viable_unit_count']}, "
              f"median £{r['viable_unit_median']}, hit_pct={r['hit_pct']}%, "
              f"aftermarket {r['aftermarket_suspect_pct']}%, score_v2={r['score_v2']}")

    cleanest = top3.sort_values(
        ["hit_pct", "aftermarket_suspect_pct"], ascending=[False, True]
    ).iloc[0]
    print(f"\nCleanest of the top 3: '{cleanest['candidate_name']}'")
    print(f"  hit_pct={cleanest['hit_pct']}%  "
          f"aftermarket_suspect_pct={cleanest['aftermarket_suspect_pct']}%")

    print(_flip_economics(metrics.iloc[0]))


if __name__ == "__main__":
    main()
