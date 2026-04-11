#!/usr/bin/env python3
"""Prepare standardized stimuli from BBQ JSONL files for all bias categories.

Usage:
    python scripts/prepare_stimuli.py --categories so,gi,race --output_dir data/processed
    python scripts/prepare_stimuli.py --categories all
    python scripts/prepare_stimuli.py --categories so --max_items 20
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bbq_loader import (
    ALL_CATEGORIES,
    CATEGORY_MAP,
    load_and_standardize,
    parse_categories,
)
from src.utils.io import atomic_save_json
from src.utils.logging import log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare standardized BBQ stimuli for activation extraction."
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated category short names (so,gi,race,...) or 'all'",
    )
    parser.add_argument(
        "--bbq_data_dir",
        type=str,
        default="datasets/bbq/data",
        help="Path to directory containing BBQ JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed stimuli JSON files",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum items per category (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for answer shuffling",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string for output filenames (default: today)",
    )

    args = parser.parse_args()
    date_str = args.date or date.today().isoformat()
    categories = parse_categories(args.categories)

    log(f"Preparing stimuli for {len(categories)} categories: {categories}")
    log(f"BBQ data dir: {args.bbq_data_dir}")
    log(f"Output dir: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for cat in categories:
        bbq_name = CATEGORY_MAP[cat]
        log(f"\n{'='*60}")
        log(f"Processing category: {cat} ({bbq_name})")
        log(f"{'='*60}")

        items = load_and_standardize(cat, args.bbq_data_dir, seed=args.seed)

        if args.max_items is not None:
            items = items[: args.max_items]
            log(f"  Truncated to {len(items)} items (--max_items={args.max_items})")

        # Validate: every item should have exactly one unknown answer
        n_valid = 0
        n_missing_unknown = 0
        for item in items:
            roles = list(item["answer_roles"].values())
            if "unknown" in roles:
                n_valid += 1
            else:
                n_missing_unknown += 1
        if n_missing_unknown > 0:
            log(f"  WARNING: {n_missing_unknown} items have no 'unknown' answer role")

        # Validate: stereotyped_target should appear in most items
        n_has_stereo = sum(
            1 for item in items if "stereotyped_target" in item["answer_roles"].values()
        )
        log(f"  Items with stereotyped_target answer: {n_has_stereo}/{len(items)}")

        # Save
        out_path = output_dir / f"stimuli_{cat}_{date_str}.json"
        atomic_save_json(items, out_path)
        log(f"  Saved {len(items)} items -> {out_path}")

        summary[cat] = {
            "n_items": len(items),
            "n_valid_unknown": n_valid,
            "n_has_stereotyped": n_has_stereo,
            "output_path": str(out_path),
            "bbq_file": bbq_name,
        }

    # Print summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    for cat, info in summary.items():
        log(f"  {cat:>20s}: {info['n_items']:>5d} items -> {info['output_path']}")

    # Save summary
    summary_path = output_dir / f"stimuli_summary_{date_str}.json"
    atomic_save_json(summary, summary_path)
    log(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
