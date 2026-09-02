"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decks import load_deck
from .generator import generate
from .modes import normalise_mode_name
from .runner import run_deck_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase-plate",
        description="Generate a phase plate from an input deck.",
    )
    parser.add_argument(
        "deck", nargs="?", type=Path, help="JSON input deck"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="resolve and validate the deck without running",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="run without writing output artifacts",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="list canonical mode names",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_modes:
        for name in (
            "continuous_wrapped",
            "continuous_relief",
            "distributed_relief",
            "multilevel",
            "random_binary",
        ):
            print(name)
        return 0
    if args.deck is None:
        raise SystemExit("a JSON input deck is required")

    deck = load_deck(args.deck)
    if args.validate_only:
        print(json.dumps(deck.as_dict(), indent=2))
        return 0
    result = (
        generate(deck, save=False)
        if args.no_save
        else run_deck_file(args.deck)
    )
    metrics = result.metadata["metrics"]
    print(
        f"mode={normalise_mode_name(deck.phase_plate.mode)} "
        f"PCC={metrics['pcc']:.6f} "
        f"relative_NRMSE={metrics['relative_nrmse']:.6f}"
    )
    if result.output_directory is not None:
        print(f"outputs={result.output_directory}")
    return 0
