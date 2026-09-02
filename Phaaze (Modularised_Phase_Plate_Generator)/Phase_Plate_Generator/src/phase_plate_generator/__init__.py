"""Modular phase-plate design package."""

from .config import ExperimentDeck
from .decks import deck_from_mapping, load_deck
from .generator import GenerationResult, generate
from .runner import run_configured_deck, run_deck_file

__all__ = [
    "ExperimentDeck",
    "GenerationResult",
    "deck_from_mapping",
    "generate",
    "load_deck",
    "run_configured_deck",
    "run_deck_file",
]
__version__ = "0.1.0"
