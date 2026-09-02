"""Consistent, self-describing names for saved run artifacts."""

from __future__ import annotations

import re


def safe_name_component(value: str) -> str:
    """Convert a mode or geometry label into a filename-safe component."""
    component = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return component or "unknown"


def ideal_kinoform_stem(mode: str) -> str:
    return f"Ideal_{safe_name_component(mode)}_Kinoform"


def grid_phase_stem(mode: str, geometry: str) -> str | None:
    labels = {
        "hexagonal": "Hex_grid",
        "square": "Square_grid",
    }
    prefix = labels.get(geometry)
    if prefix is None:
        return None
    return f"{prefix}_{safe_name_component(mode)}"


def comparison_plot_name(mode: str, geometry: str) -> str | None:
    grid_stem = grid_phase_stem(mode, geometry)
    if grid_stem is None:
        return None
    return (
        f"{ideal_kinoform_stem(mode)}_vs_"
        f"{grid_stem}_comparison.png"
    )
