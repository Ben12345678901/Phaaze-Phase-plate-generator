"""Construct typed decks from Python mappings or legacy JSON files."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from .config import (
    AlgorithmConfig,
    BeamConfig,
    ExperimentDeck,
    FacilityConfig,
    GridConfig,
    ManufacturingConfig,
    OpticsConfig,
    OutputConfig,
    PhasePlateConfig,
    TargetConfig,
)

T = TypeVar("T")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_declared_paths(
    data: dict[str, Any], *, relative_to: Path
) -> dict[str, Any]:
    """Resolve paths in the deck that actually declared them."""
    result = dict(data)
    path_fields = {
        "beam": ("array_path",),
        "target": ("array_path",),
        "algorithm": ("initial_phase_path",),
        "output": ("directory",),
    }
    for section, names in path_fields.items():
        if section not in result or not isinstance(result[section], dict):
            continue
        section_values = dict(result[section])
        for name in names:
            value = section_values.get(name)
            if value is None:
                continue
            path = Path(value)
            section_values[name] = str(
                path
                if path.is_absolute()
                else (relative_to / path).resolve()
            )
        result[section] = section_values
    return result


def _read_with_inheritance(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ValueError(f"Circular input-deck inheritance: {chain}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    parent_name = data.pop("extends", None)
    data = _resolve_declared_paths(data, relative_to=resolved.parent)
    if parent_name is None:
        return data
    parent_path = (resolved.parent / parent_name).resolve()
    parent = _read_with_inheritance(parent_path, seen)
    return _deep_merge(parent, data)


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    valid = {item.name for item in fields(cls)}
    unexpected = sorted(set(values) - valid)
    if unexpected:
        raise ValueError(
            f"Unexpected {cls.__name__} keys: {', '.join(unexpected)}"
        )
    return cls(**values)


def _resolve_optional_path(
    value: str | None, *, relative_to: Path
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (relative_to / path).resolve()


def deck_from_mapping(
    data: dict[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ExperimentDeck:
    """Resolve, type, and validate a Python input-deck dictionary."""
    source = (
        Path.cwd() / "input_deck.py"
        if source_path is None
        else Path(source_path).resolve()
    )
    data = _resolve_declared_paths(
        dict(data), relative_to=source.parent
    )
    valid_sections = {
        "facility",
        "grid",
        "optics",
        "beam",
        "target",
        "algorithm",
        "phase_plate",
        "manufacturing",
        "output",
    }
    unexpected = sorted(set(data) - valid_sections)
    if unexpected:
        raise ValueError(
            f"Unexpected input-deck sections: {', '.join(unexpected)}"
        )

    beam = dict(data.get("beam", {}))
    target = dict(data.get("target", {}))
    algorithm = dict(data.get("algorithm", {}))
    output = dict(data.get("output", {}))
    beam["array_path"] = _resolve_optional_path(
        beam.get("array_path"), relative_to=source.parent
    )
    target["array_path"] = _resolve_optional_path(
        target.get("array_path"), relative_to=source.parent
    )
    algorithm["initial_phase_path"] = _resolve_optional_path(
        algorithm.get("initial_phase_path"), relative_to=source.parent
    )
    if "directory" in output:
        output["directory"] = _resolve_optional_path(
            output["directory"], relative_to=source.parent
        )

    deck = ExperimentDeck(
        facility=_construct(FacilityConfig, data.get("facility", {})),
        grid=_construct(GridConfig, data.get("grid", {})),
        optics=_construct(OpticsConfig, data.get("optics", {})),
        beam=_construct(BeamConfig, beam),
        target=_construct(TargetConfig, target),
        algorithm=_construct(AlgorithmConfig, algorithm),
        phase_plate=_construct(
            PhasePlateConfig, data.get("phase_plate", {})
        ),
        manufacturing=_construct(
            ManufacturingConfig, data.get("manufacturing", {})
        ),
        output=_construct(OutputConfig, output),
        source_path=source,
    )
    deck.validate()
    return deck


def load_deck(path: str | Path) -> ExperimentDeck:
    """Load an inheritable legacy JSON deck.

    New project input decks are self-contained Python files. This loader is
    retained for compatibility with existing external JSON decks.
    """
    source_path = Path(path).resolve()
    data = _read_with_inheritance(source_path, set())
    return deck_from_mapping(data, source_path=source_path)
