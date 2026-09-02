"""Run a saved input deck into a unique, timestamped output folder."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shutil

from .config import ExperimentDeck
from .decks import load_deck
from .generator import GenerationResult, generate
from .outputs import save_result
from .output_layout import ensure_run_folders


def _safe_deck_name(path: Path) -> str:
    """Return a filesystem-safe form of an input deck's filename stem."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return name or "input_deck"


def create_timestamped_run_directory(
    input_deck_path: str | Path,
    outputs_root: str | Path,
    *,
    started_at: datetime | None = None,
) -> Path:
    """Create and reserve ``<deck>_<date>_<time>`` under ``outputs_root``.

    A numeric suffix is added if more than one run begins in the same second.
    """
    deck_path = Path(input_deck_path).resolve()
    root = Path(outputs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_time = (
        datetime.now().astimezone() if started_at is None else started_at
    )
    base_name = (
        f"{_safe_deck_name(deck_path)}_"
        f"{run_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    candidate = root / base_name
    suffix = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = root / f"{base_name}_{suffix:02d}"


def _run_loaded_deck(
    deck: ExperimentDeck,
    input_deck_path: Path,
    *,
    outputs_root: str | Path | None = None,
    started_at: datetime | None = None,
    source_paths: tuple[Path, ...] = (),
    launcher_path: Path | None = None,
) -> GenerationResult:
    """Execute an already constructed deck and preserve its source files."""
    deck_path = input_deck_path.resolve()
    run_time = (
        datetime.now().astimezone() if started_at is None else started_at
    )
    root = (
        Path(deck.output.directory)
        if outputs_root is None
        else Path(outputs_root)
    )
    run_directory = create_timestamped_run_directory(
        deck_path,
        root,
        started_at=run_time,
    )
    folders = ensure_run_folders(run_directory)
    deck.output.directory = run_directory

    unique_sources: list[Path] = []
    for source in (deck_path, *source_paths):
        resolved_source = source.resolve()
        if resolved_source not in unique_sources:
            unique_sources.append(resolved_source)
    for source in unique_sources:
        if not source.is_file():
            raise FileNotFoundError(
                f"Input-deck source does not exist: {source}"
            )
        shutil.copy2(
            source, folders.input_deck_used / source.name
        )
    try:
        result = generate(deck, save=False)
        result.metadata["run"] = {
            "input_deck_name": deck_path.stem,
            "input_deck_path": str(deck_path),
            "started_at": run_time.isoformat(),
            "output_directory": str(run_directory),
            "launcher_path": (
                None if launcher_path is None else str(launcher_path)
            ),
            "source_files": [str(source) for source in unique_sources],
        }
        result.output_directory = save_result(result, deck)
        if deck.output.save_plots or deck.output.show_plots:
            from .plotting import plot_generation_result

            saved_plots = plot_generation_result(result, deck)
            if saved_plots:
                print(
                    "Saved plots: "
                    + ", ".join(path.name for path in saved_plots)
                )
        return result
    except Exception as exc:
        failure = {
            "input_deck_name": deck_path.stem,
            "input_deck_path": str(deck_path),
            "started_at": run_time.isoformat(),
            "launcher_path": (
                None if launcher_path is None else str(launcher_path)
            ),
            "source_files": [str(source) for source in unique_sources],
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        (folders.json_files / "run_failed.json").write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


def run_configured_deck(
    deck: ExperimentDeck,
    input_deck_path: str | Path,
    *,
    outputs_root: str | Path | None = None,
    started_at: datetime | None = None,
) -> GenerationResult:
    """Run a self-contained Python deck into a timestamped output folder."""
    deck_path = Path(input_deck_path).resolve()
    if deck_path.suffix.lower() != ".py":
        raise ValueError(
            "run_configured_deck expects the runnable Python deck path"
        )
    deck.source_path = deck_path
    deck.validate()
    return _run_loaded_deck(
        deck,
        deck_path,
        outputs_root=outputs_root,
        started_at=started_at,
        source_paths=(deck_path,),
        launcher_path=deck_path,
    )


def run_deck_file(
    input_deck_path: str | Path,
    *,
    outputs_root: str | Path | None = None,
    started_at: datetime | None = None,
    launcher_path: str | Path | None = None,
) -> GenerationResult:
    """Load and run a legacy JSON deck.

    New project decks use :func:`run_configured_deck` directly from Python.
    """
    deck_path = Path(input_deck_path).resolve()
    launcher = (
        None if launcher_path is None else Path(launcher_path).resolve()
    )
    deck = load_deck(deck_path)
    sources = (deck_path,) if launcher is None else (deck_path, launcher)
    return _run_loaded_deck(
        deck,
        deck_path,
        outputs_root=outputs_root,
        started_at=started_at,
        source_paths=sources,
        launcher_path=launcher,
    )
