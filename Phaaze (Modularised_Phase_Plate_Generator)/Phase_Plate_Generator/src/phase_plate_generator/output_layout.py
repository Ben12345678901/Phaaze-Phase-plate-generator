"""Directory layout for one timestamped phase-plate run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunFolders:
    root: Path
    plots: Path
    json_files: Path
    arrays: Path
    input_deck_used: Path


def ensure_run_folders(root: str | Path) -> RunFolders:
    """Create and return the standard artifact folders for one run."""
    run_root = Path(root).resolve()
    folders = RunFolders(
        root=run_root,
        plots=run_root / "plots",
        json_files=run_root / "json_files",
        arrays=run_root / "Arrays",
        input_deck_used=run_root / "Input deck used",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    for folder in (
        folders.plots,
        folders.json_files,
        folders.arrays,
        folders.input_deck_used,
    ):
        folder.mkdir(exist_ok=True)
    return folders
