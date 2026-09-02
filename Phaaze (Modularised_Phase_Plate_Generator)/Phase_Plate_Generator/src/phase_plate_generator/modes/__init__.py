"""Named phase-plate representations.

Canonical names:

``continuous_wrapped``
    Unquantized phase-only kinoform. This does not guarantee a smooth physical
    surface.
``continuous_relief``
    Strictly continuous, unwrapped manufacturable relief (legacy
    ``true_continuous``).
``distributed_relief``
    Continuous random relief whose RMS and correlation statistics are retained
    during retrieval (legacy ``distributed``/``dpp``).
``multilevel``
    Optimized N-level stepped plate; N=2 is a binary kinoform.
``random_binary``
    Stochastic zero/pi random phase plate, intentionally not GS optimized.
"""

from __future__ import annotations

import numpy as np

from ..config import PhasePlateConfig
from .base import PhaseMode
from .continuous_relief import ContinuousReliefMode
from .continuous_wrapped import ContinuousWrappedMode
from .distributed_relief import DistributedReliefMode
from .multilevel import MultilevelMode
from .random_binary import RandomBinaryMode


MODE_ALIASES = {
    "continuous": "continuous_wrapped",
    "continuous_wrapped": "continuous_wrapped",
    "wrapped_continuous": "continuous_wrapped",
    "kinoform": "continuous_wrapped",
    "true continuous": "continuous_relief",
    "true-continuous": "continuous_relief",
    "true_continuous": "continuous_relief",
    "continuous_relief": "continuous_relief",
    "continuous_surface": "continuous_relief",
    "distributed": "distributed_relief",
    "dpp": "distributed_relief",
    "distributed_phase_plate": "distributed_relief",
    "distributed_relief": "distributed_relief",
    "quantized": "multilevel",
    "quantised": "multilevel",
    "multi-level": "multilevel",
    "multi_level": "multilevel",
    "multilevel": "multilevel",
    "binary": "multilevel",
    "random": "random_binary",
    "random_binary": "random_binary",
    "random_phase_plate": "random_binary",
    "rpp": "random_binary",
}


def normalise_mode_name(name: str) -> str:
    try:
        return MODE_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        available = ", ".join(
            (
                "continuous_wrapped",
                "continuous_relief",
                "distributed_relief",
                "multilevel",
                "random_binary",
            )
        )
        raise ValueError(f"Unknown phase mode; choose one of: {available}") from exc


def build_mode(
    config: PhasePlateConfig,
    *,
    pupil: np.ndarray,
    pixel_size_m: float,
    rng: np.random.Generator,
    default_correlation_length_m: float,
    input_amplitude: np.ndarray | None = None,
    binary_group_labels: np.ndarray | None = None,
    binary_group_start_iteration: int = 0,
) -> PhaseMode:
    mode_name = normalise_mode_name(config.mode)
    common = {
        "pupil": pupil,
        "pixel_size_m": pixel_size_m,
        "rng": rng,
    }
    if mode_name == "continuous_wrapped":
        return ContinuousWrappedMode(**common)
    if mode_name == "continuous_relief":
        return ContinuousReliefMode(
            correlation_length_m=(
                default_correlation_length_m
                if config.correlation_length_m is None
                else config.correlation_length_m
            ),
            initial_rms_rad=config.phase_rms_rad,
            **common,
        )
    if mode_name == "distributed_relief":
        return DistributedReliefMode(
            correlation_length_m=(
                default_correlation_length_m
                if config.correlation_length_m is None
                else config.correlation_length_m
            ),
            phase_rms_rad=config.phase_rms_rad,
            projection_strength=config.distributed_projection_strength,
            **common,
        )
    if mode_name == "multilevel":
        return MultilevelMode(
            levels=config.levels,
            ramp_fraction=config.quantization_ramp_fraction,
            binarization_method=config.binarization_method,
            amplitude_weights=input_amplitude,
            binary_group_labels=binary_group_labels,
            binary_group_start_iteration=binary_group_start_iteration,
            **common,
        )
    return RandomBinaryMode(
        pi_fraction=config.random_pi_fraction, **common
    )


__all__ = [
    "MODE_ALIASES",
    "PhaseMode",
    "build_mode",
    "normalise_mode_name",
]
