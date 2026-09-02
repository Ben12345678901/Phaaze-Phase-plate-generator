"""Binary and multi-level stepped phase plates."""

from __future__ import annotations

import numpy as np

from .base import PhaseMode


BINARIZATION_ALIASES = {
    "threshold": "threshold",
    "fixed_threshold": "threshold",
    "fixed-threshold": "threshold",
    "median": "median",
    "amplitude_weighted": "amplitude_weighted",
    "amplitude-weighted": "amplitude_weighted",
    "weighted": "amplitude_weighted",
}


def normalise_binarization_method(method: str) -> str:
    try:
        return BINARIZATION_ALIASES[str(method).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "binarization method must be 'threshold', 'median', or "
            "'amplitude_weighted'"
        ) from exc


def _balanced_selection(
    scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Select the highest scores with weight closest to half the total."""
    score_values = np.asarray(scores, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if score_values.ndim != 1 or score_values.shape != weight_values.shape:
        raise ValueError("binary balance scores and weights must be 1-D")
    if not np.isfinite(score_values).all():
        raise ValueError("binary balance scores must be finite")
    if (
        not np.isfinite(weight_values).all()
        or np.any(weight_values < 0)
        or not np.any(weight_values > 0)
    ):
        raise ValueError(
            "binary balance weights must be finite, non-negative, and "
            "contain positive weight"
        )

    selected = np.zeros(score_values.shape, dtype=bool)
    if np.all(weight_values == weight_values[0]):
        selected_count = score_values.size // 2
        if selected_count:
            highest = np.argpartition(
                -score_values, selected_count - 1
            )[:selected_count]
            selected[highest] = True
        return selected

    # Weighted quick-selection avoids sorting a multi-million-pixel plate on
    # every binary projection. Only the score band containing the weighted
    # median is retained; a small final sort determines the closest prefix.
    target_weight = float(weight_values.sum() / 2)
    selected_weight = 0.0
    active = np.arange(score_values.size)
    while active.size > 4096:
        pivot = float(np.median(score_values[active]))
        greater = active[score_values[active] > pivot]
        equal = active[score_values[active] == pivot]
        lower = active[score_values[active] < pivot]
        greater_weight = float(weight_values[greater].sum())
        equal_weight = float(weight_values[equal].sum())
        if selected_weight + greater_weight > target_weight:
            active = greater
        elif (
            selected_weight + greater_weight + equal_weight
            < target_weight
        ):
            selected[greater] = True
            selected[equal] = True
            selected_weight += greater_weight + equal_weight
            active = lower
        else:
            selected[greater] = True
            selected_weight += greater_weight
            active = equal
            break

    order = active[np.argsort(-score_values[active], kind="stable")]
    cumulative = selected_weight + np.concatenate(
        ([0.0], np.cumsum(weight_values[order], dtype=float))
    )
    selected_count = int(np.argmin(np.abs(cumulative - target_weight)))
    selected[order[:selected_count]] = True
    return selected


def _balanced_binary_phase(
    wrapped_phase_rad: np.ndarray,
    *,
    valid_mask: np.ndarray,
    balance_weights: np.ndarray,
    group_labels: np.ndarray | None,
) -> np.ndarray:
    """Return a zero/pi phase whose selected weight is closest to one half."""
    phase = np.asarray(wrapped_phase_rad, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    weights = np.asarray(balance_weights, dtype=float)
    if phase.shape != mask.shape or phase.shape != weights.shape:
        raise ValueError(
            "phase, binary balance mask, and weights must have matching shapes"
        )
    if not mask.any():
        raise ValueError("binary balance mask contains no valid samples")

    principal = np.angle(np.exp(1j * phase))
    sample_scores = np.abs(principal[mask])
    sample_weights = weights[mask]
    result = np.zeros(phase.shape, dtype=float)

    if group_labels is None:
        selected = _balanced_selection(sample_scores, sample_weights)
        valid_indices = np.flatnonzero(mask)
        result.flat[valid_indices[selected]] = np.pi
        return result

    labels = np.asarray(group_labels)
    if labels.shape != phase.shape:
        raise ValueError("binary group labels must match the phase shape")
    sample_labels = labels[mask]
    if np.any(sample_labels < 0):
        raise ValueError(
            "binary group labels must be non-negative inside the valid mask"
        )
    unique_labels, inverse = np.unique(sample_labels, return_inverse=True)
    group_count = unique_labels.size
    cosine = np.bincount(
        inverse, weights=np.cos(principal[mask]), minlength=group_count
    )
    sine = np.bincount(
        inverse, weights=np.sin(principal[mask]), minlength=group_count
    )
    group_scores = np.abs(np.arctan2(sine, cosine))
    group_weights = np.bincount(
        inverse, weights=sample_weights, minlength=group_count
    )
    selected_groups = _balanced_selection(group_scores, group_weights)
    result[mask] = np.where(selected_groups[inverse], np.pi, 0.0)
    return result


def quantize_phase(
    wrapped_phase_rad: np.ndarray,
    levels: int,
    *,
    binarization_method: str = "threshold",
    valid_mask: np.ndarray | None = None,
    amplitude_weights: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be at least 2")
    method = normalise_binarization_method(binarization_method)
    phase = np.asarray(wrapped_phase_rad, dtype=float)
    mask = (
        np.ones(phase.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != phase.shape:
        raise ValueError("binary valid mask must match the phase shape")

    if levels == 2 and method != "threshold":
        if method == "median":
            weights = np.ones(phase.shape, dtype=float)
        else:
            if amplitude_weights is None:
                raise ValueError(
                    "amplitude_weighted binarization requires incident "
                    "field-amplitude weights"
                )
            weights = np.asarray(amplitude_weights, dtype=float)
        return _balanced_binary_phase(
            phase,
            valid_mask=mask,
            balance_weights=weights,
            group_labels=group_labels,
        )
    if levels != 2 and method != "threshold":
        raise ValueError(
            "median and amplitude_weighted binarization require levels=2"
        )

    step = 2 * np.pi / levels
    result = (
        np.round(np.mod(phase, 2 * np.pi) / step) * step
    )
    result[result >= 2 * np.pi] = 0
    return np.where(mask, result, 0.0)


class MultilevelMode(PhaseMode):
    """Ramp an optimized kinoform onto exactly N equally spaced levels."""

    canonical_name = "multilevel"

    def __init__(
        self,
        *,
        levels: int,
        ramp_fraction: float,
        binarization_method: str,
        amplitude_weights: np.ndarray | None = None,
        binary_group_labels: np.ndarray | None = None,
        binary_group_start_iteration: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.levels = levels
        self.ramp_fraction = ramp_fraction
        self.binarization_method = normalise_binarization_method(
            binarization_method
        )
        self.amplitude_weights = (
            None
            if amplitude_weights is None
            else np.asarray(amplitude_weights, dtype=float)
        )
        self.binary_group_labels = (
            None
            if binary_group_labels is None
            else np.asarray(binary_group_labels)
        )
        self.binary_group_start_iteration = int(
            binary_group_start_iteration
        )
        for name, values in (
            ("amplitude weights", self.amplitude_weights),
            ("binary group labels", self.binary_group_labels),
        ):
            if values is not None and values.shape != self.pupil.shape:
                raise ValueError(f"{name} must match the mode pupil shape")

    def _quantize(
        self,
        phase_rad: np.ndarray,
        *,
        group_labels: np.ndarray | None,
    ) -> np.ndarray:
        return quantize_phase(
            phase_rad,
            self.levels,
            binarization_method=self.binarization_method,
            valid_mask=self.pupil,
            amplitude_weights=self.amplitude_weights,
            group_labels=group_labels,
        )

    def project(
        self,
        wrapped_candidate_rad: np.ndarray,
        iteration: int,
        total_iterations: int,
    ) -> np.ndarray:
        if self.ramp_fraction <= 0:
            return np.where(self.pupil, wrapped_candidate_rad, 0.0)
        ramp_iterations = max(
            1, int(round(total_iterations * self.ramp_fraction))
        )
        ramp_start = total_iterations - ramp_iterations
        alpha = float(
            np.clip(
                (iteration + 1 - ramp_start) / ramp_iterations,
                0,
                1,
            )
        )
        if alpha == 0:
            result = wrapped_candidate_rad
        else:
            group_labels = (
                self.binary_group_labels
                if iteration >= self.binary_group_start_iteration
                else None
            )
            quantized = self._quantize(
                wrapped_candidate_rad,
                group_labels=group_labels,
            )
            result = np.angle(
                (1 - alpha) * np.exp(1j * wrapped_candidate_rad)
                + alpha * np.exp(1j * quantized)
            )
        return np.where(self.pupil, result, 0.0)

    def finalize(self, wrapped_phase_rad: np.ndarray):
        return super().finalize(
            self._quantize(
                wrapped_phase_rad,
                group_labels=self.binary_group_labels,
            )
        )
