"""Safety-bounded Elliptical Fourier Descriptor (EFD) computation engine."""

import math
from typing import List
import numpy as np

from src.constants import (
    CONTOUR_COORD_DIM,
    DEFAULT_EPSILON,
    GRAY_DIMENSIONS,
    MAX_CONTOUR_POINTS_CEILING,
    MAX_HARMONICS_CEILING,
    MIN_CONTOUR_POINTS_FLOOR,
    ONE_INT,
    TEN_INT,
    TWO_FLOAT,
    TWO_INT,
    ZERO_FLOAT,
    ZERO_INT,
    verify_condition,
)
from src.models import EllipticalFourierCoefficients


def compute_elliptical_fourier_descriptors(
    contour: np.ndarray,
    order: int = TEN_INT,
    normalize: bool = True,
    max_contour_points: int = MAX_CONTOUR_POINTS_CEILING,
    max_harmonics_allowed: int = MAX_HARMONICS_CEILING,
    zero_displacement_epsilon: float = DEFAULT_EPSILON,
) -> EllipticalFourierCoefficients:
    """Computes Elliptical Fourier Descriptors for a closed 2D contour.

    Args:
        contour (np.ndarray): Contour coordinate array of shape (N, 2).
        order (int, optional): Number of Fourier harmonics. Defaults to
            TEN_INT.
        normalize (bool, optional): Whether to scale-normalize descriptors.
            Defaults to True.
        max_contour_points (int, optional): Upper limit on contour points.
            Defaults to MAX_CONTOUR_POINTS_CEILING.
        max_harmonics_allowed (int, optional): Maximum harmonic order
            supported. Defaults to MAX_HARMONICS_CEILING.
        zero_displacement_epsilon (float, optional): Epsilon value preventing
            division by zero. Defaults to DEFAULT_EPSILON.

    Returns:
        EllipticalFourierCoefficients: Validated, immutable EFD coefficients.
    """
    verify_condition(
        isinstance(contour, np.ndarray),
        "Contour must be a valid NumPy ndarray.",
    )
    verify_condition(
        contour.ndim == GRAY_DIMENSIONS
        and contour.shape[ONE_INT] == CONTOUR_COORD_DIM,
        "Contour array must have shape (N, 2).",
    )
    eff_max_harmonics = min(
        int(max_harmonics_allowed), MAX_HARMONICS_CEILING
    )
    verify_condition(
        ONE_INT <= int(order) <= eff_max_harmonics,
        "Harmonic order is outside the permitted range.",
    )
    verify_condition(
        float(zero_displacement_epsilon) > ZERO_FLOAT,
        "Zero displacement epsilon must be strictly positive.",
    )

    eff_max_points = min(
        int(max_contour_points), MAX_CONTOUR_POINTS_CEILING
    )
    num_points = min(int(len(contour)), eff_max_points)
    verify_condition(
        num_points >= MIN_CONTOUR_POINTS_FLOOR,
        "Contour requires at least 3 points for Fourier analysis.",
    )

    bounded_contour = contour[:num_points].astype(np.float64)
    dxy = np.diff(
        bounded_contour,
        axis=ZERO_INT,
        append=bounded_contour[ZERO_INT:ONE_INT],
    )
    dt = np.sqrt(np.sum(dxy**TWO_INT, axis=ONE_INT))
    dt = np.where(
        dt <= ZERO_FLOAT, float(zero_displacement_epsilon), dt
    )

    t_steps = np.concatenate(([ZERO_FLOAT], np.cumsum(dt)))
    period = float(t_steps[-ONE_INT])
    verify_condition(
        period > ZERO_FLOAT,
        "Contour perimeter period must be strictly positive.",
    )

    phi = (TWO_FLOAT * math.pi * t_steps) / period

    a_list: List[float] = []
    b_list: List[float] = []
    c_list: List[float] = []
    d_list: List[float] = []

    harmonic_limit = min(int(order), eff_max_harmonics)
    for n_idx in range(ONE_INT, harmonic_limit + ONE_INT):
        denom = TWO_FLOAT * float(n_idx**TWO_INT) * (math.pi**TWO_INT)
        const = period / denom
        phi_n = phi * float(n_idx)

        d_cos_phi = np.cos(phi_n[ONE_INT:]) - np.cos(phi_n[:-ONE_INT])
        d_sin_phi = np.sin(phi_n[ONE_INT:]) - np.sin(phi_n[:-ONE_INT])

        a_n = const * float(np.sum((dxy[:, ZERO_INT] / dt) * d_cos_phi))
        b_n = const * float(np.sum((dxy[:, ZERO_INT] / dt) * d_sin_phi))
        c_n = const * float(np.sum((dxy[:, ONE_INT] / dt) * d_cos_phi))
        d_n = const * float(np.sum((dxy[:, ONE_INT] / dt) * d_sin_phi))

        a_list.append(a_n)
        b_list.append(b_n)
        c_list.append(c_n)
        d_list.append(d_n)

    if normalize and len(a_list) > ZERO_INT:
        scale = math.sqrt(
            (a_list[ZERO_INT] ** TWO_INT) + (c_list[ZERO_INT] ** TWO_INT)
        )
        if scale > float(zero_displacement_epsilon):
            a_list = [val / scale for val in a_list]
            b_list = [val / scale for val in b_list]
            c_list = [val / scale for val in c_list]
            d_list = [val / scale for val in d_list]

    return EllipticalFourierCoefficients(
        harmonic_order=harmonic_limit,
        a_coeffs=tuple(a_list),
        b_coeffs=tuple(b_list),
        c_coeffs=tuple(c_list),
        d_coeffs=tuple(d_list),
    )