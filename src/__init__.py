"""Top-level package exports for the hardened Src EFD pipeline."""

from src.constants import (
    SrcSecurityError,
    SrcVerificationError,
    PACKAGE_VERSION,
    verify_condition,
)
from src.fourier import compute_elliptical_fourier_descriptors
from src.logging_config import configure_logging, get_logger
from src.models import (
    ApparelCollection,
    ApparelItem,
    EllipticalFourierCoefficients,
)
from src.pipeline import SrcPipeline, src

__version__: str = PACKAGE_VERSION

__all__ = [
    "PACKAGE_VERSION",
    "SrcVerificationError",
    "SrcSecurityError",
    "verify_condition",
    "src",
    "SrcPipeline",
    "ApparelItem",
    "ApparelCollection",
    "EllipticalFourierCoefficients",
    "compute_elliptical_fourier_descriptors",
    "configure_logging",
    "get_logger",
]