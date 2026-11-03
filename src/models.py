"""Immutable data models and sequence containers for the Src pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.constants import (
    BBOX_TUPLE_LENGTH,
    CENTROID_TUPLE_LENGTH,
    MAX_HARMONICS_CEILING,
    MAX_ITEMS_CEILING,
    ONE_INT,
    ROUND_DECIMALS_COEFFS,
    ROUND_DECIMALS_METRICS,
    ROUND_DECIMALS_SHAPES,
    ZERO_FLOAT,
    ZERO_INT,
    verify_condition,
)


@dataclass(frozen=True, slots=True)
class EllipticalFourierCoefficients:
    """Immutable record storing Elliptical Fourier Coefficients (EFDs)."""

    harmonic_order: int
    a_coeffs: Tuple[float, ...]
    b_coeffs: Tuple[float, ...]
    c_coeffs: Tuple[float, ...]
    d_coeffs: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Validates internal harmonic coefficient constraints at runtime."""
        verify_condition(
            ONE_INT <= int(self.harmonic_order) <= MAX_HARMONICS_CEILING,
            "Harmonic order must be positive and within ceiling.",
        )
        verify_condition(
            len(self.a_coeffs) == self.harmonic_order,
            "A coefficients tuple length mismatch.",
        )
        verify_condition(
            len(self.b_coeffs) == self.harmonic_order,
            "B coefficients tuple length mismatch.",
        )
        verify_condition(
            len(self.c_coeffs) == self.harmonic_order,
            "C coefficients tuple length mismatch.",
        )
        verify_condition(
            len(self.d_coeffs) == self.harmonic_order,
            "D coefficients tuple length mismatch.",
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts Fourier coefficients into a flattened dictionary.

        Returns:
            Dict[str, float]: Flattened dictionary of harmonic coefficients.
        """
        flat_dict: Dict[str, float] = {}
        limit = min(self.harmonic_order, MAX_HARMONICS_CEILING)
        for idx in range(limit):
            h_num = idx + ONE_INT
            flat_dict[f"a_{h_num}"] = round(
                float(self.a_coeffs[idx]), ROUND_DECIMALS_COEFFS
            )
            flat_dict[f"b_{h_num}"] = round(
                float(self.b_coeffs[idx]), ROUND_DECIMALS_COEFFS
            )
            flat_dict[f"c_{h_num}"] = round(
                float(self.c_coeffs[idx]), ROUND_DECIMALS_COEFFS
            )
            flat_dict[f"d_{h_num}"] = round(
                float(self.d_coeffs[idx]), ROUND_DECIMALS_COEFFS
            )
        return flat_dict


@dataclass(frozen=True, slots=True)
class ApparelItem:
    """Immutable representation of an extracted apparel or swimsuit item."""

    item_id: int
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    eccentricity: float
    solidity: float
    extent: float
    fourier_coefficients: EllipticalFourierCoefficients

    def __post_init__(self) -> None:
        """Validates apparel item geometric properties upon initialization."""
        verify_condition(
            int(self.item_id) >= ZERO_INT,
            "Item ID must be non-negative.",
        )
        verify_condition(
            float(self.area) > ZERO_FLOAT,
            "Area must be strictly positive.",
        )
        verify_condition(
            float(self.perimeter) >= ZERO_FLOAT,
            "Perimeter cannot be negative.",
        )
        verify_condition(
            len(self.centroid) == CENTROID_TUPLE_LENGTH,
            "Centroid must be a 2-tuple of (y, x).",
        )
        verify_condition(
            len(self.bounding_box) == BBOX_TUPLE_LENGTH,
            "Bounding box must have 4 dimensions (x, y, width, height).",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts apparel metrics and EFD coefficients into a flat dict.

        Returns:
            Dict[str, Any]: Flattened representation of item properties.
        """
        base_metrics: Dict[str, Any] = {
            "item_id": int(self.item_id),
            "area": round(float(self.area), ROUND_DECIMALS_METRICS),
            "perimeter": round(float(self.perimeter), ROUND_DECIMALS_METRICS),
            "centroid_y": round(
                float(self.centroid[ZERO_INT]), ROUND_DECIMALS_METRICS
            ),
            "centroid_x": round(
                float(self.centroid[ONE_INT]), ROUND_DECIMALS_METRICS
            ),
            "bbox_x": int(self.bounding_box[ZERO_INT]),
            "bbox_y": int(self.bounding_box[ONE_INT]),
            "bbox_w": int(self.bounding_box[2]),
            "bbox_h": int(self.bounding_box[3]),
            "eccentricity": round(
                float(self.eccentricity), ROUND_DECIMALS_SHAPES
            ),
            "solidity": round(float(self.solidity), ROUND_DECIMALS_SHAPES),
            "extent": round(float(self.extent), ROUND_DECIMALS_SHAPES),
        }
        return {**base_metrics, **self.fourier_coefficients.to_dict()}


@dataclass(frozen=True, slots=True)
class CandidateContourRegion:
    """Immutable intermediate region carrier between segmentation and EFD."""

    item_id: int
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    eccentricity: float
    solidity: float
    extent: float
    contour: np.ndarray


@dataclass(frozen=True, slots=True)
class PipelineBatchState:
    """Immutable batch state passed through the declarative stage registry."""

    rgb_image: np.ndarray
    labeled_mask: Optional[np.ndarray] = None
    candidate_regions: Tuple[CandidateContourRegion, ...] = field(
        default_factory=tuple
    )
    extracted_items: Tuple[ApparelItem, ...] = field(default_factory=tuple)
    completed_stages: Tuple[str, ...] = field(default_factory=tuple)


class ApparelCollection:
    """Sequence container supporting dunder protocols for ApparelItem."""

    def __init__(self, items: List[ApparelItem]) -> None:
        """Initializes ApparelCollection with a validated list of items.

        Args:
            items (List[ApparelItem]): Extracted apparel items.
        """
        verify_condition(
            isinstance(items, list),
            "Items must be provided as a list.",
        )
        verify_condition(
            len(items) <= MAX_ITEMS_CEILING,
            "Item count exceeds collection ceiling.",
        )
        self._items: Tuple[ApparelItem, ...] = tuple(items)

    def __len__(self) -> int:
        """Returns the number of items in the collection."""
        return len(self._items)

    def __getitem__(self, index: int) -> ApparelItem:
        """Returns the ApparelItem at the specified index."""
        verify_condition(
            ZERO_INT <= index < len(self._items),
            "Collection index out of bounds.",
        )
        return self._items[index]

    def __iter__(self) -> Iterator[ApparelItem]:
        """Returns an iterator over the collection items."""
        return iter(self._items)

    def __repr__(self) -> str:
        """Returns a developer-friendly string representation."""
        return f"<ApparelCollection(count={len(self._items)})>"

    def __bool__(self) -> bool:
        """Returns True if the collection contains at least one item."""
        return len(self._items) > ZERO_INT

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the collection of items into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing metrics for all items.
        """
        records = [item.to_dict() for item in self._items]
        return pd.DataFrame(records)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Exports all items as a list of dictionaries for JSON responses.

        Returns:
            List[Dict[str, Any]]: List of dictionary representations.
        """
        return [item.to_dict() for item in self._items]