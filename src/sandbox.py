"""Local Azurite and Fake-GCS sandbox adapter with zero-PII synthesis."""

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.constants import (
    FILLED_CONTOUR_THICKNESS,
    RGB_CHANNELS,
    SYNTHETIC_BG_VALUE,
    SYNTHETIC_BOTTOM_COLOR,
    SYNTHETIC_BOTTOM_PT1,
    SYNTHETIC_BOTTOM_PT2,
    SYNTHETIC_BOTTOM_PT3,
    SYNTHETIC_BRIDGE_X1,
    SYNTHETIC_BRIDGE_X2,
    SYNTHETIC_BRIDGE_Y1,
    SYNTHETIC_BRIDGE_Y2,
    SYNTHETIC_HEIGHT,
    SYNTHETIC_TOP_COLOR,
    SYNTHETIC_TOP_LEFT_X,
    SYNTHETIC_TOP_RADIUS,
    SYNTHETIC_TOP_RIGHT_X,
    SYNTHETIC_TOP_Y,
    SYNTHETIC_WIDTH,
    ZERO_INT,
    verify_condition,
)
from src.security import validate_outbound_url


def generate_synthetic_swimsuits_image(
    width: int = SYNTHETIC_WIDTH,
    height: int = SYNTHETIC_HEIGHT,
) -> np.ndarray:
    """Generates a deterministic, zero-PII RGB canvas with swimsuit shapes.

    Args:
        width (int, optional): Canvas pixel width. Defaults to SYNTHETIC_WIDTH.
        height (int, optional): Canvas pixel height. Defaults to
            SYNTHETIC_HEIGHT.

    Returns:
        np.ndarray: Synthetic RGB image array of shape (height, width, 3).
    """
    verify_condition(
        int(width) > ZERO_INT and int(height) > ZERO_INT,
        "Synthetic canvas dimensions must be strictly positive.",
    )
    canvas = (
        np.ones((int(height), int(width), RGB_CHANNELS), dtype=np.uint8)
        * SYNTHETIC_BG_VALUE
    )

    cv2.circle(
        canvas,
        (SYNTHETIC_TOP_LEFT_X, SYNTHETIC_TOP_Y),
        SYNTHETIC_TOP_RADIUS,
        SYNTHETIC_TOP_COLOR,
        FILLED_CONTOUR_THICKNESS,
    )
    cv2.circle(
        canvas,
        (SYNTHETIC_TOP_RIGHT_X, SYNTHETIC_TOP_Y),
        SYNTHETIC_TOP_RADIUS,
        SYNTHETIC_TOP_COLOR,
        FILLED_CONTOUR_THICKNESS,
    )
    cv2.rectangle(
        canvas,
        (SYNTHETIC_BRIDGE_X1, SYNTHETIC_BRIDGE_Y1),
        (SYNTHETIC_BRIDGE_X2, SYNTHETIC_BRIDGE_Y2),
        SYNTHETIC_TOP_COLOR,
        FILLED_CONTOUR_THICKNESS,
    )

    pts = np.array(
        [SYNTHETIC_BOTTOM_PT1, SYNTHETIC_BOTTOM_PT2, SYNTHETIC_BOTTOM_PT3],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [pts], SYNTHETIC_BOTTOM_COLOR)
    return canvas


def generate_synthetic_telemetry_coords() -> List[Tuple[float, float]]:
    """Generates zero-PII synthetic UK/EU coordinates for H3 DP testing.

    Returns:
        List[Tuple[float, float]]: Synthetic (latitude, longitude) pairs.
    """
    return [
        (52.477, 1.750),
        (52.478, 1.751),
        (52.477, 1.749),
        (52.476, 1.750),
        (51.507, -0.127),
        (51.508, -0.128),
        (51.507, -0.126),
    ]


class CloudSandboxAdapter:
    """Adapter for Azurite and Fake-GCS loopback storage with fallback."""

    def __init__(
        self,
        azurite_endpoint: str = "http://127.0.0.1:10000",
        gcs_endpoint: str = "http://127.0.0.1:4443",
        offline_fallback: bool = True,
        allowed_hosts: Sequence[str] = ("127.0.0.1", "localhost"),
    ) -> None:
        """Initializes and SSRF-validates local cloud sandbox endpoints.

        Args:
            azurite_endpoint (str, optional): Local Azurite URL.
            gcs_endpoint (str, optional): Local fake-gcs-server URL.
            offline_fallback (bool, optional): Enable in-memory fallback.
            allowed_hosts (Sequence[str], optional): Allowed sandbox hosts.
        """
        self._azurite_url = validate_outbound_url(
            azurite_endpoint, allowed_hosts=allowed_hosts
        )
        self._gcs_url = validate_outbound_url(
            gcs_endpoint, allowed_hosts=allowed_hosts
        )
        self._offline_fallback = bool(offline_fallback)
        self._memory_buckets: Dict[str, np.ndarray] = {}

    def put_sandbox_image(
        self, object_key: str, image_array: np.ndarray
    ) -> str:
        """Stores a clean image array in the local sandbox memory bucket.

        Args:
            object_key (str): Pseudonymised storage key.
            image_array (np.ndarray): Validated RGB image array.

        Returns:
            str: Sandbox URI referencing the stored object.
        """
        verify_condition(
            isinstance(object_key, str) and len(object_key) > ZERO_INT,
            "Sandbox object key must be non-empty.",
        )
        verify_condition(
            isinstance(image_array, np.ndarray) and image_array.size > ZERO_INT,
            "Sandbox image array must be non-empty.",
        )
        self._memory_buckets[object_key] = image_array.copy()
        return f"{self._gcs_url}/sandbox-bucket/{object_key}"

    def get_sandbox_image(
        self, object_key: Optional[str] = None
    ) -> np.ndarray:
        """Retrieves an image from sandbox storage or synthesizes fallback.

        Args:
            object_key (Optional[str], optional): Object key to retrieve.

        Returns:
            np.ndarray: Retrieved or synthesized RGB image array.
        """
        if object_key is not None and object_key in self._memory_buckets:
            return self._memory_buckets[object_key].copy()
        verify_condition(
            self._offline_fallback,
            "Requested sandbox object missing and offline fallback disabled.",
        )
        return generate_synthetic_swimsuits_image()