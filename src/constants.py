"""Centralized constants and runtime verification helpers for Src."""

from typing import Tuple


PACKAGE_VERSION: str = "0.2.0"
LOGGER_NAME: str = "src"
DEFAULT_CONFIG_FILENAME: str = "config.yaml"
DEFAULT_MASTER_KEY_ENV: str = "SRC_MASTER_SECRET"
FALLBACK_MASTER_KEY_HEX: str = (
    "4f6a9b2c8d1e3f5a7b9c0d2e4f6a8b1c3d5e7f9a0b2c4d6e8f0a1b3c5d7e9f1a"
)

ZERO_INT: int = 0
ONE_INT: int = 1
TWO_INT: int = 2
THREE_INT: int = 3
FOUR_INT: int = 4
SIX_INT: int = 6
EIGHT_INT: int = 8
TEN_INT: int = 10
SIXTEEN_INT: int = 16
THIRTY_TWO_INT: int = 32
SIXTY_FOUR_INT: int = 64

ZERO_FLOAT: float = 0.0
ONE_FLOAT: float = 1.0
TWO_FLOAT: float = 2.0
HALF_FLOAT: float = 0.5
DEFAULT_EPSILON: float = 1e-10
MAX_DP_EPSILON: float = 1.0
MIN_DP_EPSILON: float = 0.01

ROUND_DECIMALS_METRICS: int = 2
ROUND_DECIMALS_SHAPES: int = 4
ROUND_DECIMALS_COEFFS: int = 6

MAX_HARMONICS_CEILING: int = 100
MAX_CONTOUR_POINTS_CEILING: int = 100000
MIN_CONTOUR_POINTS_FLOOR: int = 3
MAX_ITEMS_CEILING: int = 1000
MAX_STAGES_CEILING: int = 16
MAX_FILE_BYTES_CEILING: int = 10485760
MAX_IMAGE_DIMENSION_CEILING: int = 8192
MAX_IMAGE_PIXELS_CEILING: int = 67108864
MAX_PATH_LENGTH: int = 512
MAX_LOG_RING_CAPACITY: int = 2048
DEFAULT_LOG_RING_CAPACITY: int = 256
DEFAULT_LOG_DETAIL_LIMIT: int = 180
MAX_ALERT_SINK_CAPACITY: int = 512
MAX_LEDGER_SCAN_LINES: int = 10000
MAX_RATE_LIMIT_TRACKED_CLIENTS: int = 4096
MAX_RATE_LIMIT_EVENTS_PER_CLIENT: int = 512
MAX_RESET_TOKENS_TRACKED: int = 2048
PASSWORD_RESET_TTL_SECONDS: int = 1800
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60
DEFAULT_RATE_LIMIT_MAX_REQUESTS: int = 30
DEFAULT_RETENTION_TTL_SECONDS: int = 86400
HKDF_HASH_LENGTH: int = 32
HKDF_MAX_BLOCKS: int = 16
PSEUDONYM_TOKEN_HEX_LENGTH: int = 32

H3_BASE_SCALE_DEG: float = 0.01
H3_APERTURE_FACTOR: float = 2.6457513110645907
H3_MAX_COARSEN_STEPS: int = 12
DEFAULT_K_ANONYMITY: int = 3
MIN_BLUR_BLOCK_SIZE: int = 4
MAX_BLUR_KERNEL_STEPS: int = 64

GRAY_MAX_VALUE: int = 255
RGB_CHANNELS: int = 3
GRAY_DIMENSIONS: int = 2
BBOX_TUPLE_LENGTH: int = 4
CENTROID_TUPLE_LENGTH: int = 2
CONTOUR_COORD_DIM: int = 2

SYNTHETIC_WIDTH: int = 800
SYNTHETIC_HEIGHT: int = 600
SYNTHETIC_BG_VALUE: int = 255
SYNTHETIC_TOP_LEFT_X: int = 200
SYNTHETIC_TOP_RIGHT_X: int = 320
SYNTHETIC_TOP_Y: int = 300
SYNTHETIC_TOP_RADIUS: int = 60
SYNTHETIC_BRIDGE_X1: int = 220
SYNTHETIC_BRIDGE_Y1: int = 280
SYNTHETIC_BRIDGE_X2: int = 300
SYNTHETIC_BRIDGE_Y2: int = 320
SYNTHETIC_TOP_COLOR: Tuple[int, int, int] = (20, 20, 20)
SYNTHETIC_BOTTOM_PT1: Tuple[int, int] = (500, 250)
SYNTHETIC_BOTTOM_PT2: Tuple[int, int] = (650, 250)
SYNTHETIC_BOTTOM_PT3: Tuple[int, int] = (575, 400)
SYNTHETIC_BOTTOM_COLOR: Tuple[int, int, int] = (30, 30, 30)
FILLED_CONTOUR_THICKNESS: int = -1

HTTP_STATUS_OK: int = 200
HTTP_STATUS_BAD_REQUEST: int = 400
HTTP_STATUS_FORBIDDEN: int = 403
HTTP_STATUS_NOT_FOUND: int = 404
HTTP_STATUS_GONE: int = 410
HTTP_STATUS_TOO_MANY_REQUESTS: int = 429
HTTP_STATUS_INTERNAL_ERROR: int = 500

PNG_MAGIC_BYTES: bytes = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC_BYTES: bytes = b"\xff\xd8\xff"
WEBP_RIFF_BYTES: bytes = b"RIFF"
WEBP_FOURCC_BYTES: bytes = b"WEBP"
WEBP_HEADER_MIN_LEN: int = 12

XML_FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "<!DOCTYPE",
    "<!ENTITY",
    "SYSTEM",
    "PUBLIC",
)

SQLI_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "UNION SELECT",
    "DROP TABLE",
    "OR 1=1",
    "OR '1'='1'",
    "--",
    "/*",
    "*/",
    ";SELECT",
    ";DROP",
)

XSS_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "<SCRIPT",
    "</SCRIPT",
    "JAVASCRIPT:",
    "ONERROR=",
    "ONLOAD=",
    "<IFRAME",
    "<SVG",
)

EMAIL_REGEX_PATTERN: str = (
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)
PHONE_REGEX_PATTERN: str = r"\+?[0-9][0-9\-\s]{7,14}[0-9]"
IPV4_REGEX_PATTERN: str = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
REDACTED_PLACEHOLDER: str = "[REDACTED_PII]"

DEPLOYMENT_SLOT_BLUE: str = "blue"
DEPLOYMENT_SLOT_GREEN: str = "green"
VALID_DEPLOYMENT_SLOTS: Tuple[str, str] = (
    DEPLOYMENT_SLOT_BLUE,
    DEPLOYMENT_SLOT_GREEN,
)


class SrcVerificationError(ValueError):
    """Raised when a runtime precondition or postcondition check fails."""


class SrcSecurityError(PermissionError):
    """Raised when an OWASP ASVS or pre-deployment security check fails."""


def verify_condition(
    condition: bool,
    message: str,
    use_security_error: bool = False,
) -> None:
    """Verifies a runtime invariant without relying on Python assert statements.

    Args:
        condition (bool): Boolean condition that must evaluate to True.
        message (str): Descriptive error message if condition is False.
        use_security_error (bool, optional): Raise SrcSecurityError if
            True, otherwise SrcVerificationError. Defaults to False.

    Raises:
        SrcSecurityError: If condition is False and use_security_error.
        SrcVerificationError: If condition is False.
    """
    if not condition:
        if use_security_error:
            raise SrcSecurityError(message)
        raise SrcVerificationError(message)