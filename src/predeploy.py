"""Ten-point pre-deployment operational readiness engine for Src."""

from contextlib import closing
from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from src.constants import (
    SrcSecurityError,
    DEFAULT_MASTER_KEY_ENV,
    DEFAULT_RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    DEPLOYMENT_SLOT_BLUE,
    DEPLOYMENT_SLOT_GREEN,
    FALLBACK_MASTER_KEY_HEX,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_GONE,
    HTTP_STATUS_INTERNAL_ERROR,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_TOO_MANY_REQUESTS,
    MAX_ITEMS_CEILING,
    MAX_PATH_LENGTH,
    MAX_RATE_LIMIT_EVENTS_PER_CLIENT,
    MAX_RATE_LIMIT_TRACKED_CLIENTS,
    MAX_RESET_TOKENS_TRACKED,
    ONE_INT,
    PASSWORD_RESET_TTL_SECONDS,
    ROUND_DECIMALS_SHAPES,
    SIXTEEN_INT,
    SQLI_FORBIDDEN_PATTERNS,
    VALID_DEPLOYMENT_SLOTS,
    XSS_FORBIDDEN_PATTERNS,
    ZERO_INT,
    verify_condition,
)
from src.logging_config import get_logger
from src.security import resolve_safe_path


@dataclass(frozen=True, slots=True)
class PreDeployHTTPException(Exception):
    """Structured HTTP error carrying a status code and safe client payload."""

    status_code: int
    error_code: str
    user_message: str


def verify_bola_ownership(
    authenticated_user_id: str, resource_owner_id: str
) -> bool:
    """Verifies Object-Level Authorization via constant-time HMAC comparison.

    Args:
        authenticated_user_id (str): Token or ID of the requesting user.
        resource_owner_id (str): Owner token bound to the target resource.

    Returns:
        bool: True if the caller owns the resource.

    Raises:
        PreDeployHTTPException: HTTP 403 if ownership verification fails.
    """
    if (
        not isinstance(authenticated_user_id, str)
        or not isinstance(resource_owner_id, str)
        or len(authenticated_user_id.strip()) == ZERO_INT
        or len(resource_owner_id.strip()) == ZERO_INT
    ):
        raise PreDeployHTTPException(
            status_code=HTTP_STATUS_FORBIDDEN,
            error_code="BOLA_ACCESS_DENIED",
            user_message="You do not have permission to access this resource.",
        )

    matched = hmac.compare_digest(
        authenticated_user_id.strip().encode("utf-8"),
        resource_owner_id.strip().encode("utf-8"),
    )
    if not matched:
        raise PreDeployHTTPException(
            status_code=HTTP_STATUS_FORBIDDEN,
            error_code="BOLA_ACCESS_DENIED",
            user_message="You do not have permission to access this resource.",
        )
    return True


def sanitize_and_validate_input(
    raw_value: str, max_length: int = MAX_PATH_LENGTH
) -> str:
    """Rejects SQLi and XSS patterns and HTML-escapes the validated string.

    Args:
        raw_value (str): Untrusted user string input.
        max_length (int, optional): Maximum allowed string length. Defaults to
            MAX_PATH_LENGTH.

    Returns:
        str: Sanitized and HTML-escaped string.

    Raises:
        PreDeployHTTPException: HTTP 400 if SQLi, XSS, or length check fails.
    """
    if (
        not isinstance(raw_value, str)
        or len(raw_value) == ZERO_INT
        or len(raw_value) > max_length
    ):
        raise PreDeployHTTPException(
            status_code=HTTP_STATUS_BAD_REQUEST,
            error_code="INVALID_INPUT_BOUNDS",
            user_message="Submitted input is empty or exceeds size bounds.",
        )

    upper_val = raw_value.upper()
    for idx in range(len(SQLI_FORBIDDEN_PATTERNS)):
        pattern = SQLI_FORBIDDEN_PATTERNS[idx]
        if pattern in upper_val:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="SQLI_PATTERN_BLOCKED",
                user_message="Input contains disallowed query tokens.",
            )

    for idx in range(len(XSS_FORBIDDEN_PATTERNS)):
        pattern = XSS_FORBIDDEN_PATTERNS[idx]
        if pattern in upper_val or "<" in raw_value or ">" in raw_value:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="XSS_PATTERN_BLOCKED",
                user_message="Input contains disallowed markup characters.",
            )

    return html.escape(raw_value.strip(), quote=True)


def verify_cors_origin(
    request_origin: Optional[str], allowed_origins: Sequence[str]
) -> str:
    """Verifies that a request Origin matches the strict CORS allowlist.

    Args:
        request_origin (Optional[str]): Origin header value from request.
        allowed_origins (Sequence[str]): Configured allowed CORS origins.

    Returns:
        str: Validated origin string.

    Raises:
        PreDeployHTTPException: HTTP 403 if origin is not in allowlist.
    """
    if request_origin is None or len(request_origin.strip()) == ZERO_INT:
        raise PreDeployHTTPException(
            status_code=HTTP_STATUS_FORBIDDEN,
            error_code="CORS_ORIGIN_MISSING",
            user_message="Origin header is required by strict CORS policy.",
        )

    clean_origin = request_origin.strip()
    for idx in range(min(len(allowed_origins), MAX_ITEMS_CEILING)):
        allowed = allowed_origins[idx].strip()
        if allowed != "*" and hmac.compare_digest(clean_origin, allowed):
            return clean_origin

    raise PreDeployHTTPException(
        status_code=HTTP_STATUS_FORBIDDEN,
        error_code="CORS_ORIGIN_REJECTED",
        user_message="Origin is not permitted by strict CORS policy.",
    )


class SlidingWindowRateLimiter:
    """Per-client sliding-window request rate limiter preventing bill spikes."""

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_MAX_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        """Initializes rate limiter parameters.

        Args:
            max_requests (int, optional): Max requests per window. Defaults to
                DEFAULT_RATE_LIMIT_MAX_REQUESTS.
            window_seconds (int, optional): Sliding window in seconds.
                Defaults to DEFAULT_RATE_LIMIT_WINDOW_SECONDS.
        """
        verify_condition(
            ONE_INT <= int(max_requests) <= MAX_RATE_LIMIT_EVENTS_PER_CLIENT,
            "Rate limit max_requests is outside allowed bounds.",
        )
        verify_condition(
             int(window_seconds) >= ONE_INT,
            "Rate limit window_seconds must be positive.",
        )
        self._max_requests = int(max_requests)
        self._window_seconds = float(window_seconds)
        self._client_buckets: Dict[str, Deque[float]] = {}

    def check_and_record(
        self, client_id: str, current_time: Optional[float] = None
    ) -> int:
        """Checks if client is within rate limit and records the request.

        Args:
            client_id (str): Client identifier or IP pseudonym.
            current_time (Optional[float], optional): Reference epoch seconds.

        Returns:
            int: Remaining requests in current sliding window.

        Raises:
            PreDeployHTTPException: HTTP 429 when rate limit is exceeded.
        """
        verify_condition(
            isinstance(client_id, str) and len(client_id.strip()) > ZERO_INT,
            "Client ID for rate limiting must be non-empty.",
        )
        now = current_time if current_time is not None else time.time()
        cutoff = now - self._window_seconds

        if (
            client_id not in self._client_buckets
            and len(self._client_buckets) >= MAX_RATE_LIMIT_TRACKED_CLIENTS
        ):
            oldest_key = next(iter(self._client_buckets))
            del self._client_buckets[oldest_key]

        bucket = self._client_buckets.setdefault(
            client_id, deque(maxlen=MAX_RATE_LIMIT_EVENTS_PER_CLIENT)
        )
        steps = min(len(bucket), MAX_RATE_LIMIT_EVENTS_PER_CLIENT)
        for _ in range(steps):
            if len(bucket) > ZERO_INT and bucket[ZERO_INT] <= cutoff:
                bucket.popleft()
            else:
                break

        if len(bucket) >= self._max_requests:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_TOO_MANY_REQUESTS,
                error_code="RATE_LIMIT_EXCEEDED",
                user_message="Rate limit exceeded. Please wait before retrying.",
            )

        bucket.append(now)
        return self._max_requests - len(bucket)


class PasswordResetTokenManager:
    """Issues and consumes single-use 30-minute expiring reset tokens."""

    def __init__(
        self, ttl_seconds: int = PASSWORD_RESET_TTL_SECONDS
    ) -> None:
        """Initializes the password reset token manager.

        Args:
            ttl_seconds (int, optional): Token lifetime in seconds. Must not
                exceed PASSWORD_RESET_TTL_SECONDS (1800s).
        """
        verify_condition(
            ONE_INT <= int(ttl_seconds) <= PASSWORD_RESET_TTL_SECONDS,
            "Password reset TTL must be between 1 and 1800 seconds.",
            use_security_error=True,
        )
        self._ttl_seconds = float(ttl_seconds)
        raw_secret = os.environ.get(
            DEFAULT_MASTER_KEY_ENV, FALLBACK_MASTER_KEY_HEX
        )
        self._secret = raw_secret.encode("utf-8")
        self._issued_records: Dict[str, Tuple[str, float, bool]] = {}

    def issue_reset_token(
        self, user_id: str, current_time: Optional[float] = None
    ) -> str:
        """Issues a signed, single-use password reset token valid for 30 mins.

        Args:
            user_id (str): Target account identifier.
            current_time (Optional[float], optional): Reference epoch seconds.

        Returns:
            str: Signed reset token string.
        """
        clean_user = sanitize_and_validate_input(user_id)
        now = current_time if current_time is not None else time.time()
        expires_at = now + self._ttl_seconds
        nonce = secrets.token_hex(SIXTEEN_INT)
        payload = f"{clean_user}:{nonce}:{round(expires_at, ROUND_DECIMALS_SHAPES)}"
        sig = hmac.new(
            self._secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        token = f"{payload}:{sig}"

        if len(self._issued_records) >= MAX_RESET_TOKENS_TRACKED:
            oldest_token = next(iter(self._issued_records))
            del self._issued_records[oldest_token]

        self._issued_records[token] = (clean_user, expires_at, False)
        return token

    def consume_reset_token(
        self, token: str, current_time: Optional[float] = None
    ) -> str:
        """Validates and consumes a single-use password reset token.

        Args:
            token (str): Reset token string presented by user.
            current_time (Optional[float], optional): Reference epoch seconds.

        Returns:
            str: User ID associated with the valid reset token.

        Raises:
            PreDeployHTTPException: HTTP 410 if expired/reused, 400 if invalid.
        """
        if not isinstance(token, str) or ":" not in token:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="RESET_TOKEN_MALFORMED",
                user_message="Password reset link is invalid.",
            )

        parts = token.rsplit(":", ONE_INT)
        if len(parts) != 2:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="RESET_TOKEN_MALFORMED",
                user_message="Password reset link is invalid.",
            )

        payload, provided_sig = parts[ZERO_INT], parts[ONE_INT]
        expected_sig = hmac.new(
            self._secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(provided_sig, expected_sig):
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="RESET_TOKEN_SIGNATURE_INVALID",
                user_message="Password reset signature verification failed.",
            )

        record = self._issued_records.get(token)
        if record is None:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_GONE,
                error_code="RESET_TOKEN_UNKNOWN",
                user_message="Password reset link is no longer valid.",
            )

        user_id, expires_at, already_used = record
        now = current_time if current_time is not None else time.time()
        if already_used or now > expires_at:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_GONE,
                error_code="RESET_TOKEN_EXPIRED_OR_USED",
                user_message="Password reset link has expired or was already used.",
            )

        self._issued_records[token] = (user_id, expires_at, True)
        return user_id


ERROR_SCREEN_TITLES: Dict[int, str] = {
    HTTP_STATUS_BAD_REQUEST: "400 - Invalid Request Payload",
    HTTP_STATUS_FORBIDDEN: "403 - Access Denied by Security Policy",
    HTTP_STATUS_NOT_FOUND: "404 - Requested Resource Not Found",
    HTTP_STATUS_GONE: "410 - Expiring Link No Longer Available",
    HTTP_STATUS_TOO_MANY_REQUESTS: "429 - Request Rate Limit Exceeded",
    HTTP_STATUS_INTERNAL_ERROR: "500 - Internal Service Boundary Guard",
}


def build_error_screen_payload(
    status_code: int,
    error_code: str,
    user_message: str,
) -> Dict[str, Any]:
    """Builds a structured, stack-trace-free error screen dictionary.

    Args:
        status_code (int): HTTP status code (400, 403, 404, 410, 429, 500).
        error_code (str): Machine-readable error code identifier.
        user_message (str): Safe, user-friendly explanation message.

    Returns:
        Dict[str, Any]: Sanitized error screen response model.
    """
    title = ERROR_SCREEN_TITLES.get(
        int(status_code), ERROR_SCREEN_TITLES[HTTP_STATUS_INTERNAL_ERROR]
    )
    return {
        "success": False,
        "status_code": int(status_code),
        "error_code": str(error_code),
        "screen_title": title,
        "user_message": str(user_message),
        "stack_trace_suppressed": True,
    }


def execute_with_catch_all_boundary(
    operation: Callable[[], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
    """Executes an endpoint callable inside a global 500 catch-all boundary.

    Args:
        operation (Callable[[], Dict[str, Any]]): Zero-argument callable.

    Returns:
        Tuple[int, Dict[str, Any]]: HTTP status code and response dictionary.
    """
    logger = get_logger()
    try:
        result = operation()
        return HTTP_STATUS_OK, result
    except PreDeployHTTPException as exc:
        logger.warning(
            f"PreDeploy guard triggered [{exc.status_code}]: {exc.error_code}"
        )
        return exc.status_code, build_error_screen_payload(
            status_code=exc.status_code,
            error_code=exc.error_code,
            user_message=exc.user_message,
        )
    except SrcSecurityError as sec_exc:
        logger.warning(f"Security policy violation: {sec_exc}")
        return HTTP_STATUS_BAD_REQUEST, build_error_screen_payload(
            status_code=HTTP_STATUS_BAD_REQUEST,
            error_code="SECURITY_VALIDATION_FAILED",
            user_message="Request failed security validation checks.",
        )
    except Exception as unexpected_exc:
        logger.error(
            f"Unhandled boundary exception: {type(unexpected_exc).__name__}"
        )
        return HTTP_STATUS_INTERNAL_ERROR, build_error_screen_payload(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            error_code="INTERNAL_BOUNDARY_CAUGHT",
            user_message="An unexpected error occurred. Engineers were alerted.",
        )


class IndexedBatchStore:
    """SQLite persistence layer with targeted B-tree indexes on key columns."""

    def __init__(
        self,
        db_path: str | Path = "src_records.sqlite3",
        workspace_root: Optional[str | Path] = None,
    ) -> None:
        """Initializes SQLite store and creates targeted B-tree indexes.

        Args:
            db_path (str | Path, optional): Path to SQLite database file.
            workspace_root (Optional[str | Path], optional): Workspace root.
        """
        self._db_path = resolve_safe_path(
            db_path, workspace_root=workspace_root, must_exist=False
        )
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Creates the apparel_batches table and targeted B-tree indexes."""
        with closing(sqlite3.connect(str(self._db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS apparel_batches (
                    batch_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    item_count INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_apparel_batches_owner_id
                ON apparel_batches (owner_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_apparel_batches_batch_id
                ON apparel_batches (batch_id)
                """
            )
            conn.commit()

    def save_batch(
        self,
        batch_id: str,
        owner_id: str,
        items_payload: List[Dict[str, Any]],
    ) -> None:
        """Persists an extracted batch using parameterized SQL queries.

        Args:
            batch_id (str): Unique pseudonymised batch identifier.
            owner_id (str): Pseudonymised owner token.
            items_payload (List[Dict[str, Any]]): Extracted item dictionaries.
        """
        clean_batch = sanitize_and_validate_input(batch_id)
        clean_owner = sanitize_and_validate_input(owner_id)
        serialized = json.dumps(items_payload, sort_keys=True)

        with closing(sqlite3.connect(str(self._db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO apparel_batches
                (batch_id, owner_id, created_at, item_count, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    clean_batch,
                    clean_owner,
                    round(time.time(), ROUND_DECIMALS_SHAPES),
                    len(items_payload),
                    serialized,
                ),
            )
            conn.commit()

    def fetch_batch_for_owner(
        self, batch_id: str, requesting_user_id: str
    ) -> Dict[str, Any]:
        """Retrieves a batch after verifying BOLA ownership.

        Args:
            batch_id (str): Target batch identifier.
            requesting_user_id (str): Authenticated caller token.

        Returns:
            Dict[str, Any]: Stored batch record dictionary.

        Raises:
            PreDeployHTTPException: 404 if missing or 403 on BOLA mismatch.
        """
        clean_batch = sanitize_and_validate_input(batch_id)
        with closing(sqlite3.connect(str(self._db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT batch_id, owner_id, created_at, item_count, payload_json
                FROM apparel_batches
                WHERE batch_id = ?
                """,
                (clean_batch,),
            )
            row = cursor.fetchone()

        if row is None:
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_NOT_FOUND,
                error_code="BATCH_NOT_FOUND",
                user_message="Requested apparel batch does not exist.",
            )

        stored_batch_id, stored_owner_id, created_at, item_count, raw_json = row
        verify_bola_ownership(requesting_user_id, stored_owner_id)
        return {
            "batch_id": stored_batch_id,
            "owner_id": stored_owner_id,
            "created_at": created_at,
            "item_count": item_count,
            "items": json.loads(raw_json),
        }

    def list_index_names(self) -> List[str]:
        """Returns created SQLite B-tree index names for readiness verification.

        Returns:
            List[str]: List of index names on apparel_batches.
        """
        with closing(sqlite3.connect(str(self._db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA index_list('apparel_batches')")
            rows = cursor.fetchall()
        return [str(r[ONE_INT]) for r in rows[:MAX_ITEMS_CEILING]]


class BlueGreenDeploymentManager:
    """Manages blue and green release slots with instant rollback."""

    def __init__(
        self,
        initial_slot: str = DEPLOYMENT_SLOT_BLUE,
        initial_version: str = "0.2.0",
    ) -> None:
        """Initializes active and standby deployment slots.

        Args:
            initial_slot (str, optional): Active slot ('blue' or 'green').
            initial_version (str, optional): Version running in active slot.
        """
        verify_condition(
            initial_slot in VALID_DEPLOYMENT_SLOTS,
            "Initial deployment slot must be 'blue' or 'green'.",
        )
        self._active_slot = initial_slot
        standby = (
            DEPLOYMENT_SLOT_GREEN
            if initial_slot == DEPLOYMENT_SLOT_BLUE
            else DEPLOYMENT_SLOT_BLUE
        )
        self._slot_versions: Dict[str, str] = {
            initial_slot: initial_version,
            standby: initial_version,
        }
        self._slot_health: Dict[str, bool] = {
            DEPLOYMENT_SLOT_BLUE: True,
            DEPLOYMENT_SLOT_GREEN: True,
        }

    @property
    def active_slot(self) -> str:
        """Returns the currently active traffic slot name."""
        return self._active_slot

    @property
    def standby_slot(self) -> str:
        """Returns the currently inactive standby slot name."""
        return (
            DEPLOYMENT_SLOT_GREEN
            if self._active_slot == DEPLOYMENT_SLOT_BLUE
            else DEPLOYMENT_SLOT_BLUE
        )

    def deploy_and_promote(
        self,
        candidate_version: str,
        health_probe: Callable[[], bool],
    ) -> Dict[str, Any]:
        """Stages a release in standby slot and rolls back if probe fails.

        Args:
            candidate_version (str): Release version string to stage.
            health_probe (Callable[[], bool]): Health check callable.

        Returns:
            Dict[str, Any]: Deployment status and active slot details.
        """
        clean_version = sanitize_and_validate_input(candidate_version)
        target_slot = self.standby_slot
        previous_slot = self._active_slot
        self._slot_versions[target_slot] = clean_version

        probe_passed = False
        try:
            probe_passed = bool(health_probe())
        except Exception:
            probe_passed = False

        self._slot_health[target_slot] = probe_passed
        if probe_passed:
            self._active_slot = target_slot
            return {
                "promoted": True,
                "rolled_back": False,
                "active_slot": self._active_slot,
                "active_version": self._slot_versions[self._active_slot],
            }

        self._active_slot = previous_slot
        get_logger().error(
            f"Blue-green health probe failed on {target_slot}; "
            f"instant rollback to {previous_slot} executed."
        )
        return {
            "promoted": False,
            "rolled_back": True,
            "active_slot": self._active_slot,
            "active_version": self._slot_versions[self._active_slot],
        }

    def instant_rollback(self) -> Dict[str, Any]:
        """Immediately switches traffic back to the healthy standby slot.

        Returns:
            Dict[str, Any]: Updated slot status after rollback.
        """
        target = self.standby_slot
        self._active_slot = target
        return {
            "rolled_back": True,
            "active_slot": self._active_slot,
            "active_version": self._slot_versions[self._active_slot],
        }