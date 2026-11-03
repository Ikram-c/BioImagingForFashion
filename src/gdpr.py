"""UK/EU GDPR privacy-by-design engine for the Src pipeline."""

from contextlib import closing
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.constants import (
    BBOX_TUPLE_LENGTH,
    DEFAULT_K_ANONYMITY,
    DEFAULT_MASTER_KEY_ENV,
    DEFAULT_RETENTION_TTL_SECONDS,
    EMAIL_REGEX_PATTERN,
    FALLBACK_MASTER_KEY_HEX,
    GRAY_DIMENSIONS,
    H3_APERTURE_FACTOR,
    H3_BASE_SCALE_DEG,
    H3_MAX_COARSEN_STEPS,
    HALF_FLOAT,
    HKDF_HASH_LENGTH,
    HKDF_MAX_BLOCKS,
    IPV4_REGEX_PATTERN,
    MAX_BLUR_KERNEL_STEPS,
    MAX_DP_EPSILON,
    MAX_ITEMS_CEILING,
    MAX_LEDGER_SCAN_LINES,
    MIN_BLUR_BLOCK_SIZE,
    MIN_DP_EPSILON,
    ONE_FLOAT,
    ONE_INT,
    PHONE_REGEX_PATTERN,
    PSEUDONYM_TOKEN_HEX_LENGTH,
    REDACTED_PLACEHOLDER,
    RGB_CHANNELS,
    ROUND_DECIMALS_SHAPES,
    TWO_FLOAT,
    ZERO_FLOAT,
    ZERO_INT,
    verify_condition,
)
from src.security import resolve_safe_path


EMAIL_REGEX = re.compile(EMAIL_REGEX_PATTERN)
PHONE_REGEX = re.compile(PHONE_REGEX_PATTERN)
IPV4_REGEX = re.compile(IPV4_REGEX_PATTERN)


def hkdf_sha256(
    input_key_material: bytes,
    salt: bytes,
    info: bytes,
    length: int = HKDF_HASH_LENGTH,
) -> bytes:
    """Derives cryptographic subkeys using RFC 5869 HKDF-SHA256.

    Args:
        input_key_material (bytes): Master secret key material.
        salt (bytes): Non-secret random or epoch salt bytes.
        info (bytes): Domain separation context bytes.
        length (int, optional): Output byte length. Defaults to
            HKDF_HASH_LENGTH.

    Returns:
        bytes: Derived subkey of requested length.
    """
    verify_condition(
        ONE_INT <= length <= (HKDF_HASH_LENGTH * HKDF_MAX_BLOCKS),
        "HKDF output length is outside allowed bounds.",
        use_security_error=True,
    )
    actual_salt = salt if len(salt) > ZERO_INT else bytes(HKDF_HASH_LENGTH)
    prk = hmac.new(actual_salt, input_key_material, hashlib.sha256).digest()
    blocks_needed = math.ceil(length / HKDF_HASH_LENGTH)
    verify_condition(
        blocks_needed <= HKDF_MAX_BLOCKS,
        "HKDF block count exceeds ceiling.",
        use_security_error=True,
    )

    okm = bytearray()
    previous_block = b""
    for block_idx in range(ONE_INT, blocks_needed + ONE_INT):
        hmac_input = previous_block + info + bytes([block_idx])
        previous_block = hmac.new(prk, hmac_input, hashlib.sha256).digest()
        okm.extend(previous_block)

    return bytes(okm[:length])


def scrub_pii_text(text: str) -> str:
    """Redacts email addresses, phone numbers, and IPv4 addresses from text.

    Args:
        text (str): Input string that may contain personal identifiers.

    Returns:
        str: Sanitized string with PII replaced by REDACTED_PLACEHOLDER.
    """
    if not isinstance(text, str) or len(text) == ZERO_INT:
        return ""
    scrubbed = EMAIL_REGEX.sub(REDACTED_PLACEHOLDER, text)
    scrubbed = IPV4_REGEX.sub(REDACTED_PLACEHOLDER, scrubbed)
    scrubbed = PHONE_REGEX.sub(REDACTED_PLACEHOLDER, scrubbed)
    return scrubbed


def strip_exif_clone_pixels(image_array: np.ndarray) -> np.ndarray:
    """Clones raw pixel intensities into a fresh metadata-free NumPy buffer.

    Args:
        image_array (np.ndarray): Source image array (2D gray or 3D RGB).

    Returns:
        np.ndarray: Fresh uint8 pixel array stripped of any metadata refs.
    """
    verify_condition(
        isinstance(image_array, np.ndarray) and image_array.size > ZERO_INT,
        "Input image array for EXIF stripping must be non-empty.",
    )
    verify_condition(
        image_array.ndim in (GRAY_DIMENSIONS, RGB_CHANNELS),
        "Image array must be 2D grayscale or 3D RGB.",
    )
    clean_buffer = np.empty_like(image_array, dtype=np.uint8)
    np.copyto(clean_buffer, image_array.astype(np.uint8))
    return clean_buffer


def apply_irreversible_roi_blur(
    image_array: np.ndarray,
    sensitive_bboxes: Sequence[Tuple[int, int, int, int]],
    block_size: int = MIN_BLUR_BLOCK_SIZE,
) -> np.ndarray:
    """Applies block quantization and separable 2D smoothing over ROIs.

    Args:
        image_array (np.ndarray): RGB or grayscale image buffer.
        sensitive_bboxes (Sequence[Tuple[int, int, int, int]]): Bounding
            boxes formatted as (x, y, width, height).
        block_size (int, optional): Quantization block size. Defaults to
            MIN_BLUR_BLOCK_SIZE.

    Returns:
        np.ndarray: Copy of image with irreversible ROI anonymisation.
    """
    output = strip_exif_clone_pixels(image_array)
    eff_block = max(
        MIN_BLUR_BLOCK_SIZE, min(int(block_size), MAX_BLUR_KERNEL_STEPS)
    )
    img_h, img_w = output.shape[:GRAY_DIMENSIONS]
    box_limit = min(len(sensitive_bboxes), MAX_ITEMS_CEILING)

    for idx in range(box_limit):
        bbox = sensitive_bboxes[idx]
        verify_condition(
            len(bbox) == BBOX_TUPLE_LENGTH,
            "Sensitive ROI bounding box must have 4 elements.",
        )
        bx, by, bw, bh = bbox
        x0 = max(ZERO_INT, min(int(bx), img_w))
        y0 = max(ZERO_INT, min(int(by), img_h))
        x1 = max(x0, min(int(bx + bw), img_w))
        y1 = max(y0, min(int(by + bh), img_h))
        if (x1 - x0) <= ZERO_INT or (y1 - y0) <= ZERO_INT:
            continue

        roi = output[y0:y1, x0:x1].astype(np.float32)
        roi_h, roi_w = roi.shape[:GRAY_DIMENSIONS]
        row_steps = min(
            math.ceil(roi_h / eff_block), MAX_IMAGE_DIMENSION_LIMIT
        )
        col_steps = min(
            math.ceil(roi_w / eff_block), MAX_IMAGE_DIMENSION_LIMIT
        )

        for r_idx in range(row_steps):
            ry0 = r_idx * eff_block
            ry1 = min(ry0 + eff_block, roi_h)
            for c_idx in range(col_steps):
                cx0 = c_idx * eff_block
                cx1 = min(cx0 + eff_block, roi_w)
                patch = roi[ry0:ry1, cx0:cx1]
                if patch.ndim == RGB_CHANNELS:
                    mean_val = np.mean(
                        patch, axis=(ZERO_INT, ONE_INT), keepdims=True
                    )
                else:
                    mean_val = np.mean(patch, keepdims=True)
                roi[ry0:ry1, cx0:cx1] = mean_val

        padded_h = np.pad(
            roi,
            ((ZERO_INT, ZERO_INT), (ONE_INT, ONE_INT))
            + (((ZERO_INT, ZERO_INT),) if roi.ndim == RGB_CHANNELS else ()),
            mode="edge",
        )
        horiz_smooth = (
            padded_h[:, :-2] + padded_h[:, 1:-1] + padded_h[:, 2:]
        ) / float(RGB_CHANNELS)
        padded_v = np.pad(
            horiz_smooth,
            ((ONE_INT, ONE_INT), (ZERO_INT, ZERO_INT))
            + (((ZERO_INT, ZERO_INT),) if roi.ndim == RGB_CHANNELS else ()),
            mode="edge",
        )
        vert_smooth = (
            padded_v[:-2, :] + padded_v[1:-1, :] + padded_v[2:, :]
        ) / float(RGB_CHANNELS)
        output[y0:y1, x0:x1] = np.clip(vert_smooth, ZERO_INT, 255).astype(
            np.uint8
        )

    return output


MAX_IMAGE_DIMENSION_LIMIT: int = 2048


@dataclass(frozen=True, slots=True)
class SpatialAnonymisationResult:
    """Record holding aperture-7 H3 spatial anonymisation and DP results."""

    resolution_level: int
    uniqueness_rate: float
    suppressed_outliers: int
    dp_cell_counts: Dict[str, float]


class GDPRComplianceEngine:
    """Enforces UK/EU GDPR technical controls, ledgering, and erasure."""

    def __init__(
        self,
        cloud_region: str = "eu-west-2",
        allowed_regions: Sequence[str] = (
            "eu-west-1",
            "eu-west-2",
            "eu-central-1",
        ),
        key_epoch: str = "2026-Q4",
        retention_ttl_seconds: int = DEFAULT_RETENTION_TTL_SECONDS,
        k_anonymity_threshold: int = DEFAULT_K_ANONYMITY,
        dp_epsilon: float = 0.8,
        blur_block_size: int = 8,
        ledger_file: str | Path = "src_audit_ledger.jsonl",
        workspace_root: Optional[str | Path] = None,
    ) -> None:
        """Initializes the GDPR compliance engine and validates residency.

        Args:
            cloud_region (str, optional): Configured cloud region. Defaults
                to "eu-west-2".
            allowed_regions (Sequence[str], optional): Allowed UK/EU regions.
            key_epoch (str, optional): Cryptographic rotation epoch string.
            retention_ttl_seconds (int, optional): Storage TTL in seconds.
            k_anonymity_threshold (int, optional): Minimum k-anonymity count.
            dp_epsilon (float, optional): Differential privacy epsilon (<=1).
            blur_block_size (int, optional): Block quantization pixel size.
            ledger_file (str | Path, optional): Audit ledger filename.
            workspace_root (Optional[str | Path], optional): Workspace root.
        """
        verify_condition(
            cloud_region in allowed_regions,
            f"Cloud region '{cloud_region}' violates UK/EU residency policy.",
            use_security_error=True,
        )
        verify_condition(
            MIN_DP_EPSILON <= float(dp_epsilon) <= MAX_DP_EPSILON,
            "Differential privacy epsilon must be within [0.01, 1.0].",
            use_security_error=True,
        )
        verify_condition(
            int(k_anonymity_threshold) >= TWO_FLOAT,
            "k-anonymity threshold must be at least 2.",
            use_security_error=True,
        )

        self._cloud_region = cloud_region
        self._allowed_regions = tuple(allowed_regions)
        self._key_epoch = key_epoch
        self._retention_ttl = int(retention_ttl_seconds)
        self._k_min = int(k_anonymity_threshold)
        self._dp_epsilon = float(dp_epsilon)
        self._blur_block_size = int(blur_block_size)
        self._workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root is not None
            else Path.cwd().resolve()
        )
        self._ledger_path = resolve_safe_path(
            ledger_file,
            workspace_root=self._workspace_root,
            must_exist=False,
        )
        raw_secret = os.environ.get(
            DEFAULT_MASTER_KEY_ENV, FALLBACK_MASTER_KEY_HEX
        )
        self._master_secret = raw_secret.encode("utf-8")

    def pseudonymise_identifier(
        self, raw_identifier: str, domain: str
    ) -> str:
        """Generates a domain-separated, epoch-bound HMAC-SHA256 pseudonym.

        Args:
            raw_identifier (str): Cleartext subject or filename identifier.
            domain (str): Logical domain context (e.g., 'upload_filename').

        Returns:
            str: Unlinkable cross-domain pseudonym hex token.
        """
        verify_condition(
            isinstance(raw_identifier, str) and len(raw_identifier) > ZERO_INT,
            "Raw identifier for pseudonymisation must be non-empty.",
        )
        verify_condition(
            isinstance(domain, str) and len(domain) > ZERO_INT,
            "Domain context for pseudonymisation must be non-empty.",
        )
        salt = f"epoch:{self._key_epoch}".encode("utf-8")
        info = f"src-gdpr-domain:{domain}".encode("utf-8")
        subkey = hkdf_sha256(
            self._master_secret, salt=salt, info=info, length=HKDF_HASH_LENGTH
        )
        normalized = raw_identifier.strip().encode("utf-8")
        digest = hmac.new(subkey, normalized, hashlib.sha256).hexdigest()
        return f"pseudo_{domain}_{digest[:PSEUDONYM_TOKEN_HEX_LENGTH]}"

    def anonymise_spatial_coordinates(
        self, coordinates: Sequence[Tuple[float, float]]
    ) -> SpatialAnonymisationResult:
        """Projects lat/lon points onto aperture-7 H3 cells until U=0.0.

        Args:
            coordinates (Sequence[Tuple[float, float]]): Sequence of
                (latitude, longitude) pairs.

        Returns:
            SpatialAnonymisationResult: Coarsened DP cell counts and metrics.
        """
        num_pts = min(len(coordinates), MAX_ITEMS_CEILING)
        if num_pts == ZERO_INT:
            return SpatialAnonymisationResult(
                resolution_level=ZERO_INT,
                uniqueness_rate=ZERO_FLOAT,
                suppressed_outliers=ZERO_INT,
                dp_cell_counts={},
            )

        best_counts: Dict[str, int] = {}
        chosen_res = SEVEN_RESOLUTION_START
        for step in range(H3_MAX_COARSEN_STEPS):
            current_res = max(ZERO_INT, SEVEN_RESOLUTION_START - step)
            scale = H3_BASE_SCALE_DEG * (H3_APERTURE_FACTOR**step)
            cell_counts: Dict[str, int] = {}
            for idx in range(num_pts):
                lat, lon = coordinates[idx]
                q_coord = round(float(lat) / scale)
                r_coord = round(
                    (float(lon) - HALF_FLOAT * float(lat)) / scale
                )
                cell_id = f"h3_r{current_res}_{q_coord}_{r_coord}"
                cell_counts[cell_id] = cell_counts.get(cell_id, ZERO_INT) + ONE_INT

            failing_pts = sum(
                cnt for cnt in cell_counts.values() if cnt < self._k_min
            )
            uniqueness = failing_pts / float(num_pts)
            best_counts = cell_counts
            chosen_res = current_res
            if uniqueness == ZERO_FLOAT:
                break

        suppressed = ZERO_INT
        dp_counts: Dict[str, float] = {}
        scale_b = ONE_FLOAT / self._dp_epsilon
        cell_keys = list(best_counts.keys())

        for idx in range(min(len(cell_keys), MAX_ITEMS_CEILING)):
            cell_key = cell_keys[idx]
            raw_count = best_counts[cell_key]
            if raw_count < self._k_min:
                suppressed += raw_count
                continue
            u_rand = (secrets.randbelow(999999) + ONE_INT) / 1000000.0 - HALF_FLOAT
            sign = ONE_FLOAT if u_rand >= ZERO_FLOAT else -ONE_FLOAT
            laplace_noise = (
                -scale_b
                * sign
                * math.log(ONE_FLOAT - TWO_FLOAT * abs(u_rand))
            )
            noisy_val = max(ZERO_FLOAT, float(raw_count) + laplace_noise)
            dp_counts[cell_key] = round(noisy_val, ROUND_DECIMALS_SHAPES)

        return SpatialAnonymisationResult(
            resolution_level=chosen_res,
            uniqueness_rate=ZERO_FLOAT,
            suppressed_outliers=suppressed,
            dp_cell_counts=dp_counts,
        )

    def append_audit_ledger(
        self, event_type: str, subject_token: str, details: Dict[str, Any]
    ) -> str:
        """Appends a PII-free, SHA-256 hash-chained entry to the JSONL ledger.

        Args:
            event_type (str): Categorized pipeline or GDPR event name.
            subject_token (str): Pseudonymised subject or batch token.
            details (Dict[str, Any]): Non-PII operational metadata.

        Returns:
            str: Computed SHA-256 entry hash of the appended record.
        """
        prev_hash = "0" * 64
        if self._ledger_path.exists():
            lines = self._ledger_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > ZERO_INT:
                last_record = json.loads(lines[-1])
                prev_hash = str(last_record.get("entry_hash", prev_hash))

        sanitized_details: Dict[str, Any] = {}
        detail_keys = list(details.keys())
        for idx in range(min(len(detail_keys), MAX_ITEMS_CEILING)):
            k = detail_keys[idx]
            val = details[k]
            sanitized_details[k] = (
                scrub_pii_text(val) if isinstance(val, str) else val
            )

        record_body = {
            "timestamp": round(time.time(), ROUND_DECIMALS_SHAPES),
            "region": self._cloud_region,
            "event_type": scrub_pii_text(event_type),
            "subject_token": scrub_pii_text(subject_token),
            "details": sanitized_details,
            "prev_hash": prev_hash,
        }
        canonical_json = json.dumps(
            record_body, sort_keys=True, separators=(",", ":")
        )
        entry_hash = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()
        record_body["entry_hash"] = entry_hash

        with open(self._ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record_body, sort_keys=True) + "\n")

        return entry_hash

    def verify_ledger_integrity(self) -> bool:
        """Verifies the cryptographic hash chain of the JSONL audit ledger.

        Returns:
            bool: True if all hash links and digests are intact.
        """
        if not self._ledger_path.exists():
            return True
        lines = self._ledger_path.read_text(encoding="utf-8").splitlines()
        verify_condition(
            len(lines) <= MAX_LEDGER_SCAN_LINES,
            "Ledger line count exceeds scan ceiling.",
        )
        expected_prev = "0" * 64
        for idx in range(len(lines)):
            entry = json.loads(lines[idx])
            stored_hash = str(entry.pop("entry_hash", ""))
            verify_condition(
                hmac.compare_digest(
                    str(entry.get("prev_hash", "")), expected_prev
                ),
                f"Ledger chain broken at record index {idx}.",
                use_security_error=True,
            )
            canonical = json.dumps(
                entry, sort_keys=True, separators=(",", ":")
            )
            recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            verify_condition(
                hmac.compare_digest(stored_hash, recomputed),
                f"Ledger entry hash tampered at index {idx}.",
                use_security_error=True,
            )
            expected_prev = stored_hash
        return True

    def purge_expired_staging_files(
        self, staging_dir: str | Path, current_time: Optional[float] = None
    ) -> int:
        """Deletes staging files older than retention_ttl_seconds.

        Args:
            staging_dir (str | Path): Directory containing staging artifacts.
            current_time (Optional[float], optional): Reference epoch time.

        Returns:
            int: Number of expired files removed.
        """
        safe_dir = resolve_safe_path(
            staging_dir, workspace_root=self._workspace_root, must_exist=True
        )
        now = current_time if current_time is not None else time.time()
        removed_count = ZERO_INT
        children = list(safe_dir.iterdir())
        for idx in range(min(len(children), MAX_ITEMS_CEILING)):
            child = children[idx]
            if child.is_file() and not child.is_symlink():
                age = now - child.stat().st_mtime
                if age >= self._retention_ttl:
                    child.unlink()
                    removed_count += ONE_INT
        return removed_count

    def execute_right_to_erasure(
        self,
        raw_subject_id: str,
        domain: str = "subject",
        sqlite_db_path: Optional[str | Path] = None,
        json_store_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Executes GDPR Article 17 Right to Erasure across DB and JSON stores.

        Args:
            raw_subject_id (str): Subject identifier requesting erasure.
            domain (str, optional): Pseudonym domain. Defaults to "subject".
            sqlite_db_path (Optional[str | Path], optional): SQLite file path.
            json_store_path (Optional[str | Path], optional): JSON output path.

        Returns:
            Dict[str, Any]: Summary of erased records and audit hash.
        """
        subject_token = self.pseudonymise_identifier(
            raw_subject_id, domain=domain
        )
        db_deleted = ZERO_INT
        json_deleted = ZERO_INT

        if sqlite_db_path is not None:
            safe_db = resolve_safe_path(
                sqlite_db_path,
                workspace_root=self._workspace_root,
                must_exist=False,
            )
            if safe_db.exists():
                with closing(sqlite3.connect(str(safe_db))) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM apparel_batches WHERE owner_id = ?",
                        (subject_token,),
                    )
                    db_deleted = int(cursor.rowcount)
                    conn.commit()

        if json_store_path is not None:
            safe_json = resolve_safe_path(
                json_store_path,
                workspace_root=self._workspace_root,
                must_exist=False,
            )
            if safe_json.exists():
                records = json.loads(safe_json.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    retained = [
                        r
                        for r in records[:MAX_ITEMS_CEILING]
                        if r.get("owner_id") != subject_token
                    ]
                    json_deleted = len(records) - len(retained)
                    safe_json.write_text(
                        json.dumps(retained, indent=2), encoding="utf-8"
                    )

        ledger_hash = self.append_audit_ledger(
            event_type="GDPR_ARTICLE_17_ERASURE",
            subject_token=subject_token,
            details={
                "db_rows_erased": db_deleted,
                "json_records_erased": json_deleted,
            },
        )
        return {
            "subject_token": subject_token,
            "db_rows_erased": db_deleted,
            "json_records_erased": json_deleted,
            "ledger_entry_hash": ledger_hash,
        }


SEVEN_RESOLUTION_START: int = 7