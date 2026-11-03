"""OWASP ASVS v5.0 application security guards for the Src pipeline."""

import hashlib
import hmac
import io
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse
import zipfile

import numpy as np
from PIL import Image
import yaml

from src.constants import (
    JPEG_MAGIC_BYTES,
    MAX_FILE_BYTES_CEILING,
    MAX_IMAGE_DIMENSION_CEILING,
    MAX_IMAGE_PIXELS_CEILING,
    MAX_ITEMS_CEILING,
    MAX_PATH_LENGTH,
    ONE_INT,
    PNG_MAGIC_BYTES,
    RGB_CHANNELS,
    WEBP_FOURCC_BYTES,
    WEBP_HEADER_MIN_LEN,
    WEBP_RIFF_BYTES,
    XML_FORBIDDEN_TOKENS,
    ZERO_INT,
    verify_condition,
)


def resolve_safe_path(
    target_path: str | Path,
    workspace_root: Optional[str | Path] = None,
    must_exist: bool = True,
) -> Path:
    """Canonicalizes a path and verifies containment inside workspace_root.

    Args:
        target_path (str | Path): Candidate path to validate.
        workspace_root (Optional[str | Path], optional): Allowed root
            directory. Defaults to current working directory.
        must_exist (bool, optional): Whether the target file must already
            exist on disk. Defaults to True.

    Returns:
        Path: Canonicalized, validated Path object.
    """
    raw_str = str(target_path)
    verify_condition(
        ZERO_INT < len(raw_str) <= MAX_PATH_LENGTH,
        "Path length is outside allowed bounds.",
        use_security_error=True,
    )
    verify_condition(
        "\x00" not in raw_str,
        "Null bytes are forbidden in file paths.",
        use_security_error=True,
    )
    raw_parts = Path(raw_str).parts
    verify_condition(
        ".." not in raw_parts,
        "Parent directory traversal segments are forbidden.",
        use_security_error=True,
    )

    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    candidate = Path(raw_str)
    if not candidate.is_absolute():
        candidate = root / candidate

    verify_condition(
        not candidate.is_symlink(),
        "Symbolic links are rejected by security policy.",
        use_security_error=True,
    )
    resolved = candidate.resolve()
    verify_condition(
        resolved.is_relative_to(root),
        "Resolved path escapes the workspace boundary.",
        use_security_error=True,
    )
    if must_exist:
        verify_condition(
            resolved.exists(),
            f"Required path does not exist: {resolved.name}",
            use_security_error=True,
        )
    return resolved


def validate_safe_archive(
    archive_path: str | Path,
    workspace_root: Optional[str | Path] = None,
    max_members: int = MAX_ITEMS_CEILING,
    max_total_bytes: int = MAX_FILE_BYTES_CEILING,
) -> List[str]:
    """Validates a ZIP archive against Zip-Slip and decompression bombs.

    Args:
        archive_path (str | Path): Path to the archive file.
        workspace_root (Optional[str | Path], optional): Workspace root.
            Defaults to None.
        max_members (int, optional): Maximum archive entries. Defaults to
            MAX_ITEMS_CEILING.
        max_total_bytes (int, optional): Maximum uncompressed byte ceiling.
            Defaults to MAX_FILE_BYTES_CEILING.

    Returns:
        List[str]: Validated member names inside the archive.
    """
    safe_archive = resolve_safe_path(
        archive_path, workspace_root=workspace_root, must_exist=True
    )
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    validated_names: List[str] = []
    running_bytes = ZERO_INT

    with zipfile.ZipFile(safe_archive, mode="r") as zf:
        info_list = zf.infolist()
        verify_condition(
            len(info_list) <= max_members,
            "Archive member count exceeds ceiling.",
            use_security_error=True,
        )
        for idx in range(min(len(info_list), max_members)):
            member = info_list[idx]
            verify_condition(
                ".." not in Path(member.filename).parts,
                "Archive member contains path traversal segment.",
                use_security_error=True,
            )
            dest = (root / member.filename).resolve()
            verify_condition(
                dest.is_relative_to(root),
                "Archive member escapes workspace root.",
                use_security_error=True,
            )
            running_bytes += int(member.file_size)
            verify_condition(
                running_bytes <= max_total_bytes,
                "Archive uncompressed size exceeds ceiling.",
                use_security_error=True,
            )
            validated_names.append(member.filename)

    return validated_names


def verify_sha256_digest(payload: bytes, expected_hex_digest: str) -> bool:
    """Verifies a SHA-256 digest using constant-time comparison.

    Args:
        payload (bytes): Raw bytes to hash.
        expected_hex_digest (str): Expected lowercase hex SHA-256 string.

    Returns:
        bool: True if digests match.
    """
    verify_condition(
        isinstance(payload, (bytes, bytearray)),
        "Payload must be bytes for SHA-256 verification.",
        use_security_error=True,
    )
    computed = hashlib.sha256(payload).hexdigest().lower()
    expected_clean = expected_hex_digest.strip().lower()
    matched = hmac.compare_digest(computed, expected_clean)
    verify_condition(
        matched,
        "SHA-256 artifact integrity verification failed.",
        use_security_error=True,
    )
    return matched


def load_safe_yaml(
    config_path: str | Path,
    workspace_root: Optional[str | Path] = None,
    max_bytes: int = MAX_FILE_BYTES_CEILING,
) -> Dict[str, Any]:
    """Loads a YAML configuration safely after path and byte validation.

    Args:
        config_path (str | Path): Path to the YAML file.
        workspace_root (Optional[str | Path], optional): Workspace root.
            Defaults to None.
        max_bytes (int, optional): Maximum allowed file bytes. Defaults to
            MAX_FILE_BYTES_CEILING.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    safe_path = resolve_safe_path(
        config_path, workspace_root=workspace_root, must_exist=True
    )
    raw_bytes = safe_path.read_bytes()
    verify_condition(
        ZERO_INT < len(raw_bytes) <= max_bytes,
        "YAML file size exceeds security ceiling.",
        use_security_error=True,
    )
    parsed = yaml.safe_load(raw_bytes.decode("utf-8"))
    verify_condition(
        isinstance(parsed, dict),
        "Root YAML configuration must deserialize to a dictionary.",
        use_security_error=True,
    )
    return parsed


def load_verified_npz(
    npz_path: str | Path,
    expected_sha256: str,
    workspace_root: Optional[str | Path] = None,
) -> Dict[str, np.ndarray]:
    """Loads a NumPy .npz archive with pickle disabled and SHA-256 check.

    Args:
        npz_path (str | Path): Path to the .npz weights/coefficients file.
        expected_sha256 (str): Expected SHA-256 hex digest.
        workspace_root (Optional[str | Path], optional): Workspace root.
            Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary of validated NumPy arrays.
    """
    safe_path = resolve_safe_path(
        npz_path, workspace_root=workspace_root, must_exist=True
    )
    raw_bytes = safe_path.read_bytes()
    verify_sha256_digest(raw_bytes, expected_sha256)
    arrays: Dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as data:
        keys = list(data.files)
        for idx in range(min(len(keys), MAX_ITEMS_CEILING)):
            key = keys[idx]
            arrays[key] = np.array(data[key], copy=True)
    return arrays


def inspect_xml_text_safety(xml_text: str) -> str:
    """Blocks XXE and DTD entity declarations in XML/SVG strings.

    Args:
        xml_text (str): Raw XML string to inspect.

    Returns:
        str: Validated XML string if safe.
    """
    verify_condition(
        isinstance(xml_text, str) and len(xml_text) <= MAX_FILE_BYTES_CEILING,
        "XML payload must be a bounded string.",
        use_security_error=True,
    )
    upper_text = xml_text.upper()
    for idx in range(len(XML_FORBIDDEN_TOKENS)):
        token = XML_FORBIDDEN_TOKENS[idx]
        verify_condition(
            token not in upper_text,
            f"Forbidden XML entity token detected: {token}",
            use_security_error=True,
        )
    return xml_text


def verify_image_magic_bytes(raw_bytes: bytes) -> bool:
    """Verifies that raw image bytes match PNG, JPEG, or WebP magic headers.

    Args:
        raw_bytes (bytes): Uploaded image byte stream.

    Returns:
        bool: True if magic header is valid.
    """
    verify_condition(
        len(raw_bytes) >= WEBP_HEADER_MIN_LEN,
        "Image payload is shorter than minimum header size.",
        use_security_error=True,
    )
    is_png = raw_bytes.startswith(PNG_MAGIC_BYTES)
    is_jpeg = raw_bytes.startswith(JPEG_MAGIC_BYTES)
    is_webp = (
        raw_bytes[:4] == WEBP_RIFF_BYTES
        and raw_bytes[8:WEBP_HEADER_MIN_LEN] == WEBP_FOURCC_BYTES
    )
    verify_condition(
        is_png or is_jpeg or is_webp,
        "Unsupported or spoofed image magic bytes.",
        use_security_error=True,
    )
    return True


def decode_and_validate_image_bytes(
    raw_bytes: bytes,
    max_bytes: int = MAX_FILE_BYTES_CEILING,
    max_width: int = MAX_IMAGE_DIMENSION_CEILING,
    max_height: int = MAX_IMAGE_DIMENSION_CEILING,
    max_pixels: int = MAX_IMAGE_PIXELS_CEILING,
) -> np.ndarray:
    """Validates byte size, magic headers, and pixel bounds before decoding.

    Args:
        raw_bytes (bytes): Raw uploaded image bytes.
        max_bytes (int, optional): Maximum byte size. Defaults to
            MAX_FILE_BYTES_CEILING.
        max_width (int, optional): Maximum pixel width. Defaults to
            MAX_IMAGE_DIMENSION_CEILING.
        max_height (int, optional): Maximum pixel height. Defaults to
            MAX_IMAGE_DIMENSION_CEILING.
        max_pixels (int, optional): Maximum total pixels. Defaults to
            MAX_IMAGE_PIXELS_CEILING.

    Returns:
        np.ndarray: Validated RGB image NumPy array of shape (H, W, 3).
    """
    verify_condition(
        isinstance(raw_bytes, (bytes, bytearray)),
        "Image payload must be raw bytes.",
        use_security_error=True,
    )
    verify_condition(
        ZERO_INT < len(raw_bytes) <= max_bytes,
        "Image byte length exceeds maximum upload ceiling.",
        use_security_error=True,
    )
    verify_image_magic_bytes(raw_bytes)

    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(io.BytesIO(raw_bytes)) as pil_img:
        width, height = pil_img.size
        verify_condition(
            ONE_INT <= width <= max_width and ONE_INT <= height <= max_height,
            "Image dimensions exceed configured security bounds.",
            use_security_error=True,
        )
        verify_condition(
            (width * height) <= max_pixels,
            "Total image pixel count exceeds decompression bomb ceiling.",
            use_security_error=True,
        )
        rgb_img = pil_img.convert("RGB")
        clean_array = np.array(rgb_img, dtype=np.uint8, copy=True)

    verify_condition(
        clean_array.ndim == RGB_CHANNELS
        and clean_array.shape[2] == RGB_CHANNELS,
        "Decoded image array did not produce a valid 3-channel RGB buffer.",
        use_security_error=True,
    )
    return clean_array


def validate_outbound_url(
    url: str,
    allowed_schemes: Sequence[str] = ("http", "https"),
    allowed_hosts: Sequence[str] = ("127.0.0.1", "localhost"),
) -> str:
    """Validates an outbound URL against SSRF scheme and host allowlists.

    Args:
        url (str): Candidate URL string.
        allowed_schemes (Sequence[str], optional): Allowed schemes. Defaults
            to ("http", "https").
        allowed_hosts (Sequence[str], optional): Allowed hostnames. Defaults
            to ("127.0.0.1", "localhost").

    Returns:
        str: Validated URL string.
    """
    verify_condition(
        isinstance(url, str) and ZERO_INT < len(url) <= MAX_PATH_LENGTH,
        "URL must be a non-empty bounded string.",
        use_security_error=True,
    )
    parsed = urlparse(url)
    verify_condition(
        parsed.scheme in allowed_schemes,
        f"URL scheme '{parsed.scheme}' is not in allowlist.",
        use_security_error=True,
    )
    verify_condition(
        parsed.hostname in allowed_hosts,
        f"URL host '{parsed.hostname}' is not in allowlist.",
        use_security_error=True,
    )
    verify_condition(
        parsed.username is None and parsed.password is None,
        "Embedded credentials in URLs are prohibited.",
        use_security_error=True,
    )
    return url


def run_safe_command(
    command_args: Sequence[str],
    allowed_binaries: Sequence[str] = ("python3", "uv"),
    timeout_seconds: int = 15,
) -> subprocess.CompletedProcess[str]:
    """Executes an allowlisted binary with shell=False and sanitized env.

    Args:
        command_args (Sequence[str]): Command argument sequence.
        allowed_binaries (Sequence[str], optional): Permitted binary names.
            Defaults to ("python3", "uv").
        timeout_seconds (int, optional): Execution timeout. Defaults to 15.

    Returns:
        subprocess.CompletedProcess[str]: Completed subprocess result.
    """
    verify_condition(
        len(command_args) >= ONE_INT,
        "Command arguments cannot be empty.",
        use_security_error=True,
    )
    binary_name = Path(command_args[ZERO_INT]).name
    verify_condition(
        binary_name in allowed_binaries,
        f"Binary '{binary_name}' is not in the execution allowlist.",
        use_security_error=True,
    )
    for idx in range(len(command_args)):
        arg = str(command_args[idx])
        verify_condition(
            "\x00" not in arg and ";" not in arg and "|" not in arg,
            "Command argument contains forbidden shell metacharacters.",
            use_security_error=True,
        )

    sanitized_env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C.UTF-8"}
    return subprocess.run(
        list(command_args),
        shell=False,
        env=sanitized_env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=True,
    )