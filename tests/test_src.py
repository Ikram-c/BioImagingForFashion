"""Comprehensive unittest suite for OWASP, GDPR, Pre-Deploy, and EFD core."""

from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
import numpy as np

from src.api import app, encode_synthetic_png_bytes
from src.constants import (
    SrcSecurityError,
    SrcVerificationError,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_GONE,
    HTTP_STATUS_INTERNAL_ERROR,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_TOO_MANY_REQUESTS,
)
from src.fourier import compute_elliptical_fourier_descriptors
from src.gdpr import (
    GDPRComplianceEngine,
    apply_irreversible_roi_blur,
    scrub_pii_text,
    strip_exif_clone_pixels,
)
from src.logging_config import configure_logging, get_ring_buffer_handler
from src.pipeline import SrcPipeline
from src.predeploy import (
    BlueGreenDeploymentManager,
    IndexedBatchStore,
    PasswordResetTokenManager,
    PreDeployHTTPException,
    SlidingWindowRateLimiter,
    execute_with_catch_all_boundary,
    sanitize_and_validate_input,
    verify_bola_ownership,
    verify_cors_origin,
)
from src.sandbox import (
    generate_synthetic_swimsuits_image,
    generate_synthetic_telemetry_coords,
)
from src.security import (
    decode_and_validate_image_bytes,
    inspect_xml_text_safety,
    resolve_safe_path,
    validate_outbound_url,
    verify_sha256_digest,
)


class TestSrcSecurityAndGDPR(unittest.TestCase):
    """Tests OWASP ASVS v5.0 security guards and UK/EU GDPR privacy controls."""

    def test_path_traversal_and_null_byte_blocked(self) -> None:
        """Verifies CWE-22 path traversal and null-byte rejection."""
        with self.assertRaises(SrcSecurityError):
            resolve_safe_path("../outside_secret.yaml", must_exist=False)
        with self.assertRaises(SrcSecurityError):
            resolve_safe_path("config\x00.yaml", must_exist=False)

    def test_xml_xxe_and_ssrf_guards(self) -> None:
        """Verifies CWE-611 XML DTD blocker and CWE-918 SSRF allowlist."""
        with self.assertRaises(SrcSecurityError):
            inspect_xml_text_safety("<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'x'>]>")
        with self.assertRaises(SrcSecurityError):
            validate_outbound_url("http://169.254.169.254/latest/meta-data")
        valid_url = validate_outbound_url("http://127.0.0.1:10000/blob")
        self.assertEqual(valid_url, "http://127.0.0.1:10000/blob")

    def test_decompression_bomb_and_magic_bytes(self) -> None:
        """Verifies CWE-400 magic header check and pixel ceiling enforcement."""
        with self.assertRaises(SrcSecurityError):
            decode_and_validate_image_bytes(b"NOT_AN_IMAGE_HEADER_12345")
        png_bytes = encode_synthetic_png_bytes()
        with self.assertRaises(SrcSecurityError):
            decode_and_validate_image_bytes(
                png_bytes, max_width=100, max_height=100
            )

    def test_gdpr_hkdf_pseudonymisation_and_pii_scrub(self) -> None:
        """Verifies domain-separated HKDF tokens and regex PII scrubbing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GDPRComplianceEngine(
                workspace_root=tmpdir,
                ledger_file=Path(tmpdir) / "ledger.jsonl",
            )
            t1 = engine.pseudonymise_identifier("alice@example.com", "dom_a")
            t2 = engine.pseudonymise_identifier("alice@example.com", "dom_b")
            self.assertNotEqual(t1, t2)
            scrubbed = scrub_pii_text(
                "Contact alice@example.com or 192.168.1.10 or +44 7428810579"
            )
            self.assertNotIn("alice@example.com", scrubbed)
            self.assertNotIn("192.168.1.10", scrubbed)

    def test_gdpr_spatial_h3_and_ledger_and_erasure(self) -> None:
        """Verifies H3 U=0.0 anonymisation, ledger hash chain, and Art. 17."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite3"
            ledger_path = Path(tmpdir) / "ledger.jsonl"
            engine = GDPRComplianceEngine(
                workspace_root=tmpdir, ledger_file=ledger_path
            )
            store = IndexedBatchStore(db_path=db_path, workspace_root=tmpdir)
            owner_tok = engine.pseudonymise_identifier("user_42", "subject")
            store.save_batch("batch_1", owner_tok, [{"item_id": 1}])

            coords = generate_synthetic_telemetry_coords()
            spatial = engine.anonymise_spatial_coordinates(coords)
            self.assertEqual(spatial.uniqueness_rate, 0.0)

            erasure = engine.execute_right_to_erasure(
                raw_subject_id="user_42",
                domain="subject",
                sqlite_db_path=db_path,
            )
            self.assertEqual(erasure["db_rows_erased"], 1)
            self.assertTrue(engine.verify_ledger_integrity())

    def test_exif_strip_and_irreversible_roi_blur(self) -> None:
        """Verifies clean pixel cloning and ROI block quantization blur."""
        img = generate_synthetic_swimsuits_image()
        clean = strip_exif_clone_pixels(img)
        blurred = apply_irreversible_roi_blur(
            clean, sensitive_bboxes=[(180, 260, 80, 80)], block_size=8
        )
        self.assertEqual(blurred.shape, img.shape)
        self.assertFalse(np.array_equal(clean[260:340, 180:260], blurred[260:340, 180:260]))


class TestTenPointPreDeploymentChecklist(unittest.TestCase):
    """Tests all 10 operational readiness controls in src/predeploy.py."""

    def test_1_bola_ownership_verification(self) -> None:
        """Check #1: Constant-time BOLA ownership verification."""
        self.assertTrue(verify_bola_ownership("owner_abc", "owner_abc"))
        with self.assertRaises(PreDeployHTTPException) as ctx:
            verify_bola_ownership("owner_abc", "owner_xyz")
        self.assertEqual(ctx.exception.status_code, HTTP_STATUS_FORBIDDEN)

    def test_2_sqli_and_xss_input_sanitization(self) -> None:
        """Check #2: SQLi and XSS token blocking."""
        with self.assertRaises(PreDeployHTTPException) as sqli_ctx:
            sanitize_and_validate_input("admin' OR 1=1 --")
        self.assertEqual(sqli_ctx.exception.status_code, HTTP_STATUS_BAD_REQUEST)
        with self.assertRaises(PreDeployHTTPException) as xss_ctx:
            sanitize_and_validate_input("<script>alert(1)</script>")
        self.assertEqual(xss_ctx.exception.status_code, HTTP_STATUS_BAD_REQUEST)

    def test_3_strict_cors_origin_allowlist(self) -> None:
        """Check #3: Strict CORS origin verification."""
        allowed = ("http://localhost:3000", "http://127.0.0.1:8000")
        self.assertEqual(
            verify_cors_origin("http://localhost:3000", allowed),
            "http://localhost:3000",
        )
        with self.assertRaises(PreDeployHTTPException) as ctx:
            verify_cors_origin("https://evil.example.com", allowed)
        self.assertEqual(ctx.exception.status_code, HTTP_STATUS_FORBIDDEN)

    def test_4_sliding_window_rate_limiting(self) -> None:
        """Check #4: Sliding-window per-client rate limiter (HTTP 429)."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        limiter.check_and_record("client_1", current_time=100.0)
        limiter.check_and_record("client_1", current_time=101.0)
        with self.assertRaises(PreDeployHTTPException) as ctx:
            limiter.check_and_record("client_1", current_time=102.0)
        self.assertEqual(
            ctx.exception.status_code, HTTP_STATUS_TOO_MANY_REQUESTS
        )

    def test_5_thirty_minute_expiring_password_reset_tokens(self) -> None:
        """Check #5: Single-use 30-minute expiring password reset links."""
        mgr = PasswordResetTokenManager(ttl_seconds=1800)
        tok = mgr.issue_reset_token("operator_1", current_time=1000.0)
        with self.assertRaises(PreDeployHTTPException) as exp_ctx:
            mgr.consume_reset_token(tok, current_time=2805.0)
        self.assertEqual(exp_ctx.exception.status_code, HTTP_STATUS_GONE)

        tok2 = mgr.issue_reset_token("operator_1", current_time=1000.0)
        user = mgr.consume_reset_token(tok2, current_time=1500.0)
        self.assertEqual(user, "operator_1")
        with self.assertRaises(PreDeployHTTPException) as reuse_ctx:
            mgr.consume_reset_token(tok2, current_time=1501.0)
        self.assertEqual(reuse_ctx.exception.status_code, HTTP_STATUS_GONE)

    def test_6_8_9_catch_all_boundary_ring_logs_and_alerts(self) -> None:
        """Checks #6, #8, #9: 500 boundary, ring buffer, and breakage alerts."""
        alerts = []
        configure_logging(
            level="INFO",
            enabled=True,
            ring_capacity=16,
            max_detail_length=60,
            alert_callback=lambda a: alerts.append(a),
        )

        def _failing_op() -> dict:
            raise RuntimeError("Simulated database crash with secret trace")

        status, payload = execute_with_catch_all_boundary(_failing_op)
        self.assertEqual(status, HTTP_STATUS_INTERNAL_ERROR)
        self.assertTrue(payload["stack_trace_suppressed"])
        self.assertNotIn("Simulated database crash", payload["user_message"])
        self.assertEqual(len(alerts), 1)
        ring = get_ring_buffer_handler()
        self.assertIsNotNone(ring)

    def test_7_targeted_sqlite_btree_indexes(self) -> None:
        """Check #7: Targeted B-tree indexes on owner_id and batch_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IndexedBatchStore(
                db_path=Path(tmpdir) / "idx.sqlite3", workspace_root=tmpdir
            )
            indexes = store.list_index_names()
            self.assertIn("idx_apparel_batches_owner_id", indexes)
            self.assertIn("idx_apparel_batches_batch_id", indexes)

    def test_10_blue_green_deployment_and_instant_rollback(self) -> None:
        """Check #10: Blue-green deployment promotion and instant rollback."""
        bg = BlueGreenDeploymentManager(initial_slot="blue")
        failed = bg.deploy_and_promote("0.3.0-bad", health_probe=lambda: False)
        self.assertTrue(failed["rolled_back"])
        self.assertEqual(bg.active_slot, "blue")

        promoted = bg.deploy_and_promote("0.3.0-ok", health_probe=lambda: True)
        self.assertTrue(promoted["promoted"])
        self.assertEqual(bg.active_slot, "green")


class TestPipelineAndEditorialAPI(unittest.TestCase):
    """Tests the EFD math core, stage pipeline, and FastAPI endpoints."""

    def test_pipeline_and_api_endpoints(self) -> None:
        """Verifies synthetic EFD extraction and FastAPI editorial routes."""
        client = TestClient(app)
        ui_resp = client.get("/")
        self.assertEqual(ui_resp.status_code, HTTP_STATUS_OK)
        self.assertIn("SAIKASÍ", ui_resp.text)
        self.assertIn("#F6F4F5", ui_resp.text)

        png_bytes = encode_synthetic_png_bytes()
        analyze_resp = client.post(
            "/analyze",
            headers={
                "X-Owner-Id": "flutter_client_1",
                "Origin": "http://10.0.2.2:8000",
            },
            files={"file": ("swimsuit.png", png_bytes, "image/png")},
        )
        self.assertEqual(analyze_resp.status_code, HTTP_STATUS_OK)
        body = analyze_resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["item_count"], 2)


if __name__ == "__main__":
    unittest.main()