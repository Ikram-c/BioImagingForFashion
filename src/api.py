"""Hardened FastAPI server and SAIKASI editorial UI for Src."""

import io
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, Header, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from src.constants import (
     DEFAULT_CONFIG_FILENAME,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_OK,
    PACKAGE_VERSION,
    ZERO_INT,
)
from src.logging_config import get_ring_buffer_handler
from src.pipeline import SrcPipeline
from src.predeploy import (
    BlueGreenDeploymentManager,
    IndexedBatchStore,
    PasswordResetTokenManager,
    PreDeployHTTPException,
    SlidingWindowRateLimiter,
    execute_with_catch_all_boundary,
    sanitize_and_validate_input,
    verify_cors_origin,
)
from src.sandbox import (
    CloudSandboxAdapter,
    generate_synthetic_swimsuits_image,
)


pipeline_instance = SrcPipeline(config_path=DEFAULT_CONFIG_FILENAME)
_cfg = pipeline_instance.config
_pd_cfg = _cfg.get("predeploy", {})
_srv_cfg = _cfg.get("server", {})
_sbx_cfg = _cfg.get("sandbox", {})

ALLOWED_CORS_ORIGINS = tuple(
    _srv_cfg.get(
        "allowed_cors_origins",
        (
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://10.0.2.2:8000",
        ),
    )
)

rate_limiter = SlidingWindowRateLimiter(
    max_requests=int(_pd_cfg.get("rate_limit_per_minute", 30)),
    window_seconds=int(_pd_cfg.get("rate_limit_window_seconds", 60)),
)
reset_manager = PasswordResetTokenManager(
    ttl_seconds=int(_pd_cfg.get("password_reset_ttl_seconds", 1800))
)
batch_store = IndexedBatchStore(
    db_path=str(_pd_cfg.get("sqlite_db_path", "src_records.sqlite3"))
)
deployment_manager = BlueGreenDeploymentManager(
    initial_slot=str(_pd_cfg.get("active_deployment_slot", "blue")),
    initial_version=PACKAGE_VERSION,
)
sandbox_adapter = CloudSandboxAdapter(
    azurite_endpoint=str(
        _sbx_cfg.get("azurite_endpoint", "http://127.0.0.1:10000")
    ),
    gcs_endpoint=str(
        _sbx_cfg.get("gcs_endpoint", "http://127.0.0.1:4443")
    ),
    offline_fallback=bool(_sbx_cfg.get("offline_fallback", True)),
)

app = FastAPI(
    title="Saikasi Src Swimsuit EFD Atelier API",
    description=(
        "Hardened OWASP ASVS & UK/EU GDPR backend for apparel "
        "Elliptical Fourier Descriptors and Flutter UI integration."
    ),
    version=PACKAGE_VERSION,
)


def _resolve_client_key(request: Request) -> str:
    """Derives a pseudonymised client rate-limiting key from request host.

    Args:
        request (Request): Incoming FastAPI request object.

    Returns:
        str: Pseudonymised client identifier token.
    """
    raw_host = (
        request.client.host if request.client is not None else "127.0.0.1"
    )
    return pipeline_instance.gdpr_engine.pseudonymise_identifier(
        raw_host, domain="client_rate_limit"
    )


def _build_json_response(
    status_code: int,
    payload: Dict[str, Any],
    validated_origin: Optional[str] = None,
) -> JSONResponse:
    """Constructs a JSONResponse with optional strict CORS headers.

    Args:
        status_code (int): HTTP status code.
        payload (Dict[str, Any]): Response body dictionary.
        validated_origin (Optional[str], optional): Verified CORS origin.

    Returns:
        JSONResponse: Configured FastAPI JSONResponse.
    """
    headers: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
    }
    if validated_origin is not None:
        headers["Access-Control-Allow-Origin"] = validated_origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=status_code, content=payload, headers=headers
    )


SAIKASI_EDITORIAL_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAIKASÍ — Src Morphometric Swimwear Atelier</title>
<style>
:root {
  --color-cultured: #F6F4F5;
  --color-malted-milk: #E8D9CE;
  --color-sunspark: #FBCA69;
  --color-red-clay: #E45E32;
  --color-witch-soup: #692746;
  --color-cultist-robe: #331233;
  --radius-card: 12px;
  --radius-pill: 999px;
  --max-container: 1240px;
  --grid-gap: 24px;
  --transition-fast: 0.2s ease;
}
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
body {
  background-color: var(--color-cultured);
  color: var(--color-cultist-robe);
  font-family: "Georgia", "Times New Roman", serif;
  line-height: 1.5;
}
.utility-bar {
  background-color: var(--color-cultist-robe);
  color: var(--color-cultured);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 8px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.utility-pill {
  background-color: var(--color-witch-soup);
  color: var(--color-sunspark);
  padding: 2px 10px;
  border-radius: var(--radius-pill);
  font-weight: 600;
}
.site-header {
  max-width: var(--max-container);
  margin: 0 auto;
  padding: 22px 24px;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--color-malted-milk);
}
.brand-logo {
  font-size: 2.1rem;
  letter-spacing: 0.18em;
  font-weight: 400;
  color: var(--color-cultist-robe);
  text-transform: uppercase;
}
.nav-links {
  display: flex;
  gap: 24px;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.82rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.nav-links a {
  color: var(--color-cultist-robe);
  text-decoration: none;
}
.nav-links a:hover {
  color: var(--color-red-clay);
}
.hero-grid {
  max-width: var(--max-container);
  margin: 28px auto;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: var(--grid-gap);
}
.hero-sunset-panel {
  border-radius: var(--radius-card);
  min-height: 420px;
  padding: 40px;
  display: flex;
  align-items: center;
  background: linear-gradient(
    145deg,
    var(--color-cultist-robe) 0%,
    var(--color-witch-soup) 42%,
    var(--color-red-clay) 76%,
    var(--color-sunspark) 100%
  );
  position: relative;
  overflow: hidden;
}
.hero-sun-disc {
  position: absolute;
  right: 14%;
  top: 22%;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  background: radial-gradient(
    circle,
    var(--color-sunspark) 0%,
    var(--color-red-clay) 75%
  );
  opacity: 0.85;
}
.hero-floating-card {
  background-color: var(--color-cultured);
  color: var(--color-cultist-robe);
  padding: 36px;
  border-radius: var(--radius-card);
  max-width: 430px;
  position: relative;
  z-index: 2;
  box-shadow: 0 16px 36px rgba(51, 18, 51, 0.25);
}
.eyebrow {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--color-witch-soup);
  margin-bottom: 8px;
}
.hero-floating-card h1 {
  font-size: 2rem;
  font-weight: 400;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}
.hero-floating-card p {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.9rem;
  color: var(--color-witch-soup);
  margin-bottom: 20px;
}
.btn-pill {
  display: inline-block;
  background-color: var(--color-cultist-robe);
  color: var(--color-cultured);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 12px 26px;
  border-radius: var(--radius-pill);
  border: none;
  cursor: pointer;
  transition: background-color var(--transition-fast);
}
.btn-pill:hover {
  background-color: var(--color-red-clay);
}
.btn-pill-secondary {
  background-color: var(--color-witch-soup);
  color: var(--color-sunspark);
}
.hero-side-panel {
  background-color: var(--color-malted-milk);
  border-radius: var(--radius-card);
  padding: 32px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.intro-statement {
  max-width: 720px;
  margin: 56px auto;
  text-align: center;
  padding: 0 24px;
}
.intro-statement h2 {
  font-size: 1.85rem;
  font-weight: 400;
  margin-bottom: 12px;
}
.intro-statement p {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.92rem;
  color: var(--color-witch-soup);
}
.section-container {
  max-width: var(--max-container);
  margin: 48px auto;
  padding: 0 24px;
}
.section-heading {
  text-align: center;
  margin-bottom: 32px;
}
.section-heading h2 {
  font-size: 1.8rem;
  font-weight: 400;
  margin-bottom: 6px;
}
.section-heading p {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.86rem;
  color: var(--color-witch-soup);
}
.category-trio {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--grid-gap);
}
.category-card {
  border-radius: var(--radius-card);
  padding: 32px 24px;
  min-height: 250px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  color: var(--color-cultured);
  background: linear-gradient(
    180deg,
    var(--color-witch-soup) 0%,
    var(--color-cultist-robe) 100%
  );
  position: relative;
}
.category-card h3 {
  font-size: 1.4rem;
  font-weight: 400;
  margin-bottom: 4px;
}
.category-card span {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.8rem;
  color: var(--color-sunspark);
}
.product-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--grid-gap);
}
.product-card {
  background-color: var(--color-malted-milk);
  border-radius: var(--radius-card);
  padding: 20px;
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.badge-stack {
  display: flex;
  gap: 6px;
  margin-bottom: 14px;
}
.badge-clay {
  background-color: var(--color-red-clay);
  color: var(--color-cultured);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.68rem;
  padding: 3px 9px;
  border-radius: var(--radius-pill);
  text-transform: uppercase;
}
.badge-plum {
  background-color: var(--color-witch-soup);
  color: var(--color-sunspark);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.68rem;
  padding: 3px 9px;
  border-radius: var(--radius-pill);
  text-transform: uppercase;
}
.garment-preview {
  height: 170px;
  border-radius: 8px;
  background-color: var(--color-cultured);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  font-family: monospace;
  font-size: 0.78rem;
  color: var(--color-witch-soup);
  padding: 12px;
  text-align: center;
}
.product-title {
  font-size: 1rem;
  margin-bottom: 4px;
}
.product-meta {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.8rem;
  color: var(--color-witch-soup);
  margin-bottom: 10px;
}
.swatch-row {
  display: flex;
  gap: 6px;
}
.swatch-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 1px solid var(--color-cultist-robe);
}
.promo-split {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--grid-gap);
}
.promo-card {
  border-radius: var(--radius-card);
  padding: 36px;
  background: linear-gradient(
    135deg,
    var(--color-witch-soup) 0%,
    var(--color-red-clay) 100%
  );
  color: var(--color-cultured);
}
.promo-card h3 {
  font-size: 1.6rem;
  font-weight: 400;
  margin-bottom: 8px;
}
.promo-card p {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.88rem;
  margin-bottom: 18px;
}
.error-banner {
  display: none;
  max-width: var(--max-container);
  margin: 18px auto;
  padding: 16px 24px;
  border-radius: var(--radius-card);
  background-color: var(--color-red-clay);
  color: var(--color-cultured);
  font-family: system-ui, -apple-system, sans-serif;
}
.coastal-banner {
  max-width: var(--max-container);
  margin: 56px auto;
  padding: 64px 32px;
  border-radius: var(--radius-card);
  text-align: center;
  background: linear-gradient(
    160deg,
    var(--color-cultist-robe) 0%,
    var(--color-witch-soup) 45%,
    var(--color-red-clay) 80%,
    var(--color-sunspark) 100%
  );
  color: var(--color-cultured);
}
.coastal-banner h2 {
  font-size: 2.4rem;
  letter-spacing: 0.14em;
  font-weight: 400;
  margin-bottom: 12px;
}
.site-footer {
  background-color: var(--color-malted-milk);
  color: var(--color-cultist-robe);
  padding: 48px 24px;
  margin-top: 64px;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.84rem;
}
.footer-grid {
  max-width: var(--max-container);
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--grid-gap);
}
@media (max-width: 900px) {
  .hero-grid,
  .category-trio,
  .product-grid,
  .promo-split,
  .footer-grid {
    grid-template-columns: 1fr;
  }
}
</style>
</head>
<body>
  <div class="utility-bar">
    <span>UK/EU GDPR Residency: eu-west-2 | OWASP ASVS v5.0 Enforced</span>
    <span id="slot-indicator" class="utility-pill">Active Slot: BLUE</span>
  </div>

  <header class="site-header">
    <div class="brand-logo">SAIKASÍ</div>
    <nav class="nav-links">
      <a href="#stages">Pipeline Stages</a>
      <a href="#top-picks">Top Picks (EFD)</a>
      <a href="#governance">GDPR &amp; Pre-Deploy</a>
    </nav>
    <button class="btn-pill" onclick="runSyntheticDemo()">
      Run Synthetic Swimsuit Demo
    </button>
  </header>

  <div id="error-screen" class="error-banner"></div>

  <section class="hero-grid">
    <div class="hero-sunset-panel">
      <div class="hero-sun-disc"></div>
      <div class="hero-floating-card">
        <div class="eyebrow">Product Development Launch For Saikasí</div>
        <h1>Hipster Bikini &amp; EFD Atelier</h1>
        <p>
          Scale-invariant Elliptical Fourier Descriptors and morphological
          garment extraction with EXIF stripping and zero-PII ledgering.
        </p>
        <input
          type="file"
          id="image-upload"
          accept="image/png,image/jpeg,image/webp"
          style="margin-bottom: 14px; font-size: 0.8rem;"
        >
        <br>
        <button class="btn-pill" onclick="uploadAndAnalyze()">
          Explore &amp; Analyze Upload
        </button>
      </div>
    </div>

    <div class="hero-side-panel">
      <div>
        <div class="eyebrow">MP229 Coastal Sunset Palette</div>
        <h3 style="font-size: 1.4rem; font-weight: 400; margin-bottom: 10px;">
          Live Telemetry &amp; Rate Gate
        </h3>
        <p style="font-family: sans-serif; font-size: 0.85rem;">
          Every upload is verified against magic-byte headers, pixel
          decompression ceilings, and HKDF-SHA256 domain pseudonymisation.
        </p>
      </div>
      <div id="telemetry-summary" style="font-family: monospace; font-size: 0.82rem;">
        Status: Ready | Items Extracted: 0
      </div>
    </div>
  </section>

  <section class="intro-statement">
    <h2>Hello! We're Saikasí Morphometric Store</h2>
    <p>
      Bridging high-fashion swimwear curation with mathematical contour
      harmonics. Every silhouette is decomposed into 10 normalized Elliptical
      Fourier Descriptors inside a strict UK/EU privacy boundary.
    </p>
  </section>

  <section id="stages" class="section-container">
    <div class="category-trio">
      <div class="category-card">
        <h3>01. Bikinis — Preprocess</h3>
        <span>Gaussian Blur &amp; Otsu Morphological Closure</span>
      </div>
      <div class="category-card">
        <h3>02. Bottoms — Contours</h3>
        <span>Connected-Component RegionProps &amp; BBox Isolation</span>
      </div>
      <div class="category-card">
        <h3>03. One Pieces — Harmonics</h3>
        <span>Scale-Invariant Elliptical Fourier Descriptors (EFD)</span>
      </div>
    </div>
  </section>

  <section id="top-picks" class="section-container">
    <div class="section-heading">
      <h2>Top Picks — Extracted Garment Harmonics</h2>
      <p>Live morphometric descriptors rendered from the Src pipeline</p>
    </div>
    <div id="items-grid" class="product-grid"></div>
  </section>

  <section id="governance" class="section-container">
    <div class="promo-split">
      <div class="promo-card">
        <div class="eyebrow" style="color: var(--color-sunspark);">
          Hello Summer // UK &amp; EU GDPR Art. 17
        </div>
        <h3>Right to Erasure &amp; Audit Ledger</h3>
        <p>
          Recompute HKDF-SHA256 subject tokens and purge matching records
          across SQLite B-tree tables and hash-chained JSONL audit logs.
        </p>
        <button class="btn-pill btn-pill-secondary" onclick="triggerErasure()">
          Execute Art. 17 Erasure
        </button>
      </div>
      <div class="promo-card">
        <div class="eyebrow" style="color: var(--color-sunspark);">
          Here's To You // Pre-Deploy #10
        </div>
        <h3>Blue-Green Slot Instant Rollback</h3>
        <p>
          Promote standby release slots with automated health probes or
          trigger an instant zero-downtime traffic rollback.
        </p>
        <button class="btn-pill btn-pill-secondary" onclick="triggerRollback()">
          Instant Slot Rollback
        </button>
      </div>
    </div>
  </section>

  <section class="coastal-banner">
    <div class="eyebrow" style="color: var(--color-sunspark);">
      Product Development Launch For
    </div>
    <h2>SAIKASÍ</h2>
    <p style="font-family: sans-serif; font-size: 0.9rem; margin-bottom: 20px;">
      Everyday For Your Morphometric Precision — Flutter &amp; Web Ready
    </p>
  </section>

  <footer class="site-footer">
    <div class="footer-grid">
      <div>
        <strong>SAIKASÍ // SRC</strong>
        <p style="margin-top: 8px;">
          Elliptical Fourier Descriptor Swimwear Extraction Pipeline.
        </p>
      </div>
      <div>
        <strong>Palette Swatches (MP229)</strong>
        <p style="margin-top: 8px;">
          #F6F4F5 | #E8D9CE | #FBCA69 | #E45E32 | #692746 | #331233
        </p>
      </div>
      <div>
        <strong>OWASP ASVS v5.0</strong>
        <p style="margin-top: 8px;">
          CWE-22 Path Guard, CWE-400 Bomb Guard, CWE-502 Safe YAML.
        </p>
      </div>
      <div>
        <strong>10-Point Readiness</strong>
        <p style="margin-top: 8px;">
          BOLA HMAC, Sliding Rate Limit, B-Tree SQLite, Ring Buffer Logs.
        </p>
      </div>
    </div>
  </footer>

<script>
const UI_CONSTANTS = Object.freeze({
  ORIGIN_HEADER: window.location.origin || "http://127.0.0.1:8000",
  DEFAULT_OWNER: "saikasi_atelier_operator"
});

function showErrorBanner(payload) {
  const banner = document.getElementById("error-screen");
  banner.style.display = "block";
  banner.textContent =
    (payload.screen_title || "Service Notice") +
    " — " +
    (payload.user_message || "Request could not be completed.");
}

function clearErrorBanner() {
  const banner = document.getElementById("error-screen");
  banner.style.display = "none";
  banner.textContent = "";
}

function renderExtractedItems(items) {
  const grid = document.getElementById("items-grid");
  grid.innerHTML = "";
  const list = Array.isArray(items) ? items : [];
  for (let i = 0; i < list.length; i += 1) {
    const item = list[i];
    const card = document.createElement("div");
    card.className = "product-card";
    card.innerHTML =
      '<div class="badge-stack">' +
      '<span class="badge-clay">Item #' + item.item_id + '</span>' +
      '<span class="badge-plum">EFD-10</span>' +
      '</div>' +
      '<div class="garment-preview">' +
      'Area: ' + item.area + ' px²<br>' +
      'Perim: ' + item.perimeter + ' px<br>' +
      'BBox: [' + item.bbox_x + ', ' + item.bbox_y + ', ' +
      item.bbox_w + ', ' + item.bbox_h + ']' +
      '</div>' +
      '<div>' +
      '<div class="product-title">Silhouette Contour #' + item.item_id + '</div>' +
      '<div class="product-meta">' +
      'Eccentricity: ' + item.eccentricity +
      ' | Solidity: ' + item.solidity +
      '<br>a1: ' + item.a_1 + ' | d1: ' + item.d_1 +
      '</div>' +
      '<div class="swatch-row">' +
      '<span class="swatch-dot" style="background:#E45E32"></span>' +
      '<span class="swatch-dot" style="background:#FBCA69"></span>' +
      '<span class="swatch-dot" style="background:#692746"></span>' +
      '<span class="swatch-dot" style="background:#331233"></span>' +
      '</div>' +
      '</div>';
    grid.appendChild(card);
  }
}

async function runSyntheticDemo() {
  clearErrorBanner();
  const resp = await fetch("/demo-synthetic", {
    method: "POST",
    headers: {"X-Owner-Id": UI_CONSTANTS.DEFAULT_OWNER}
  });
  const data = await resp.json();
  if (!resp.ok) {
    showErrorBanner(data);
    return;
  }
  document.getElementById("telemetry-summary").textContent =
    "Batch: " + data.batch_id + " | Items Extracted: " + data.item_count;
  renderExtractedItems(data.items);
}

async function uploadAndAnalyze() {
  clearErrorBanner();
  const input = document.getElementById("image-upload");
  if (!input.files || input.files.length === 0) {
    showErrorBanner({
      screen_title: "400 - Invalid Request Payload",
      user_message: "Please select a PNG, JPEG, or WebP swimsuit image first."
    });
    return;
  }
  const formData = new FormData();
  formData.append("file", input.files[0]);
  const resp = await fetch("/analyze", {
    method: "POST",
    headers: {"X-Owner-Id": UI_CONSTANTS.DEFAULT_OWNER},
    body: formData
  });
  const data = await resp.json();
  if (!resp.ok) {
    showErrorBanner(data);
    return;
  }
  document.getElementById("telemetry-summary").textContent =
    "Token: " + data.filename_token + " | Items: " + data.item_count;
  renderExtractedItems(data.items);
}

async function triggerErasure() {
  clearErrorBanner();
  const resp = await fetch(
    "/gdpr/erasure?subject_id=" +
      encodeURIComponent(UI_CONSTANTS.DEFAULT_OWNER),
    {method: "POST"}
  );
  const data = await resp.json();
  if (!resp.ok) {
    showErrorBanner(data);
    return;
  }
  document.getElementById("telemetry-summary").textContent =
    "GDPR Art. 17 Erased Rows: " + data.db_rows_erased;
  renderExtractedItems([]);
}

async function triggerRollback() {
  clearErrorBanner();
  const resp = await fetch("/deploy/rollback", {method: "POST"});
  const data = await resp.json();
  if (!resp.ok) {
    showErrorBanner(data);
    return;
  }
  document.getElementById("slot-indicator").textContent =
    "Active Slot: " + String(data.active_slot).toUpperCase();
}

window.addEventListener("DOMContentLoaded", () => {
  runSyntheticDemo();
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_editorial_storefront() -> HTMLResponse:
    """Serves the SAIKASI editorial swimsuit EFD storefront UI."""
    return HTMLResponse(
        content=SAIKASI_EDITORIAL_HTML, status_code=HTTP_STATUS_OK
    )


@app.get("/health")
async def health_check(
    request: Request,
    origin: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Provides a rate-limited health and deployment slot check endpoint.

    Args:
        request (Request): Incoming HTTP request.
        origin (Optional[str], optional): Optional CORS Origin header.

    Returns:
        JSONResponse: Health status, active blue/green slot, and alerts.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        remaining = rate_limiter.check_and_record(client_key)
        ring = get_ring_buffer_handler()
        alert_count = (
            len(ring.get_dispatched_alerts()) if ring is not None else ZERO_INT
        )
        return {
            "status": "healthy",
            "service": "src-api",
            "active_slot": deployment_manager.active_slot,
            "rate_limit_remaining": remaining,
            "alerts_dispatched": alert_count,
        }

    validated_origin: Optional[str] = None
    if origin is not None:
        try:
            validated_origin = verify_cors_origin(origin, ALLOWED_CORS_ORIGINS)
        except PreDeployHTTPException as cors_exc:
            status, err_body = execute_with_catch_all_boundary(
                lambda: (_ for _ in ()).throw(cors_exc)
            )
            return _build_json_response(status, err_body)

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload, validated_origin)


@app.post("/demo-synthetic")
async def analyze_synthetic_demo(
    request: Request,
    x_owner_id: str = Header(default="saikasi_atelier_operator"),
) -> JSONResponse:
    """Runs the Src pipeline on a zero-PII synthetic swimsuit canvas.

    Args:
        request (Request): Incoming HTTP request.
        x_owner_id (str, optional): Caller owner identifier for BOLA.

    Returns:
        JSONResponse: Extracted apparel items and pseudonymised batch ID.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        clean_owner = sanitize_and_validate_input(x_owner_id)
        owner_token = pipeline_instance.gdpr_engine.pseudonymise_identifier(
            clean_owner, domain="subject"
        )
        synthetic_img = sandbox_adapter.get_sandbox_image()
        collection = pipeline_instance.process(synthetic_img)
        items_list = collection.to_dict_list()
        batch_id = pipeline_instance.gdpr_engine.pseudonymise_identifier(
            f"{clean_owner}:synthetic", domain="batch_id"
        )
        batch_store.save_batch(
            batch_id=batch_id, owner_id=owner_token, items_payload=items_list
        )
        return {
            "success": True,
            "batch_id": batch_id,
            "owner_id": owner_token,
            "item_count": len(collection),
            "items": items_list,
        }

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


@app.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    x_owner_id: str = Header(default="saikasi_atelier_operator"),
    origin: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Analyzes an uploaded swimsuit image for Flutter/Web clients.

    Args:
        request (Request): Incoming HTTP request.
        file (UploadFile): Uploaded multipart image file.
        x_owner_id (str, optional): Caller owner ID for BOLA binding.
        origin (Optional[str], optional): Optional CORS Origin header.

    Returns:
        JSONResponse: Pseudonymised filename token and EFD item records.
    """
    client_key = _resolve_client_key(request)
    raw_bytes = await file.read()

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        validated_origin: Optional[str] = None
        if origin is not None:
            validated_origin = verify_cors_origin(origin, ALLOWED_CORS_ORIGINS)

        content_type = str(file.content_type or "")
        if not content_type.startswith("image/"):
            raise PreDeployHTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST,
                error_code="INVALID_MIME_TYPE",
                user_message="Uploaded file must have an image/* content type.",
            )

        raw_filename = sanitize_and_validate_input(
            str(file.filename or "upload.png")
        )
        clean_owner = sanitize_and_validate_input(x_owner_id)
        owner_token = pipeline_instance.gdpr_engine.pseudonymise_identifier(
            clean_owner, domain="subject"
        )
        filename_token = pipeline_instance.gdpr_engine.pseudonymise_identifier(
            raw_filename, domain="upload_filename"
        )

        collection = pipeline_instance.process(raw_bytes)
        items_list = collection.to_dict_list()
        batch_store.save_batch(
            batch_id=filename_token,
            owner_id=owner_token,
            items_payload=items_list,
        )
        return {
            "success": True,
            "filename_token": filename_token,
            "owner_id": owner_token,
            "cors_origin": validated_origin,
            "item_count": len(collection),
            "items": items_list,
        }

    status_code, payload = execute_with_catch_all_boundary(_op)
    cors_hdr = payload.pop("cors_origin", None)
    return _build_json_response(status_code, payload, cors_hdr)


@app.get("/batch/{batch_id}")
async def get_saved_batch(
    batch_id: str,
    request: Request,
    x_owner_id: str = Header(default="saikasi_atelier_operator"),
) -> JSONResponse:
    """Retrieves a persisted batch after verifying BOLA ownership.

    Args:
        batch_id (str): Target batch identifier.
        request (Request): Incoming HTTP request.
        x_owner_id (str, optional): Requesting user identifier.

    Returns:
        JSONResponse: Batch record or 403/404 error screen payload.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        clean_owner = sanitize_and_validate_input(x_owner_id)
        owner_token = pipeline_instance.gdpr_engine.pseudonymise_identifier(
            clean_owner, domain="subject"
        )
        record = batch_store.fetch_batch_for_owner(
            batch_id=batch_id, requesting_user_id=owner_token
        )
        return {"success": True, **record}

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


@app.post("/gdpr/erasure")
async def gdpr_erasure_endpoint(
    request: Request,
    subject_id: str = Query(...),
) -> JSONResponse:
    """Executes a GDPR Article 17 Right to Erasure request.

    Args:
        request (Request): Incoming HTTP request.
        subject_id (str): Subject identifier requesting erasure.

    Returns:
        JSONResponse: Summary of erased SQLite rows and audit hash.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        clean_subject = sanitize_and_validate_input(subject_id)
        result = pipeline_instance.gdpr_engine.execute_right_to_erasure(
            raw_subject_id=clean_subject,
            domain="subject",
            sqlite_db_path=str(
                _pd_cfg.get("sqlite_db_path", "src_records.sqlite3")
            ),
        )
        return {"success": True, **result}

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


@app.post("/auth/reset-request")
async def issue_password_reset(
    request: Request,
    user_id: str = Query(...),
) -> JSONResponse:
    """Issues a single-use, 30-minute expiring password reset link token.

    Args:
        request (Request): Incoming HTTP request.
        user_id (str): Account identifier requesting reset.

    Returns:
        JSONResponse: Signed 30-minute reset token.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        token = reset_manager.issue_reset_token(user_id)
        return {
            "success": True,
            "ttl_seconds": int(
                _pd_cfg.get("password_reset_ttl_seconds", 1800)
            ),
            "reset_token": token,
        }

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


@app.post("/auth/reset-consume")
async def consume_password_reset(
    request: Request,
    token: str = Query(...),
) -> JSONResponse:
    """Consumes a single-use 30-minute password reset token.

    Args:
        request (Request): Incoming HTTP request.
        token (str): Reset token string.

    Returns:
        JSONResponse: Verified user_id or HTTP 410 if expired/reused.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        verified_user = reset_manager.consume_reset_token(token)
        return {"success": True, "user_id": verified_user}

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


@app.post("/deploy/rollback")
async def trigger_instant_rollback(request: Request) -> JSONResponse:
    """Triggers an instant blue-green traffic rollback to the standby slot.

    Args:
        request (Request): Incoming HTTP request.

    Returns:
        JSONResponse: Updated slot status.
    """
    client_key = _resolve_client_key(request)

    def _op() -> Dict[str, Any]:
        rate_limiter.check_and_record(client_key)
        status = deployment_manager.instant_rollback()
        return {"success": True, **status}

    status_code, payload = execute_with_catch_all_boundary(_op)
    return _build_json_response(status_code, payload)


def encode_synthetic_png_bytes() -> bytes:
    """Encodes the synthetic swimsuits image into valid PNG bytes for testing.

    Returns:
        bytes: PNG-encoded byte buffer.
    """
    rgb_arr = generate_synthetic_swimsuits_image()
    pil_img = Image.fromarray(rgb_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()