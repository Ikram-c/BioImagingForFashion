# BioImagingForFashion
a test repo to see how effective my bio imaging pipelines are for fashion market research


flowchart TD
    subgraph Client_Tier["Untrusted Client Tier (Flutter UI / Tactical Web Client)"]
        A["Client Request (Multipart Image / REST API / UI)"]
    end

    subgraph Edge_PreDeploy["Trust Boundary 1: 10-Point Pre-Deployment Gate (src/predeploy.py)"]
        B["Strict CORS Origin Allowlist Check"]
        C["Sliding-Window Rate Limiter (HTTP 429)"]
        D["SQLi / XSS Input Sanitizer (HTTP 400)"]
        E["BOLA Owner Verification via HMAC Compare (HTTP 403)"]
        F["30-Min Single-Use Reset Token Gate (HTTP 410)"]
    end

    subgraph Security_GDPR["Trust Boundary 2: OWASP ASVS & GDPR Privacy Engine (src/security.py & gdpr.py)"]
        G["Path Canonicalization & Decompression Bomb Guard"]
        H["EXIF Metadata Stripper & Clean Pixel Buffer Clone"]
        I["Irreversible Block-Quantized Visual Blur on Sensitive ROIs"]
        J["HKDF-SHA256 Domain Pseudonymiser & H3 Spatial Anonymiser (U=0.0, Laplace DP)"]
    end

    subgraph Pipeline_Core["Trust Boundary 3: Declarative Stage Registry & Bounded EFD Core (src/pipeline.py & fourier.py)"]
        K["Stage 1: Grayscale & Gaussian Smoothing"]
        L["Stage 2: Otsu Binary Inversion & Elliptical Morphology"]
        M["Stage 3: Connected-Component Labeling & Bounded Contour Extraction"]
        N["Stage 4: Bounded Elliptical Fourier Descriptors (EFD) Computation"]
    end

    subgraph Storage_Ops["Trust Boundary 4: Indexed Persistence, Audit Ledger & Blue-Green Ops"]
        O["SQLite Store with Targeted B-Tree Indexes (owner_id, batch_id)"]
        P["SHA-256 Hash-Chained JSONL Accountability Ledger & Art. 17 Erasure"]
        Q["Cost-Capped Ring Buffer Logger & Immediate On-Call Alert Sink"]
        R["Blue-Green Slot Router & Cloud Sandbox (Azurite / Fake-GCS)"]
    end

    A --> B --> C --> D --> E --> F
    F --> G --> H --> I --> J
    J --> K --> L --> M --> N
    N --> O & P & Q & R
