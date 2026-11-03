"""CLI entry point for local Src demo execution and server launch."""

import os
import uvicorn

from src import SrcPipeline
from src.constants import DEFAULT_CONFIG_FILENAME
from src.sandbox import (
    generate_synthetic_swimsuits_image,
    generate_synthetic_telemetry_coords,
)
from src.security import load_safe_yaml


def run_demo_extraction(launch_server: bool = False) -> int:
    """Executes local synthetic extraction, GDPR checks, and optional server.

    Args:
        launch_server (bool, optional): Whether to start the Uvicorn server.
            Defaults to False.

    Returns:
        int: Number of extracted apparel items.
    """
    pipeline = SrcPipeline(config_path=DEFAULT_CONFIG_FILENAME)
    synthetic_image = generate_synthetic_swimsuits_image()
    collection = pipeline.process(synthetic_image)

    coords = generate_synthetic_telemetry_coords()
    spatial_res = pipeline.gdpr_engine.anonymise_spatial_coordinates(coords)
    ledger_ok = pipeline.gdpr_engine.verify_ledger_integrity()

    df = collection.to_dataframe()
    cols = [
        "item_id",
        "area",
        "perimeter",
        "eccentricity",
        "a_1",
        "b_1",
        "c_1",
        "d_1",
    ]
    print(
        f"Extracted {len(collection)} items | "
        f"H3 Uniqueness: {spatial_res.uniqueness_rate} | "
        f"Ledger Verified: {ledger_ok}"
    )
    if not df.empty:
        print(df[cols].to_string(index=False))

    if launch_server:
        cfg = load_safe_yaml(DEFAULT_CONFIG_FILENAME)
        srv_cfg = cfg["server"]
        uvicorn.run(
            "src.api:app",
            host=str(srv_cfg["host"]),
            port=int(srv_cfg["port"]),
            reload=False,
        )

    return len(collection)


def main() -> None:
    """Main entry point checking SRC_AUTO_SERVE environment toggle."""
    auto_serve = os.environ.get("SRC_AUTO_SERVE", "1") == "1"
    run_demo_extraction(launch_server=auto_serve)


if __name__ == "__main__":
    main()