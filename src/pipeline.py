"""Declarative stage-registry pipeline for Src apparel extraction."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from skimage.measure import label, regionprops

from src.constants import (
    DEFAULT_CONFIG_FILENAME,
    GRAY_DIMENSIONS,
    GRAY_MAX_VALUE,
    MAX_CONTOUR_POINTS_CEILING,
    MAX_HARMONICS_CEILING,
    MAX_IMAGE_DIMENSION_CEILING,
    MAX_IMAGE_PIXELS_CEILING,
    MAX_ITEMS_CEILING,
    MAX_STAGES_CEILING,
    MIN_CONTOUR_POINTS_FLOOR,
    ONE_INT,
    RGB_CHANNELS,
    TWO_INT,
    ZERO_INT,
    verify_condition,
)
from src.fourier import compute_elliptical_fourier_descriptors
from src.gdpr import (
    GDPRComplianceEngine,
    apply_irreversible_roi_blur,
    strip_exif_clone_pixels,
)
from src.logging_config import configure_logging
from src.models import (
    ApparelCollection,
    ApparelItem,
    CandidateContourRegion,
    PipelineBatchState,
)
from src.security import (
    decode_and_validate_image_bytes,
    load_safe_yaml,
    resolve_safe_path,
)


StageHandler = Callable[[PipelineBatchState], PipelineBatchState]


class SrcPipeline:
    """Data-driven pipeline isolating apparel items and extracting EFDs."""

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_FILENAME,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the pipeline and stage registry from config.yaml.

        Args:
            config_path (Union[str, Path], optional): Path to config.yaml.
                Defaults to DEFAULT_CONFIG_FILENAME.
            workspace_root (Optional[Union[str, Path]], optional): Workspace
                containment directory. Defaults to None.
        """
        self._workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root is not None
            else Path.cwd().resolve()
        )
        self._config = self._load_config(config_path)

        p_cfg = self._config["pipeline"]
        f_cfg = self._config["fourier"]
        l_cfg = self._config["logging"]
        s_cfg = self._config.get("security", {})
        g_cfg = self._config.get("gdpr", {})
        pd_cfg = self._config.get("predeploy", {})

        self._harmonics = int(p_cfg["num_harmonics"])
        self._min_area = float(p_cfg["min_area"])
        self._normalize = bool(p_cfg["normalize_efd"])
        self._blur_kernel: Tuple[int, int] = (
            int(p_cfg["gaussian_blur_kernel"][ZERO_INT]),
            int(p_cfg["gaussian_blur_kernel"][ONE_INT]),
        )
        self._morph_kernel_size: Tuple[int, int] = (
            int(p_cfg["morphology_kernel_size"][ZERO_INT]),
            int(p_cfg["morphology_kernel_size"][ONE_INT]),
        )
        self._morph_iterations = int(p_cfg["morphology_iterations"])
        self._max_items = min(
            int(p_cfg["max_items_per_image"]), MAX_ITEMS_CEILING
        )
        self._configured_stages: Tuple[str, ...] = tuple(
            p_cfg.get(
                "stages",
                (
                    "preprocess_segment",
                    "extract_contours",
                    "compute_descriptors",
                ),
            )
        )

        self._max_contour_points = min(
            int(f_cfg["max_contour_points"]), MAX_CONTOUR_POINTS_CEILING
        )
        self._max_harmonics_allowed = min(
            int(f_cfg["max_harmonics_allowed"]), MAX_HARMONICS_CEILING
        )
        self._zero_epsilon = float(f_cfg["zero_displacement_epsilon"])
        self._min_contour_points = max(
            int(f_cfg.get("min_contour_points", MIN_CONTOUR_POINTS_FLOOR)),
            MIN_CONTOUR_POINTS_FLOOR,
        )

        self._max_upload_bytes = int(
            s_cfg.get("max_upload_bytes", 5242880)
        )
        self._max_image_width = int(
            s_cfg.get("max_image_width", MAX_IMAGE_DIMENSION_CEILING)
        )
        self._max_image_height = int(
            s_cfg.get("max_image_height", MAX_IMAGE_DIMENSION_CEILING)
        )
        self._max_image_pixels = int(
            s_cfg.get("max_image_pixels", MAX_IMAGE_PIXELS_CEILING)
        )

        self._gdpr_engine = GDPRComplianceEngine(
            cloud_region=str(g_cfg.get("cloud_region", "eu-west-2")),
            allowed_regions=tuple(
                g_cfg.get(
                    "allowed_regions",
                    ("eu-west-1", "eu-west-2", "eu-central-1"),
                )
            ),
            key_epoch=str(g_cfg.get("key_epoch", "2026-Q4")),
            retention_ttl_seconds=int(
                g_cfg.get("retention_ttl_seconds", 86400)
            ),
            k_anonymity_threshold=int(
                g_cfg.get("k_anonymity_threshold", 3)
            ),
            dp_epsilon=float(g_cfg.get("dp_epsilon", 0.8)),
            blur_block_size=int(g_cfg.get("blur_block_size", 8)),
            ledger_file=str(
                g_cfg.get("ledger_file", "src_audit_ledger.jsonl")
            ),
            workspace_root=self._workspace_root,
        )

        self._logger = configure_logging(
            level=str(l_cfg.get("level", "INFO")),
            enabled=bool(l_cfg.get("enabled", True)),
            ring_capacity=int(
                pd_cfg.get("log_ring_buffer_capacity", 256)
            ),
            max_detail_length=int(
                pd_cfg.get("log_max_detail_length", 180)
            ),
        )

        self._stage_registry: Dict[str, StageHandler] = {
            "preprocess_segment": self._stage_preprocess_segment,
            "extract_contours": self._stage_extract_contours,
            "compute_descriptors": self._stage_compute_descriptors,
        }
        verify_condition(
            ONE_INT <= len(self._configured_stages) <= MAX_STAGES_CEILING,
            "Configured pipeline stage count is outside bounds.",
        )
        for idx in range(len(self._configured_stages)):
            stage_name = self._configured_stages[idx]
            verify_condition(
                stage_name in self._stage_registry,
                f"Unknown stage '{stage_name}' in configuration.",
            )

        self._logger.debug("Initialized hardened SrcPipeline.")

    @property
    def gdpr_engine(self) -> GDPRComplianceEngine:
        """Returns the pipeline's UK/EU GDPR compliance engine."""
        return self._gdpr_engine

    @property
    def config(self) -> Dict[str, Any]:
        """Returns a copy of the loaded configuration dictionary."""
        return dict(self._config)

    def _load_config(
        self, config_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Loads and validates the YAML configuration file safely.

        Args:
            config_path (Union[str, Path]): File path to config.yaml.

        Returns:
            Dict[str, Any]: Validated configuration dictionary.
        """
        cfg = load_safe_yaml(
            config_path, workspace_root=self._workspace_root
        )
        verify_condition(
            "pipeline" in cfg, "Missing 'pipeline' section in config.yaml."
        )
        verify_condition(
            "fourier" in cfg, "Missing 'fourier' section in config.yaml."
        )
        verify_condition(
            "logging" in cfg, "Missing 'logging' section in config.yaml."
        )
        return cfg

    def __repr__(self) -> str:
        """Returns a formal representation of the pipeline configuration."""
        return (
            f"<SrcPipeline(harmonics={self._harmonics}, "
            f"min_area={self._min_area}, normalize={self._normalize})>"
        )

    def _load_image(
        self,
        input_source: Union[str, Path, bytes, np.ndarray],
        sensitive_rois: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """Loads, validates, strips EXIF, and optionally blurs an input image.

        Args:
            input_source (Union[str, Path, bytes, np.ndarray]): Path, raw
                bytes, or NumPy array.
            sensitive_rois (Optional[Sequence[Tuple[int, int, int, int]]]):
                Optional bounding boxes for irreversible GDPR visual blur.

        Returns:
            np.ndarray: Clean, validated RGB NumPy image array.
        """
        verify_condition(
            input_source is not None, "Image input source cannot be None."
        )

        rgb_array: Optional[np.ndarray] = None

        if isinstance(input_source, (bytes, bytearray)):
            rgb_array = decode_and_validate_image_bytes(
                bytes(input_source),
                max_bytes=self._max_upload_bytes,
                max_width=self._max_image_width,
                max_height=self._max_image_height,
                max_pixels=self._max_image_pixels,
            )
        elif isinstance(input_source, (str, Path)):
            safe_img_path = resolve_safe_path(
                input_source,
                workspace_root=self._workspace_root,
                must_exist=True,
            )
            pseudo_file = self._gdpr_engine.pseudonymise_identifier(
                safe_img_path.name, domain="local_file"
            )
            self._logger.info(f"Loading validated image token: {pseudo_file}")
            raw_bytes = safe_img_path.read_bytes()
            rgb_array = decode_and_validate_image_bytes(
                raw_bytes,
                max_bytes=self._max_upload_bytes,
                max_width=self._max_image_width,
                max_height=self._max_image_height,
                max_pixels=self._max_image_pixels,
            )
        elif isinstance(input_source, np.ndarray):
            verify_condition(
                input_source.size > ZERO_INT,
                "Input image NumPy array cannot be empty.",
            )
            verify_condition(
                input_source.ndim in (GRAY_DIMENSIONS, RGB_CHANNELS),
                "Input NumPy array must be 2D grayscale or 3D RGB.",
            )
            img_h, img_w = input_source.shape[:GRAY_DIMENSIONS]
            verify_condition(
                ONE_INT <= img_w <= self._max_image_width
                and ONE_INT <= img_h <= self._max_image_height
                and (img_w * img_h) <= self._max_image_pixels,
                "Input NumPy array dimensions exceed security bounds.",
                use_security_error=True,
            )
            if input_source.ndim == GRAY_DIMENSIONS:
                rgb_array = cv2.cvtColor(
                    input_source.astype(np.uint8), cv2.COLOR_GRAY2RGB
                )
            else:
                rgb_array = input_source.astype(np.uint8).copy()
        else:
            raise TypeError(
                f"Unsupported image input type: {type(input_source).__name__}"
            )

        clean_rgb = strip_exif_clone_pixels(rgb_array)
        if sensitive_rois is not None and len(sensitive_rois) > ZERO_INT:
            clean_rgb = apply_irreversible_roi_blur(
                clean_rgb, sensitive_bboxes=sensitive_rois
            )
        return clean_rgb

    def _stage_preprocess_segment(
        self, state: PipelineBatchState
    ) -> PipelineBatchState:
        """Stage 1: Grayscale conversion, Gaussian blur, Otsu, and morphology.

        Args:
            state (PipelineBatchState): Current immutable batch state.

        Returns:
            PipelineBatchState: Updated batch state with labeled_mask.
        """
        rgb_image = state.rgb_image
        verify_condition(
            rgb_image.ndim == RGB_CHANNELS,
            "Stage 1 requires a 3-channel RGB image array.",
        )

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, self._blur_kernel, ZERO_INT)

        _, thresh = cv2.threshold(
            blurred,
            ZERO_INT,
            GRAY_MAX_VALUE,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self._morph_kernel_size
        )
        cleaned = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self._morph_iterations,
        )
        verify_condition(
            cleaned.shape == gray.shape,
            "Preprocessed morphology mask shape mismatch.",
        )

        labeled_matrix = label(cleaned)
        return PipelineBatchState(
            rgb_image=state.rgb_image,
            labeled_mask=labeled_matrix,
            candidate_regions=state.candidate_regions,
            extracted_items=state.extracted_items,
            completed_stages=state.completed_stages + ("preprocess_segment",),
        )

    def _stage_extract_contours(
        self, state: PipelineBatchState
    ) -> PipelineBatchState:
        """Stage 2: Filters labeled regions and extracts external contours.

        Args:
            state (PipelineBatchState): Batch state containing labeled_mask.

        Returns:
            PipelineBatchState: Updated state with candidate_regions.
        """
        verify_condition(
            state.labeled_mask is not None,
            "Stage 2 requires a populated labeled_mask from Stage 1.",
        )
        labeled_mask = state.labeled_mask
        regions = regionprops(labeled_mask)
        self._logger.info(
            f"Discovered {len(regions)} raw segmented candidate regions."
        )

        valid_regions = [
            reg for reg in regions if float(reg.area) >= self._min_area
        ][: self._max_items]

        candidates: List[CandidateContourRegion] = []
        region_limit = min(len(valid_regions), self._max_items)

        for idx in range(region_limit):
            region = valid_regions[idx]
            item_idx = idx + ONE_INT
            mask = (
                labeled_mask == region.label
            ).astype(np.uint8) * GRAY_MAX_VALUE
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue

            contour = contours[ZERO_INT].reshape(-ONE_INT, TWO_INT)
            if len(contour) < self._min_contour_points:
                continue

            min_r, min_c, max_r, max_c = region.bbox
            bbox = (
                int(min_c),
                int(min_r),
                int(max_c - min_c),
                int(max_r - min_r),
            )
            candidates.append(
                CandidateContourRegion(
                    item_id=item_idx,
                    area=float(region.area),
                    perimeter=float(region.perimeter),
                    centroid=(
                        float(region.centroid[ZERO_INT]),
                        float(region.centroid[ONE_INT]),
                    ),
                    bounding_box=bbox,
                    eccentricity=float(region.eccentricity),
                    solidity=float(region.solidity),
                    extent=float(region.extent),
                    contour=contour,
                )
            )

        return PipelineBatchState(
            rgb_image=state.rgb_image,
            labeled_mask=state.labeled_mask,
            candidate_regions=tuple(candidates),
            extracted_items=state.extracted_items,
            completed_stages=state.completed_stages + ("extract_contours",),
        )

    def _stage_compute_descriptors(
        self, state: PipelineBatchState
    ) -> PipelineBatchState:
        """Stage 3: Computes Elliptical Fourier Descriptors for candidates.

        Args:
            state (PipelineBatchState): State containing candidate_regions.

        Returns:
            PipelineBatchState: Updated state with extracted_items.
        """
        extracted: List[ApparelItem] = []
        candidate_limit = min(len(state.candidate_regions), self._max_items)

        for idx in range(candidate_limit):
            cand = state.candidate_regions[idx]
            efd_coeffs = compute_elliptical_fourier_descriptors(
                cand.contour,
                order=self._harmonics,
                normalize=self._normalize,
                max_contour_points=self._max_contour_points,
                max_harmonics_allowed=self._max_harmonics_allowed,
                zero_displacement_epsilon=self._zero_epsilon,
            )
            item = ApparelItem(
                item_id=cand.item_id,
                area=cand.area,
                perimeter=cand.perimeter,
                centroid=cand.centroid,
                bounding_box=cand.bounding_box,
                eccentricity=cand.eccentricity,
                solidity=cand.solidity,
                extent=cand.extent,
                fourier_coefficients=efd_coeffs,
            )
            extracted.append(item)

        return PipelineBatchState(
            rgb_image=state.rgb_image,
            labeled_mask=state.labeled_mask,
            candidate_regions=state.candidate_regions,
            extracted_items=tuple(extracted),
            completed_stages=state.completed_stages
            + ("compute_descriptors",),
        )

    def process_with_state(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        subject_identifier: str = "anonymous_batch",
        sensitive_rois: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[ApparelCollection, PipelineBatchState]:
        """Executes the declarative stage registry and returns full state.

        Args:
            image_input (Union[str, Path, bytes, np.ndarray]): Input image.
            subject_identifier (str, optional): Identifier for GDPR ledger.
            sensitive_rois (Optional[Sequence[Tuple[int, int, int, int]]]):
                Optional sensitive bounding boxes to blur prior to analysis.

        Returns:
            Tuple[ApparelCollection, PipelineBatchState]: Extracted collection
                and final immutable batch state.
        """
        rgb_img = self._load_image(
            image_input, sensitive_rois=sensitive_rois
        )
        state = PipelineBatchState(rgb_image=rgb_img)

        stage_limit = min(len(self._configured_stages), MAX_STAGES_CEILING)
        for idx in range(stage_limit):
            stage_name = self._configured_stages[idx]
            handler = self._stage_registry[stage_name]
            state = handler(state)

        collection = ApparelCollection(list(state.extracted_items))
        subject_token = self._gdpr_engine.pseudonymise_identifier(
            subject_identifier, domain="pipeline_execution"
        )
        self._gdpr_engine.append_audit_ledger(
            event_type="PIPELINE_EXTRACTION_COMPLETE",
            subject_token=subject_token,
            details={
                "item_count": len(collection),
                "stages_executed": list(state.completed_stages),
            },
        )
        self._logger.info(
            f"Successfully processed {len(collection)} apparel items."
        )
        return collection, state

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        sensitive_rois: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> ApparelCollection:
        """Executes full garment extraction pipeline on given input image.

        Args:
            image_input (Union[str, Path, bytes, np.ndarray]): File path,
                raw bytes, or NumPy image array.
            sensitive_rois (Optional[Sequence[Tuple[int, int, int, int]]]):
                Optional bounding boxes for GDPR irreversible visual blur.

        Returns:
            ApparelCollection: Collection of extracted apparel items.
        """
        collection, _ = self.process_with_state(
            image_input=image_input, sensitive_rois=sensitive_rois
        )
        return collection


def src(
    image_input: Union[str, Path, bytes, np.ndarray],
    config_path: Union[str, Path] = DEFAULT_CONFIG_FILENAME,
    workspace_root: Optional[Union[str, Path]] = None,
) -> ApparelCollection:
    """Convenience functional interface for the Src processing pipeline.

    Args:
        image_input (Union[str, Path, bytes, np.ndarray]): Path to image,
            raw bytes, or image NumPy array.
        config_path (Union[str, Path], optional): Path to YAML configuration.
            Defaults to DEFAULT_CONFIG_FILENAME.
        workspace_root (Optional[Union[str, Path]], optional): Workspace root
            directory. Defaults to None.

    Returns:
        ApparelCollection: Collection of extracted apparel items.
    """
    pipeline = SrcPipeline(
        config_path=config_path, workspace_root=workspace_root
    )
    return pipeline.process(image_input)