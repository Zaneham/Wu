"""
H.264 Forensic Analysis Module

Provides forensic-grade analysis of H.264 bitstreams without error concealment.
Unlike standard decoders (FFmpeg, etc.) which silently correct errors, this module
detects and logs anomalies that may indicate manipulation, splicing, or re-encoding.

Key principle: detect problems, don't fix them.

References:
    ITU-T H.264 (04/2017) - Advanced video coding for generic audiovisual services
    ISO/IEC 14496-10 - MPEG-4 Part 10: Advanced Video Coding
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class AnomalyType(Enum):
    """
    Categories of H.264 bitstream anomalies.

    Each type indicates a specific kind of irregularity that may
    suggest manipulation or non-standard encoding.
    """
    # NAL-level anomalies
    NAL_INVALID_TYPE = "nal_invalid_type"
    NAL_UNEXPECTED_ORDER = "nal_unexpected_order"
    NAL_MISSING_SPS = "nal_missing_sps"
    NAL_MISSING_PPS = "nal_missing_pps"
    NAL_DUPLICATE_PARAMETER_SET = "nal_duplicate_parameter_set"
    NAL_EMULATION_PREVENTION_ERROR = "nal_emulation_prevention_error"

    # Slice-level anomalies
    SLICE_HEADER_INVALID = "slice_header_invalid"
    SLICE_QP_OUT_OF_RANGE = "slice_qp_out_of_range"
    SLICE_QP_DISCONTINUITY = "slice_qp_discontinuity"
    SLICE_UNEXPECTED_IDR = "slice_unexpected_idr"
    SLICE_MISSING_IDR = "slice_missing_idr"
    SLICE_TYPE_MISMATCH = "slice_type_mismatch"

    # Macroblock-level anomalies
    MB_QP_BOUNDARY_ANOMALY = "mb_qp_boundary_anomaly"
    MB_TYPE_INVALID = "mb_type_invalid"
    MB_COEFFICIENT_OVERFLOW = "mb_coefficient_overflow"
    MB_PREDICTION_ERROR = "mb_prediction_error"

    # Motion vector anomalies
    MV_OUT_OF_RANGE = "mv_out_of_range"
    MV_REFERENCE_INVALID = "mv_reference_invalid"
    MV_SPATIAL_DISCONTINUITY = "mv_spatial_discontinuity"
    MV_TEMPORAL_DISCONTINUITY = "mv_temporal_discontinuity"

    # Entropy coding anomalies
    VLC_INVALID_CODEWORD = "vlc_invalid_codeword"
    VLC_UNEXPECTED_END = "vlc_unexpected_end"
    CAVLC_CONTEXT_ERROR = "cavlc_context_error"

    # Reference frame anomalies
    REF_FRAME_MISSING = "ref_frame_missing"
    REF_FRAME_MISMATCH = "ref_frame_mismatch"
    REF_LIST_REORDER_ANOMALY = "ref_list_reorder_anomaly"

    # Structural anomalies (potential splice indicators)
    SPLICE_QP_REGION = "splice_qp_region"
    SPLICE_MV_REGION = "splice_mv_region"
    SPLICE_COMPRESSION_BOUNDARY = "splice_compression_boundary"


@dataclass
class Anomaly:
    """
    A single detected anomaly in the H.264 bitstream.

    Captures precise location and context for forensic reporting.
    """
    type: AnomalyType
    frame_num: int
    mb_x: Optional[int] = None  # Macroblock X position (None if frame-level)
    mb_y: Optional[int] = None  # Macroblock Y position (None if frame-level)
    nal_offset: Optional[int] = None  # Byte offset in stream
    bit_offset: Optional[int] = None  # Bit position within NAL
    severity: str = "medium"  # "low", "medium", "high", "critical"
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "frame_num": self.frame_num,
            "position": {"mb_x": self.mb_x, "mb_y": self.mb_y} if self.mb_x is not None else None,
            "stream_position": {"nal_offset": self.nal_offset, "bit_offset": self.bit_offset},
            "severity": self.severity,
            "description": self.description,
            "context": self.context,
        }


class AnomalyLog:
    """
    Accumulates and analyses anomalies detected during H.264 decode.

    This is the core forensic instrument. Rather than concealing errors
    (as playback decoders do), we log them with full context for later
    analysis. Patterns in anomalies can reveal:

    - Splice points (sudden QP/MV discontinuities)
    - Re-encoding (compression artefact patterns)
    - Bitstream manipulation (invalid VLC sequences)
    - Source mixing (different encoding characteristics)
    """

    def __init__(self):
        self.anomalies: List[Anomaly] = []
        self.frame_count = 0
        self.current_frame = 0
        self.current_nal_offset = 0

        # Statistics for pattern detection
        self._qp_history: List[float] = []  # Average QP per frame
        self._mv_magnitude_history: List[float] = []  # Average MV magnitude per frame
        self._anomaly_density: List[int] = []  # Anomalies per frame

    def set_frame(self, frame_num: int, nal_offset: int = 0):
        """Set current frame context for subsequent anomalies."""
        self.current_frame = frame_num
        self.current_nal_offset = nal_offset
        self.frame_count = max(self.frame_count, frame_num + 1)

        # Ensure history lists are long enough
        while len(self._anomaly_density) <= frame_num:
            self._anomaly_density.append(0)

    def log(
        self,
        anomaly_type: AnomalyType,
        description: str,
        mb_x: Optional[int] = None,
        mb_y: Optional[int] = None,
        bit_offset: Optional[int] = None,
        severity: str = "medium",
        **context
    ):
        """
        Log an anomaly with current frame context.

        Args:
            anomaly_type: Category of anomaly
            description: Human-readable explanation
            mb_x, mb_y: Macroblock position (if applicable)
            bit_offset: Bit position within current NAL
            severity: "low", "medium", "high", or "critical"
            **context: Additional key-value pairs for forensic detail
        """
        anomaly = Anomaly(
            type=anomaly_type,
            frame_num=self.current_frame,
            mb_x=mb_x,
            mb_y=mb_y,
            nal_offset=self.current_nal_offset,
            bit_offset=bit_offset,
            severity=severity,
            description=description,
            context=context,
        )
        self.anomalies.append(anomaly)

        # Update density tracking
        if self.current_frame < len(self._anomaly_density):
            self._anomaly_density[self.current_frame] += 1

    def log_qp(self, frame_num: int, avg_qp: float):
        """Record average QP for a frame (for discontinuity detection)."""
        while len(self._qp_history) <= frame_num:
            self._qp_history.append(0.0)
        self._qp_history[frame_num] = avg_qp

        # Check for QP discontinuity
        if frame_num > 0 and len(self._qp_history) > frame_num:
            prev_qp = self._qp_history[frame_num - 1]
            if prev_qp > 0 and abs(avg_qp - prev_qp) > 15:
                self.log(
                    AnomalyType.SLICE_QP_DISCONTINUITY,
                    f"QP jumped from {prev_qp:.1f} to {avg_qp:.1f} between frames",
                    severity="high",
                    previous_qp=prev_qp,
                    current_qp=avg_qp,
                    delta=avg_qp - prev_qp,
                )

    def log_mv_stats(self, frame_num: int, avg_magnitude: float):
        """Record average motion vector magnitude for a frame."""
        while len(self._mv_magnitude_history) <= frame_num:
            self._mv_magnitude_history.append(0.0)
        self._mv_magnitude_history[frame_num] = avg_magnitude

    def get_anomalies_at_frame(self, frame_num: int) -> List[Anomaly]:
        """Get all anomalies for a specific frame."""
        return [a for a in self.anomalies if a.frame_num == frame_num]

    def get_anomalies_in_region(
        self,
        mb_x_start: int,
        mb_y_start: int,
        mb_x_end: int,
        mb_y_end: int,
        frame_num: Optional[int] = None
    ) -> List[Anomaly]:
        """Get anomalies within a macroblock region."""
        results = []
        for a in self.anomalies:
            if a.mb_x is None or a.mb_y is None:
                continue
            if frame_num is not None and a.frame_num != frame_num:
                continue
            if mb_x_start <= a.mb_x <= mb_x_end and mb_y_start <= a.mb_y <= mb_y_end:
                results.append(a)
        return results

    def get_anomalies_by_type(self, anomaly_type: AnomalyType) -> List[Anomaly]:
        """Get all anomalies of a specific type."""
        return [a for a in self.anomalies if a.type == anomaly_type]

    def get_severity_counts(self) -> Dict[str, int]:
        """Count anomalies by severity level."""
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for a in self.anomalies:
            counts[a.severity] = counts.get(a.severity, 0) + 1
        return counts

    def detect_splice_candidates(self) -> List[Dict[str, Any]]:
        """
        Analyse anomaly patterns to identify potential splice points.

        A splice typically manifests as:
        - Sudden QP discontinuity
        - Motion vector field disruption
        - Compression characteristic change
        - Cluster of anomalies at frame boundary

        Returns list of candidate splice points with confidence scores.
        """
        candidates = []

        # Method 1: QP discontinuities
        for i in range(1, len(self._qp_history)):
            if self._qp_history[i-1] > 0 and self._qp_history[i] > 0:
                delta = abs(self._qp_history[i] - self._qp_history[i-1])
                if delta > 10:
                    candidates.append({
                        "frame": i,
                        "type": "qp_discontinuity",
                        "confidence": min(delta / 20.0, 1.0),
                        "evidence": f"QP delta of {delta:.1f}",
                    })

        # Method 2: Anomaly density spikes
        if len(self._anomaly_density) > 2:
            mean_density = np.mean(self._anomaly_density)
            std_density = np.std(self._anomaly_density) + 0.1  # Avoid div by zero
            for i, density in enumerate(self._anomaly_density):
                z_score = (density - mean_density) / std_density
                if z_score > 2.5:
                    candidates.append({
                        "frame": i,
                        "type": "anomaly_cluster",
                        "confidence": min(z_score / 5.0, 1.0),
                        "evidence": f"{density} anomalies (z={z_score:.1f})",
                    })

        # Method 3: IDR frames at unexpected positions (non-GOP-aligned)
        idr_anomalies = self.get_anomalies_by_type(AnomalyType.SLICE_UNEXPECTED_IDR)
        for a in idr_anomalies:
            candidates.append({
                "frame": a.frame_num,
                "type": "unexpected_idr",
                "confidence": 0.7,
                "evidence": a.description,
            })

        # Sort by confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive forensic report."""
        severity_counts = self.get_severity_counts()
        type_counts: Dict[str, int] = {}
        for a in self.anomalies:
            type_counts[a.type.value] = type_counts.get(a.type.value, 0) + 1

        return {
            "total_frames": self.frame_count,
            "total_anomalies": len(self.anomalies),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "splice_candidates": self.detect_splice_candidates(),
            "qp_history": self._qp_history,
            "anomaly_density": self._anomaly_density,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


class QPAnalyzer:
    """
    Analyses quantisation parameter distribution for forensic indicators.

    QP (Quantisation Parameter) controls compression quality per macroblock.
    In authentic video, QP typically varies smoothly based on scene complexity.
    Anomalies include:

    - Sharp QP boundaries not aligned with content (splice indicator)
    - Different QP distributions in different regions (compositing)
    - Unusual QP patterns (re-encoding from different source)

    Reference: ITU-T H.264 Section 7.4.5 (Macroblock layer semantics)
    """

    def __init__(self, width_mbs: int, height_mbs: int):
        """
        Initialise QP analyser for a video frame.

        Args:
            width_mbs: Frame width in macroblocks (pixels // 16)
            height_mbs: Frame height in macroblocks (pixels // 16)
        """
        self.width_mbs = width_mbs
        self.height_mbs = height_mbs
        self.qp_map: np.ndarray = np.zeros((height_mbs, width_mbs), dtype=np.int8)
        self.valid_mask: np.ndarray = np.zeros((height_mbs, width_mbs), dtype=bool)

    def set_mb_qp(self, mb_x: int, mb_y: int, qp: int):
        """Record QP for a macroblock (valid range: 0-51)."""
        if 0 <= mb_x < self.width_mbs and 0 <= mb_y < self.height_mbs:
            self.qp_map[mb_y, mb_x] = np.clip(qp, 0, 51)
            self.valid_mask[mb_y, mb_x] = True

    def get_average_qp(self) -> float:
        """Calculate average QP across all valid macroblocks."""
        if not np.any(self.valid_mask):
            return 0.0
        return float(np.mean(self.qp_map[self.valid_mask]))

    def get_qp_histogram(self) -> np.ndarray:
        """Get histogram of QP values (52 bins, 0-51)."""
        valid_qps = self.qp_map[self.valid_mask]
        return np.histogram(valid_qps, bins=52, range=(0, 52))[0]

    def detect_qp_boundaries(self, threshold: int = 8) -> List[Dict[str, Any]]:
        """
        Detect sharp QP boundaries that may indicate splicing.

        Args:
            threshold: Minimum QP difference to flag as boundary

        Returns:
            List of boundary regions with coordinates and QP delta
        """
        boundaries = []

        # Horizontal boundaries
        for y in range(self.height_mbs):
            for x in range(1, self.width_mbs):
                if self.valid_mask[y, x] and self.valid_mask[y, x-1]:
                    delta = abs(int(self.qp_map[y, x]) - int(self.qp_map[y, x-1]))
                    if delta >= threshold:
                        boundaries.append({
                            "type": "horizontal",
                            "mb_x": x,
                            "mb_y": y,
                            "qp_left": int(self.qp_map[y, x-1]),
                            "qp_right": int(self.qp_map[y, x]),
                            "delta": delta,
                        })

        # Vertical boundaries
        for y in range(1, self.height_mbs):
            for x in range(self.width_mbs):
                if self.valid_mask[y, x] and self.valid_mask[y-1, x]:
                    delta = abs(int(self.qp_map[y, x]) - int(self.qp_map[y-1, x]))
                    if delta >= threshold:
                        boundaries.append({
                            "type": "vertical",
                            "mb_x": x,
                            "mb_y": y,
                            "qp_above": int(self.qp_map[y-1, x]),
                            "qp_below": int(self.qp_map[y, x]),
                            "delta": delta,
                        })

        return boundaries

    def detect_qp_regions(self, min_region_size: int = 16) -> List[Dict[str, Any]]:
        """
        Segment frame into regions of similar QP.

        Different regions with significantly different QP statistics
        may indicate content from different sources.

        Args:
            min_region_size: Minimum macroblocks to consider a region

        Returns:
            List of regions with statistics
        """
        from scipy import ndimage  # Local import to avoid hard dependency

        if not np.any(self.valid_mask):
            return []

        # Quantise QP into bands (e.g., 0-10, 11-20, 21-30, 31-40, 41-51)
        qp_bands = (self.qp_map // 10).astype(np.int8)
        qp_bands[~self.valid_mask] = -1  # Invalid marker

        # Label connected regions
        labeled, num_features = ndimage.label(qp_bands >= 0)

        regions = []
        for region_id in range(1, num_features + 1):
            mask = labeled == region_id
            if np.sum(mask) >= min_region_size:
                region_qps = self.qp_map[mask]
                regions.append({
                    "region_id": region_id,
                    "size_mbs": int(np.sum(mask)),
                    "mean_qp": float(np.mean(region_qps)),
                    "std_qp": float(np.std(region_qps)),
                    "min_qp": int(np.min(region_qps)),
                    "max_qp": int(np.max(region_qps)),
                })

        return regions

    def to_heatmap(self) -> np.ndarray:
        """
        Generate QP heatmap for visualisation.

        Returns:
            2D array of QP values (0-51), -1 for invalid macroblocks
        """
        result = self.qp_map.astype(np.int8).copy()
        result[~self.valid_mask] = -1
        return result


class MVFieldAnalyzer:
    """
    Analyses motion vector fields for forensic indicators.

    Motion vectors describe how macroblocks move between frames.
    In authentic video, MV fields exhibit spatial coherence (nearby
    blocks move similarly). Anomalies include:

    - Sharp MV boundaries not aligned with object boundaries (splice)
    - MVs pointing outside valid reference area (corruption/manipulation)
    - Implausible MV magnitudes for the scene content
    - Statistical outliers in MV distribution

    Reference: ITU-T H.264 Section 8.4 (Inter prediction process)
    """

    def __init__(self, width_mbs: int, height_mbs: int):
        """
        Initialise MV analyser for a video frame.

        Args:
            width_mbs: Frame width in macroblocks
            height_mbs: Frame height in macroblocks
        """
        self.width_mbs = width_mbs
        self.height_mbs = height_mbs

        # Store MVs as quarter-pixel values (as per H.264 spec)
        # Shape: (height, width, 2) for (mv_x, mv_y)
        self.mv_field: np.ndarray = np.zeros((height_mbs, width_mbs, 2), dtype=np.int16)
        self.valid_mask: np.ndarray = np.zeros((height_mbs, width_mbs), dtype=bool)
        self.ref_idx: np.ndarray = np.zeros((height_mbs, width_mbs), dtype=np.int8)

    def set_mb_mv(self, mb_x: int, mb_y: int, mv_x: int, mv_y: int, ref_idx: int = 0):
        """
        Record motion vector for a macroblock.

        Args:
            mb_x, mb_y: Macroblock position
            mv_x, mv_y: Motion vector in quarter-pixel units
            ref_idx: Reference frame index
        """
        if 0 <= mb_x < self.width_mbs and 0 <= mb_y < self.height_mbs:
            self.mv_field[mb_y, mb_x, 0] = mv_x
            self.mv_field[mb_y, mb_x, 1] = mv_y
            self.ref_idx[mb_y, mb_x] = ref_idx
            self.valid_mask[mb_y, mb_x] = True

    def get_magnitude_field(self) -> np.ndarray:
        """Calculate MV magnitude for each macroblock."""
        magnitudes = np.sqrt(
            self.mv_field[:, :, 0].astype(np.float32)**2 +
            self.mv_field[:, :, 1].astype(np.float32)**2
        )
        magnitudes[~self.valid_mask] = 0
        return magnitudes

    def get_average_magnitude(self) -> float:
        """Calculate average MV magnitude across valid macroblocks."""
        if not np.any(self.valid_mask):
            return 0.0
        magnitudes = self.get_magnitude_field()
        return float(np.mean(magnitudes[self.valid_mask]))

    def detect_spatial_discontinuities(self, threshold: float = 32.0) -> List[Dict[str, Any]]:
        """
        Detect sharp MV discontinuities that may indicate splicing.

        Authentic video typically has smooth MV transitions except at
        object boundaries. Sharp discontinuities in homogeneous regions
        are suspicious.

        Args:
            threshold: Minimum MV difference (quarter-pixels) to flag

        Returns:
            List of discontinuity locations with magnitude
        """
        discontinuities = []

        for y in range(self.height_mbs):
            for x in range(self.width_mbs):
                if not self.valid_mask[y, x]:
                    continue

                mv = self.mv_field[y, x]

                # Check neighbours
                neighbours = []
                if x > 0 and self.valid_mask[y, x-1]:
                    neighbours.append(("left", self.mv_field[y, x-1]))
                if x < self.width_mbs-1 and self.valid_mask[y, x+1]:
                    neighbours.append(("right", self.mv_field[y, x+1]))
                if y > 0 and self.valid_mask[y-1, x]:
                    neighbours.append(("above", self.mv_field[y-1, x]))
                if y < self.height_mbs-1 and self.valid_mask[y+1, x]:
                    neighbours.append(("below", self.mv_field[y+1, x]))

                for direction, neighbour_mv in neighbours:
                    diff = np.sqrt(
                        (float(mv[0]) - float(neighbour_mv[0]))**2 +
                        (float(mv[1]) - float(neighbour_mv[1]))**2
                    )
                    if diff >= threshold:
                        discontinuities.append({
                            "mb_x": x,
                            "mb_y": y,
                            "direction": direction,
                            "mv": (int(mv[0]), int(mv[1])),
                            "neighbour_mv": (int(neighbour_mv[0]), int(neighbour_mv[1])),
                            "difference": float(diff),
                        })

        return discontinuities

    def detect_outliers(self, z_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect statistical outliers in MV field.

        Args:
            z_threshold: Z-score threshold for outlier detection

        Returns:
            List of outlier macroblocks with MV and z-score
        """
        if not np.any(self.valid_mask):
            return []

        magnitudes = self.get_magnitude_field()
        valid_mags = magnitudes[self.valid_mask]

        mean_mag = np.mean(valid_mags)
        std_mag = np.std(valid_mags) + 0.1  # Avoid div by zero

        outliers = []
        for y in range(self.height_mbs):
            for x in range(self.width_mbs):
                if self.valid_mask[y, x]:
                    mag = magnitudes[y, x]
                    z_score = (mag - mean_mag) / std_mag
                    if abs(z_score) >= z_threshold:
                        outliers.append({
                            "mb_x": x,
                            "mb_y": y,
                            "mv": (int(self.mv_field[y, x, 0]), int(self.mv_field[y, x, 1])),
                            "magnitude": float(mag),
                            "z_score": float(z_score),
                        })

        return outliers

    def detect_reference_anomalies(
        self,
        frame_width: int,
        frame_height: int,
        ref_frame_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Detect MVs that reference invalid positions.

        Args:
            frame_width, frame_height: Reference frame dimensions in pixels
            ref_frame_count: Number of available reference frames

        Returns:
            List of macroblocks with invalid references
        """
        anomalies = []

        for y in range(self.height_mbs):
            for x in range(self.width_mbs):
                if not self.valid_mask[y, x]:
                    continue

                # Check reference index
                ref = self.ref_idx[y, x]
                if ref >= ref_frame_count or ref < 0:
                    anomalies.append({
                        "mb_x": x,
                        "mb_y": y,
                        "type": "invalid_ref_idx",
                        "ref_idx": int(ref),
                        "max_ref": ref_frame_count,
                    })
                    continue

                # Check if MV points outside frame (with reasonable margin)
                mv = self.mv_field[y, x]
                target_x = x * 16 * 4 + mv[0]  # In quarter-pixels
                target_y = y * 16 * 4 + mv[1]

                margin = 64 * 4  # 64 pixels margin in quarter-pel
                if (target_x < -margin or target_x >= (frame_width * 4 + margin) or
                    target_y < -margin or target_y >= (frame_height * 4 + margin)):
                    anomalies.append({
                        "mb_x": x,
                        "mb_y": y,
                        "type": "mv_out_of_bounds",
                        "mv": (int(mv[0]), int(mv[1])),
                        "target_qpel": (int(target_x), int(target_y)),
                    })

        return anomalies

    def to_flow_visualization(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate flow field for visualisation.

        Returns:
            Tuple of (magnitude, angle) arrays for HSV visualisation
        """
        mv_x = self.mv_field[:, :, 0].astype(np.float32)
        mv_y = self.mv_field[:, :, 1].astype(np.float32)

        magnitude = np.sqrt(mv_x**2 + mv_y**2)
        angle = np.arctan2(mv_y, mv_x)

        magnitude[~self.valid_mask] = 0
        angle[~self.valid_mask] = 0

        return magnitude, angle
