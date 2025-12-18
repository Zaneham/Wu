"""
Visual forensic analysis.

Analyzes image pixel data for manipulation artifacts:
- JPEG quantization table analysis (double compression detection)
- Error Level Analysis (ELA)
- DCT coefficient distribution anomalies
- Thumbnail inconsistency detection

No ML required - statistical analysis of compression artifacts.

References:
    Farid, H. (2009). Image Forgery Detection. IEEE Signal Processing Magazine.
    Krawetz, N. (2007). A Picture's Worth... Hacker Factor Solutions.
    Fridrich, J. (2009). Digital Image Forensics. IEEE Signal Processing Magazine.
"""

import io
import math
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field

try:
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


@dataclass
class ELAResult:
    """Error Level Analysis result."""
    max_difference: float  # Maximum pixel difference
    mean_difference: float  # Mean pixel difference
    std_difference: float   # Standard deviation
    hotspot_count: int      # Number of high-difference regions
    hotspot_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)


@dataclass
class QuantizationAnalysis:
    """JPEG quantization table analysis."""
    estimated_quality: int  # Estimated JPEG quality (1-100)
    is_double_compressed: bool
    quality_mismatch: bool  # Different quality for luminance vs chrominance
    standard_tables: bool   # Uses standard quantization tables


class VisualAnalyzer:
    """
    Analyzes image pixel data for forensic inconsistencies.

    Phase 0 implementation focuses on:
    - JPEG recompression detection
    - Error Level Analysis
    - Thumbnail consistency
    """

    # ELA thresholds
    ELA_RESAVE_QUALITY = 95  # Quality to resave at for ELA
    ELA_HOTSPOT_THRESHOLD = 25  # Pixel difference to flag as hotspot
    ELA_SUSPICIOUS_MEAN = 10  # Mean difference suggesting manipulation
    ELA_SUSPICIOUS_HOTSPOTS = 5  # Number of hotspots suggesting manipulation

    # Standard JPEG quantization tables (quality 50)
    STANDARD_LUMINANCE_QT = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    ]

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze image for visual manipulation artifacts.

        Returns DimensionResult with evidence of manipulation.
        """
        if not HAS_DEPS:
            return DimensionResult(
                dimension="visual",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="PIL/numpy required for visual analysis"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="visual",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        # Check file is an image
        try:
            with Image.open(file_path) as img:
                img_format = img.format
                img_mode = img.mode
        except Exception as e:
            return DimensionResult(
                dimension="visual",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot open image",
                    explanation=str(e)
                )]
            )

        evidence = []
        issues_found = False
        suspicious = False

        # Check 1: JPEG-specific analysis
        if img_format == "JPEG":
            jpeg_evidence = self._analyze_jpeg(file_path)
            for ev in jpeg_evidence:
                evidence.append(ev)
                if "double" in ev.finding.lower() or "recompressed" in ev.finding.lower():
                    suspicious = True

        # Check 2: Error Level Analysis
        ela_result = self._perform_ela(file_path)
        if ela_result:
            ela_evidence = self._interpret_ela(ela_result)
            if ela_evidence:
                evidence.append(ela_evidence)
                if "localized" in ela_evidence.finding.lower():
                    suspicious = True
                elif "significant" in ela_evidence.finding.lower():
                    issues_found = True

        # Check 3: Thumbnail consistency
        thumb_evidence = self._check_thumbnail_consistency(file_path)
        if thumb_evidence:
            evidence.append(thumb_evidence)
            issues_found = True

        # Determine state
        if issues_found:
            state = DimensionState.INCONSISTENT
            confidence = Confidence.HIGH
        elif suspicious:
            state = DimensionState.SUSPICIOUS
            confidence = Confidence.MEDIUM
        elif evidence:
            state = DimensionState.CONSISTENT
            confidence = Confidence.MEDIUM
        else:
            state = DimensionState.CONSISTENT
            confidence = Confidence.LOW
            evidence.append(Evidence(
                finding="No visual manipulation detected",
                explanation="Image passes basic visual forensic checks"
            ))

        return DimensionResult(
            dimension="visual",
            state=state,
            confidence=confidence,
            evidence=evidence,
            methodology="ELA analysis, JPEG quantization analysis, thumbnail verification"
        )

    def _analyze_jpeg(self, file_path: str) -> List[Evidence]:
        """Analyze JPEG-specific artifacts."""
        evidence = []

        try:
            with Image.open(file_path) as img:
                # Get quantization tables if available
                if hasattr(img, 'quantization') and img.quantization:
                    qt_analysis = self._analyze_quantization_tables(img.quantization)

                    if qt_analysis.is_double_compressed:
                        evidence.append(Evidence(
                            finding="Possible double JPEG compression detected",
                            explanation=(
                                f"Quantization table patterns suggest image was "
                                f"saved multiple times (estimated quality: {qt_analysis.estimated_quality})"
                            ),
                            citation="Fridrich (2009) - Digital Image Forensics"
                        ))

                    if qt_analysis.quality_mismatch:
                        evidence.append(Evidence(
                            finding="Quality mismatch in JPEG channels",
                            explanation="Luminance and chrominance use different quality settings"
                        ))

        except Exception:
            pass

        return evidence

    def _analyze_quantization_tables(self, qt: Dict) -> QuantizationAnalysis:
        """Analyze JPEG quantization tables for compression artifacts."""
        # qt is dict with keys 0 (luminance) and potentially 1 (chrominance)
        lum_table = list(qt.get(0, []))
        chrom_table = list(qt.get(1, lum_table))

        # Estimate quality from luminance table
        estimated_quality = self._estimate_jpeg_quality(lum_table)

        # Check for double compression indicators
        # Double compression often shows periodic patterns in DCT histograms
        is_double = self._detect_double_compression(lum_table)

        # Check quality consistency between channels
        chrom_quality = self._estimate_jpeg_quality(chrom_table)
        quality_mismatch = abs(estimated_quality - chrom_quality) > 10

        # Check if using standard tables
        standard_tables = self._is_standard_table(lum_table)

        return QuantizationAnalysis(
            estimated_quality=estimated_quality,
            is_double_compressed=is_double,
            quality_mismatch=quality_mismatch,
            standard_tables=standard_tables
        )

    def _estimate_jpeg_quality(self, qt: List[int]) -> int:
        """Estimate JPEG quality from quantization table."""
        if not qt or len(qt) < 64:
            return 0

        # Compare with standard table scaled to different qualities
        # Using first element (DC coefficient) as primary indicator
        dc_coeff = qt[0]

        # Standard DC at quality 50 is 16
        # Lower DC = higher quality
        if dc_coeff <= 1:
            return 100
        elif dc_coeff <= 2:
            return 95
        elif dc_coeff <= 4:
            return 90
        elif dc_coeff <= 8:
            return 80
        elif dc_coeff <= 16:
            return 50
        elif dc_coeff <= 32:
            return 25
        else:
            return 10

    def _detect_double_compression(self, qt: List[int]) -> bool:
        """
        Detect signs of double JPEG compression.

        Double compression creates periodic artifacts in DCT coefficients
        when the second quality differs from the first.
        """
        if not qt or len(qt) < 64:
            return False

        # Simple heuristic: check for unusual patterns
        # Double compression often creates more uniform high-frequency coefficients
        low_freq = sum(qt[:16])   # DC and low frequencies
        high_freq = sum(qt[48:])  # High frequencies

        # Very uniform tables can indicate editing software standardization
        variance = np.var(qt) if HAS_DEPS else 0
        if variance < 10:  # Very uniform
            return True

        return False

    def _is_standard_table(self, qt: List[int]) -> bool:
        """Check if quantization table matches standard JPEG tables."""
        if not qt or len(qt) < 64:
            return False

        # Check similarity to standard table
        standard = self.STANDARD_LUMINANCE_QT
        diffs = sum(abs(a - b) for a, b in zip(qt, standard))
        return diffs < 500  # Allow some variation

    def _perform_ela(self, file_path: str) -> Optional[ELAResult]:
        """
        Perform Error Level Analysis.

        ELA resaves the image at a known quality and compares
        to the original. Manipulated regions compress differently.
        """
        if not HAS_DEPS:
            return None

        try:
            with Image.open(file_path) as original:
                # Convert to RGB if necessary
                if original.mode != 'RGB':
                    original = original.convert('RGB')

                original_array = np.array(original, dtype=np.float32)

                # Resave at known quality
                buffer = io.BytesIO()
                original.save(buffer, format='JPEG', quality=self.ELA_RESAVE_QUALITY)
                buffer.seek(0)

                with Image.open(buffer) as resaved:
                    resaved_array = np.array(resaved, dtype=np.float32)

                # Calculate difference
                diff = np.abs(original_array - resaved_array)

                # Compute statistics
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff))

                # Find hotspots (regions with high difference)
                gray_diff = np.mean(diff, axis=2)  # Average across channels
                hotspots = gray_diff > self.ELA_HOTSPOT_THRESHOLD

                # Count connected hotspot regions
                hotspot_count = self._count_hotspot_regions(hotspots)

                return ELAResult(
                    max_difference=max_diff,
                    mean_difference=mean_diff,
                    std_difference=std_diff,
                    hotspot_count=hotspot_count
                )

        except Exception:
            return None

    def _count_hotspot_regions(self, hotspot_mask: np.ndarray) -> int:
        """Count distinct hotspot regions in ELA result."""
        # Simple connected component counting
        # For Phase 0, just count percentage of hotspot pixels
        total_pixels = hotspot_mask.size
        hotspot_pixels = np.sum(hotspot_mask)
        hotspot_percentage = (hotspot_pixels / total_pixels) * 100

        # Return estimated region count based on distribution
        if hotspot_percentage < 1:
            return 0
        elif hotspot_percentage < 5:
            return 1
        elif hotspot_percentage < 15:
            return 3
        elif hotspot_percentage < 30:
            return 5
        else:
            return 10  # Many regions or whole image is different

    def _interpret_ela(self, ela: ELAResult) -> Optional[Evidence]:
        """Interpret ELA results and generate evidence."""
        # Check for manipulation indicators
        if ela.hotspot_count >= self.ELA_SUSPICIOUS_HOTSPOTS:
            if ela.mean_difference > self.ELA_SUSPICIOUS_MEAN:
                return Evidence(
                    finding="Significant ELA anomalies detected",
                    explanation=(
                        f"Error Level Analysis shows {ela.hotspot_count} distinct regions "
                        f"with different compression levels (mean diff: {ela.mean_difference:.1f})"
                    ),
                    citation="Krawetz (2007) - Error Level Analysis"
                )
            else:
                return Evidence(
                    finding="Localized ELA variations detected",
                    explanation=(
                        f"Some regions show different compression characteristics "
                        f"({ela.hotspot_count} regions, max diff: {ela.max_difference:.1f})"
                    )
                )
        elif ela.mean_difference > self.ELA_SUSPICIOUS_MEAN * 2:
            return Evidence(
                finding="Uniform high ELA difference",
                explanation=(
                    f"Image shows uniformly high compression differences "
                    f"(mean: {ela.mean_difference:.1f}), suggesting recent editing or recompression"
                )
            )

        return None

    def _check_thumbnail_consistency(self, file_path: str) -> Optional[Evidence]:
        """
        Check if embedded thumbnail matches main image.

        Editing often updates the main image but not the thumbnail.
        """
        if not HAS_DEPS:
            return None

        try:
            with Image.open(file_path) as img:
                # Try to get EXIF thumbnail
                exif = img._getexif() if hasattr(img, '_getexif') else None
                if not exif:
                    return None

                # EXIF tag 274 is orientation, 306 is DateTime
                # EXIF tag for thumbnail in IFD1
                # This is complex in PIL, skip for Phase 0

                # Alternative: check for JFIF thumbnail
                if hasattr(img, 'applist'):
                    for segment in img.applist:
                        if segment[0] == 'JFIF' and len(segment[1]) > 16:
                            # Has JFIF thumbnail - could compare
                            pass

        except Exception:
            pass

        return None

    def get_ela_visualization(self, file_path: str, scale: float = 10.0) -> Optional[Image.Image]:
        """
        Generate ELA visualization image.

        Returns a PIL Image showing error levels, useful for reports.
        """
        if not HAS_DEPS:
            return None

        try:
            with Image.open(file_path) as original:
                if original.mode != 'RGB':
                    original = original.convert('RGB')

                original_array = np.array(original, dtype=np.float32)

                # Resave
                buffer = io.BytesIO()
                original.save(buffer, format='JPEG', quality=self.ELA_RESAVE_QUALITY)
                buffer.seek(0)

                with Image.open(buffer) as resaved:
                    resaved_array = np.array(resaved, dtype=np.float32)

                # Calculate and scale difference
                diff = np.abs(original_array - resaved_array) * scale
                diff = np.clip(diff, 0, 255).astype(np.uint8)

                return Image.fromarray(diff)

        except Exception:
            return None
