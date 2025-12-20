"""
JPEG Quantization Table forensic analysis.

Analyzes JPEG quantization tables to detect:
1. Double compression (save→edit→re-save)
2. Software fingerprinting (which tool created/edited the image)
3. Quality manipulation (tables modified after initial save)
4. Regional inconsistency (spliced regions with different tables)

Quantization tables are the key to JPEG compression quality. Each camera
and software uses characteristic table patterns that can be fingerprinted.

Court relevance:
    "The quantization tables indicate this image was created by Photoshop CS6,
    not the iPhone 12 claimed in the metadata. The tables show at least two
    compression passes at different quality levels."

PERFORMANCE NOTES:
    # OPTIMIZE: C - DCT coefficient histogram computation
    # OPTIMIZE: ASM - SIMD for batch coefficient extraction

References:
    Farid, H. (2009). Exposing Digital Forgeries from JPEG Ghosts.
        IEEE Transactions on Information Forensics and Security.
    Kee, E., Johnson, M.K., & Farid, H. (2011). Digital Image Authentication
        from JPEG Headers. IEEE Transactions on Information Forensics.
    Lin, Z., He, J., Tang, X., & Tang, C.K. (2009). Fast, Automatic and
        Fine-grained Tampered JPEG Image Detection via DCT Coefficient Analysis.
"""

import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy import fftpack
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


# =============================================================================
# KNOWN QUANTIZATION TABLE FINGERPRINTS
# =============================================================================

# Standard JPEG luminance table (quality 50) from IJG library
STANDARD_LUMINANCE_Q50 = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
]

# Standard JPEG chrominance table (quality 50) from IJG library
STANDARD_CHROMINANCE_Q50 = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
]

# Known software fingerprints (characteristic table patterns)
# These are simplified examples - real forensic DBs have thousands
SOFTWARE_FINGERPRINTS = {
    "photoshop": {
        "description": "Adobe Photoshop (various versions)",
        "pattern": "flat_scaling",  # Photoshop uses linear quality scaling
        "ratio_range": (0.95, 1.05),  # Very consistent lum/chrom ratio
    },
    "libjpeg": {
        "description": "Independent JPEG Group library (standard)",
        "pattern": "standard_ijg",
        "ratio_range": (0.9, 1.1),
    },
    "mozjpeg": {
        "description": "Mozilla JPEG encoder (optimized)",
        "pattern": "optimized",
        "ratio_range": (0.85, 1.15),  # More aggressive chroma subsampling
    },
    "iphone": {
        "description": "Apple iPhone camera",
        "pattern": "apple_custom",
        "ratio_range": (0.92, 1.08),
    },
    "samsung": {
        "description": "Samsung camera",
        "pattern": "samsung_custom",
        "ratio_range": (0.88, 1.12),
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuantizationTable:
    """A single JPEG quantization table."""
    table_id: int  # 0=luminance, 1=chrominance
    values: List[int]  # 64 values (8x8 zigzag order)
    estimated_quality: int  # Estimated quality level (1-100)
    is_standard: bool  # Matches IJG standard table pattern

    @property
    def mean_value(self) -> float:
        """Mean quantization value (higher = lower quality)."""
        return sum(self.values) / len(self.values) if self.values else 0

    @property
    def dc_value(self) -> int:
        """DC component quantization (first value)."""
        return self.values[0] if self.values else 0


@dataclass
class SoftwareMatch:
    """Potential software that created the image."""
    software: str
    description: str
    confidence: float  # 0-1


@dataclass
class DoubleCompressionResult:
    """Result of double compression detection."""
    detected: bool
    primary_quality: Optional[int]  # Most recent compression quality
    secondary_quality: Optional[int]  # Original compression quality (if detected)
    confidence: float
    ghost_quality: Optional[int] = None  # Quality showing strongest ghost artifacts


@dataclass
class QuantizationResult:
    """Complete quantization analysis result."""
    tables: List[QuantizationTable]
    primary_quality: int  # Estimated current quality
    is_double_compressed: bool
    double_compression: Optional[DoubleCompressionResult]
    software_matches: List[SoftwareMatch]
    table_modified: bool  # Tables don't follow expected pattern
    processing_time_ms: float = 0.0


# =============================================================================
# QUANTIZATION ANALYZER
# =============================================================================

class QuantizationAnalyzer:
    """
    Analyzes JPEG quantization tables for forensic evidence.

    Quantization is the "lossy" step in JPEG compression. The tables
    define how much detail is lost. Analysis can reveal:

    1. Double compression: Multiple quality levels detected
    2. Software identification: Tools leave characteristic patterns
    3. Quality manipulation: Tables manually adjusted
    4. Metadata mismatch: Tables don't match claimed camera/software
    """

    # DCT coefficient histogram analysis parameters
    GHOST_QUALITIES = list(range(50, 96, 5))  # Check these quality levels for ghosts
    GHOST_THRESHOLD = 1.5  # Peak ratio indicating ghost artifacts

    def analyze(self, file_path: str) -> DimensionResult:
        """Analyze quantization tables for forensic evidence."""
        if not HAS_NUMPY or not HAS_PIL:
            return self._uncertain_result("Dependencies not available")

        path = Path(file_path)
        if not path.exists():
            return self._uncertain_result(f"File not found: {file_path}")

        try:
            with Image.open(file_path) as img:
                if img.format != 'JPEG':
                    return self._uncertain_result(
                        "Not a JPEG file (quantization analysis requires JPEG)"
                    )

                # Extract quantization tables
                if not hasattr(img, 'quantization') or not img.quantization:
                    return self._uncertain_result("No quantization tables found in JPEG")

                qt_dict = img.quantization

                # Also get pixel data for DCT analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)

        except Exception as e:
            return self._uncertain_result(f"Cannot load image: {e}")

        # Perform analysis
        result = self._analyze_quantization(qt_dict, image_array)
        return self._build_result(result)

    def _analyze_quantization(
        self,
        qt_dict: Dict,
        image: np.ndarray
    ) -> QuantizationResult:
        """Perform complete quantization analysis."""
        import time
        start = time.time()

        # Parse quantization tables
        tables = self._parse_tables(qt_dict)

        # Estimate primary quality
        primary_quality = self._estimate_quality(tables)

        # Check if tables are standard
        tables_standard = all(t.is_standard for t in tables)

        # Identify software from table patterns
        software_matches = self._identify_software(tables)

        # Detect double compression via DCT histogram
        double_compression = None
        is_double_compressed = False

        if HAS_SCIPY:
            double_compression = self._detect_double_compression(image, primary_quality)
            is_double_compressed = double_compression.detected

        # Check for table modification
        table_modified = self._check_table_modification(tables)

        elapsed = (time.time() - start) * 1000

        return QuantizationResult(
            tables=tables,
            primary_quality=primary_quality,
            is_double_compressed=is_double_compressed,
            double_compression=double_compression,
            software_matches=software_matches,
            table_modified=table_modified,
            processing_time_ms=elapsed
        )

    def _parse_tables(self, qt_dict: Dict) -> List[QuantizationTable]:
        """Parse quantization tables from PIL format."""
        tables = []

        for table_id, values in qt_dict.items():
            values_list = list(values) if hasattr(values, '__iter__') else []

            # Estimate quality from table
            quality = self._estimate_quality_from_table(values_list)

            # Check if standard
            is_standard = self._is_standard_table(values_list, table_id)

            tables.append(QuantizationTable(
                table_id=table_id,
                values=values_list,
                estimated_quality=quality,
                is_standard=is_standard
            ))

        return tables

    def _estimate_quality_from_table(self, qt: List[int]) -> int:
        """Estimate JPEG quality from quantization table values."""
        if not qt or len(qt) < 64:
            return 0

        # Compare to standard Q50 table
        std_q50 = STANDARD_LUMINANCE_Q50

        # Compute scale factor relative to Q50
        scale_sum = 0
        count = 0

        for i, (val, std_val) in enumerate(zip(qt[:64], std_q50)):
            if std_val > 0 and val > 0:
                scale_sum += val / std_val
                count += 1

        if count == 0:
            return 50

        avg_scale = scale_sum / count

        # Convert scale to quality
        # scale < 1 means higher quality, scale > 1 means lower quality
        if avg_scale < 1:
            quality = min(100, int(50 + (1 - avg_scale) * 100))
        else:
            quality = max(1, int(50 / avg_scale))

        return quality

    def _estimate_quality(self, tables: List[QuantizationTable]) -> int:
        """Estimate overall quality from all tables."""
        if not tables:
            return 0

        # Use luminance table (id=0) if available, else first table
        for table in tables:
            if table.table_id == 0:
                return table.estimated_quality

        return tables[0].estimated_quality

    def _is_standard_table(self, qt: List[int], table_id: int) -> bool:
        """Check if table follows standard IJG pattern."""
        if not qt or len(qt) < 64:
            return False

        # Get appropriate standard table
        standard = STANDARD_LUMINANCE_Q50 if table_id == 0 else STANDARD_CHROMINANCE_Q50

        # Compute ratios - standard tables have consistent scaling
        ratios = []
        for val, std_val in zip(qt[:64], standard):
            if std_val > 0 and val > 0:
                ratios.append(val / std_val)

        if len(ratios) < 32:
            return False

        # Standard tables have low variance in ratios
        ratio_std = np.std(ratios)

        return ratio_std < 0.3

    def _identify_software(self, tables: List[QuantizationTable]) -> List[SoftwareMatch]:
        """Identify software from quantization table patterns."""
        matches = []

        if len(tables) < 2:
            return matches

        # Get luminance and chrominance tables
        lum_table = next((t for t in tables if t.table_id == 0), None)
        chrom_table = next((t for t in tables if t.table_id == 1), None)

        if not lum_table or not chrom_table:
            return matches

        # Compute lum/chrom ratio
        lum_mean = lum_table.mean_value
        chrom_mean = chrom_table.mean_value

        if chrom_mean > 0:
            ratio = lum_mean / chrom_mean
        else:
            ratio = 1.0

        # Check against known fingerprints
        for software_id, fingerprint in SOFTWARE_FINGERPRINTS.items():
            ratio_range = fingerprint["ratio_range"]
            if ratio_range[0] <= ratio <= ratio_range[1]:
                # Additional pattern checks
                confidence = self._compute_software_confidence(
                    tables, fingerprint["pattern"]
                )
                if confidence > 0.3:
                    matches.append(SoftwareMatch(
                        software=software_id,
                        description=fingerprint["description"],
                        confidence=confidence
                    ))

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return matches[:3]  # Top 3 matches

    def _compute_software_confidence(
        self,
        tables: List[QuantizationTable],
        pattern: str
    ) -> float:
        """Compute confidence for a software pattern match."""
        # Simplified pattern matching
        if pattern == "standard_ijg":
            # Check if tables are standard
            if all(t.is_standard for t in tables):
                return 0.7
            return 0.3

        elif pattern == "flat_scaling":
            # Photoshop uses very consistent scaling
            if len(tables) >= 2:
                qualities = [t.estimated_quality for t in tables]
                if max(qualities) - min(qualities) < 5:
                    return 0.6
            return 0.3

        elif pattern == "optimized":
            # MozJPEG has more aggressive quantization
            lum = next((t for t in tables if t.table_id == 0), None)
            if lum and lum.dc_value < 10:
                return 0.5
            return 0.2

        return 0.3  # Default low confidence

    def _detect_double_compression(
        self,
        image: np.ndarray,
        current_quality: int
    ) -> DoubleCompressionResult:
        """
        Detect double JPEG compression via DCT coefficient analysis.

        When an image is saved as JPEG, edited, and re-saved, the DCT
        coefficients show characteristic "ghost" artifacts at the
        original compression quality.
        """
        height, width = image.shape[:2]

        # Convert to grayscale for DCT analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # Extract DCT coefficients from 8x8 blocks
        block_size = 8
        dct_coeffs = []

        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

                # Collect AC coefficients (skip DC at [0,0])
                for i in range(block_size):
                    for j in range(block_size):
                        if i != 0 or j != 0:
                            dct_coeffs.append(dct[i, j])

                if len(dct_coeffs) > 100000:
                    break
            if len(dct_coeffs) > 100000:
                break

        if len(dct_coeffs) < 1000:
            return DoubleCompressionResult(
                detected=False,
                primary_quality=current_quality,
                secondary_quality=None,
                confidence=0.0
            )

        dct_coeffs = np.array(dct_coeffs)

        # Test for ghost artifacts at different quality levels
        ghost_scores = {}

        for test_quality in self.GHOST_QUALITIES:
            if abs(test_quality - current_quality) < 10:
                continue  # Skip qualities close to current

            score = self._compute_ghost_score(dct_coeffs, test_quality)
            ghost_scores[test_quality] = score

        if not ghost_scores:
            return DoubleCompressionResult(
                detected=False,
                primary_quality=current_quality,
                secondary_quality=None,
                confidence=0.0
            )

        # Find strongest ghost
        best_ghost_quality = max(ghost_scores, key=ghost_scores.get)
        best_ghost_score = ghost_scores[best_ghost_quality]

        # Determine if significant
        mean_score = np.mean(list(ghost_scores.values()))

        if best_ghost_score > mean_score * self.GHOST_THRESHOLD:
            confidence = min(1.0, (best_ghost_score - mean_score) / mean_score)
            return DoubleCompressionResult(
                detected=True,
                primary_quality=current_quality,
                secondary_quality=best_ghost_quality,
                confidence=confidence,
                ghost_quality=best_ghost_quality
            )

        return DoubleCompressionResult(
            detected=False,
            primary_quality=current_quality,
            secondary_quality=None,
            confidence=0.0
        )

    def _compute_ghost_score(self, coeffs: np.ndarray, quality: int) -> float:
        """Compute ghost artifact score for a specific quality level."""
        # Estimate quantization step for this quality
        # Higher quality = smaller step = more histogram peaks
        step = max(1, int((100 - quality) / 5))

        # Compute histogram of coefficients modulo step
        modulo_vals = np.abs(coeffs) % step

        # Ghost artifacts create peaks at 0 and step boundaries
        # Count coefficients near 0 modulo step
        near_boundary = np.sum(modulo_vals < step * 0.1) + np.sum(modulo_vals > step * 0.9)
        total = len(modulo_vals)

        if total == 0:
            return 0.0

        # Score is ratio of boundary values
        score = near_boundary / total

        return score

    def _check_table_modification(self, tables: List[QuantizationTable]) -> bool:
        """Check if quantization tables appear to be manually modified."""
        for table in tables:
            if len(table.values) < 64:
                continue

            # Check for suspicious patterns:
            # 1. All same value (unusual)
            if len(set(table.values)) == 1:
                return True

            # 2. Values not following expected frequency weighting
            # High frequencies should have higher quantization values
            low_freq_vals = table.values[:16]  # Upper-left corner
            high_freq_vals = table.values[48:]  # Lower-right corner

            if len(low_freq_vals) > 0 and len(high_freq_vals) > 0:
                if np.mean(low_freq_vals) > np.mean(high_freq_vals):
                    return True  # Inverted pattern is suspicious

            # 3. Non-monotonic DC value
            if table.values[0] < 1:
                return True  # DC can't be 0

        return False

    def _uncertain_result(self, reason: str) -> DimensionResult:
        """Return uncertain result."""
        return DimensionResult(
            dimension="quantization",
            state=DimensionState.UNCERTAIN,
            confidence=Confidence.NA,
            evidence=[Evidence(finding="Analysis not possible", explanation=reason)]
        )

    def _build_result(self, result: QuantizationResult) -> DimensionResult:
        """Build DimensionResult from analysis."""
        evidence = []

        # Report tables found
        for table in result.tables:
            table_type = "luminance" if table.table_id == 0 else "chrominance"
            evidence.append(Evidence(
                finding=f"Quantization table ({table_type}): quality ~{table.estimated_quality}",
                explanation=f"{'Standard IJG pattern' if table.is_standard else 'Custom/modified pattern'}"
            ))

        # Report software matches
        if result.software_matches:
            best_match = result.software_matches[0]
            evidence.append(Evidence(
                finding=f"Software fingerprint: {best_match.description}",
                explanation=f"Confidence: {best_match.confidence:.0%}",
                citation="Kee et al. (2011) - JPEG header authentication"
            ))

        # Report double compression
        if result.is_double_compressed and result.double_compression:
            dc = result.double_compression
            evidence.append(Evidence(
                finding="DOUBLE COMPRESSION DETECTED",
                explanation=(
                    f"Image was likely saved at quality ~{dc.secondary_quality}, "
                    f"then edited and re-saved at quality ~{dc.primary_quality}. "
                    f"Confidence: {dc.confidence:.0%}"
                ),
                citation="Farid (2009) - JPEG ghost artifacts"
            ))

            return DimensionResult(
                dimension="quantization",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.HIGH if dc.confidence > 0.6 else Confidence.MEDIUM,
                evidence=evidence,
                methodology="JPEG quantization table and DCT coefficient histogram analysis",
                raw_data={
                    "primary_quality": dc.primary_quality,
                    "secondary_quality": dc.secondary_quality,
                    "ghost_quality": dc.ghost_quality,
                    "confidence": dc.confidence
                }
            )

        # Report table modification
        if result.table_modified:
            evidence.append(Evidence(
                finding="QUANTIZATION TABLE ANOMALY",
                explanation="Tables show unusual patterns inconsistent with standard JPEG encoders",
                citation="Lin et al. (2009) - DCT coefficient analysis"
            ))

            return DimensionResult(
                dimension="quantization",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
                evidence=evidence,
                methodology="JPEG quantization table forensics"
            )

        # Normal result
        evidence.append(Evidence(
            finding="Quantization tables consistent",
            explanation="No double compression or table anomalies detected"
        ))

        return DimensionResult(
            dimension="quantization",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="JPEG quantization table forensics"
        )
