"""
JPEG Block Grid forensic analysis.

Detects misaligned JPEG compression blocks, which can indicate:
1. Image cropping (grid offset from origin)
2. Image splicing (different regions have different grid alignments)
3. Re-compression (double JPEG compression artifacts)

JPEG uses 8x8 pixel blocks for DCT compression. When an image is:
- Cropped: The grid shifts away from (0,0)
- Spliced: Pasted regions may have different grid phases
- Re-saved: Shows "ghost" artifacts from previous compression

Court relevance:
    "The compression blocks in this region don't align with the rest
    of the image, indicating the region was added from a different source."

PERFORMANCE NOTES:
    This module contains algorithms optimized for forensic accuracy.
    Areas marked with:

    # OPTIMIZE: CYTHON - Would benefit from Cython compilation
    # OPTIMIZE: C - Should be rewritten in C for production
    # OPTIMIZE: ASM - Critical inner loop, consider SIMD/Assembly

References:
    Farid, H. (2009). Exposing Digital Forgeries from JPEG Ghosts.
        IEEE Transactions on Information Forensics and Security.
    Lin, Z., He, J., Tang, X., & Tang, C.K. (2009). Fast, Automatic and
        Fine-grained Tampered JPEG Image Detection via DCT Coefficient Analysis.
        Pattern Recognition.
    Bianchi, T. & Piva, A. (2012). Image Forgery Localization via Block-Grained
        Analysis of JPEG Artifacts. IEEE Transactions on Information
        Forensics and Security.
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
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Native SIMD library for accelerated computation
try:
    from ..native import simd as native_simd
    HAS_NATIVE_SIMD = native_simd.is_available()
except ImportError:
    HAS_NATIVE_SIMD = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


@dataclass
class BlockGridOffset:
    """Detected JPEG block grid offset."""
    x_offset: int  # 0-7, horizontal offset from origin
    y_offset: int  # 0-7, vertical offset from origin
    confidence: float  # 0-1, confidence in this measurement
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) if regional


@dataclass
class GridRegionAnalysis:
    """Analysis of a region's grid alignment."""
    x: int
    y: int
    width: int
    height: int
    primary_offset: BlockGridOffset
    inconsistent: bool  # True if offset differs from image primary


@dataclass
class BlockGridResult:
    """Result of block grid analysis."""
    primary_offset: BlockGridOffset  # Dominant grid offset for whole image
    is_cropped: bool  # True if grid suggests cropping
    has_spliced_regions: bool  # True if regions have different grids
    spliced_regions: List[GridRegionAnalysis] = field(default_factory=list)
    double_compression_detected: bool = False
    original_quality: Optional[int] = None  # Estimated original JPEG quality
    processing_time_ms: float = 0.0


class BlockGridAnalyzer:
    """
    Analyzes JPEG block grid alignment for forgery detection.

    JPEG compression works on 8x8 pixel blocks aligned to a grid starting
    at (0,0). Analysis can reveal:

    1. Cropping detection: If the dominant grid doesn't start at (0,0),
       the image was likely cropped.

    2. Splicing detection: If different regions have different grid phases,
       those regions likely came from different source images.

    3. Double compression: If an image was saved as JPEG, edited, then
       re-saved as JPEG, "ghost" artifacts appear at certain qualities.

    Limitations:
    - Only works on JPEG images (or images that were JPEG at some point)
    - High-quality JPEGs have weaker artifacts
    - Very low-quality JPEGs may have artifacts that mask analysis
    - PNG/TIFF images that were never JPEG will show no grid
    """

    BLOCK_SIZE = 8  # JPEG DCT block size
    ANALYSIS_REGION_SIZE = 64  # Size of regions for local analysis
    INCONSISTENCY_THRESHOLD = 0.3  # Threshold for detecting different grids

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze image for JPEG block grid anomalies.

        Returns DimensionResult indicating grid consistency.
        """
        if not HAS_NUMPY or not HAS_PIL:
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="numpy and PIL required for block grid analysis"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        # Load image
        try:
            with Image.open(file_path) as img:
                # Check if it's a JPEG
                is_jpeg = img.format == 'JPEG'

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)
        except Exception as e:
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot load image",
                    explanation=str(e)
                )]
            )

        evidence = []

        # Detect JPEG artifacts (even in non-JPEG files that were once JPEG)
        if not HAS_SCIPY:
            evidence.append(Evidence(
                finding="scipy not available",
                explanation="Full block grid analysis requires scipy"
            ))
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.LOW,
                evidence=evidence
            )

        # Perform block grid analysis
        result = self._analyze_block_grid(image_array)

        # Report findings
        if result.primary_offset.confidence < 0.3:
            evidence.append(Evidence(
                finding="Weak JPEG artifacts",
                explanation=(
                    "Image has weak block compression artifacts. "
                    "May be high-quality JPEG, PNG, or never-compressed image."
                )
            ))
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.LOW,
                evidence=evidence,
                methodology="JPEG block grid analysis"
            )

        # Report primary grid offset
        x_off = result.primary_offset.x_offset
        y_off = result.primary_offset.y_offset

        if x_off == 0 and y_off == 0:
            evidence.append(Evidence(
                finding="Block grid aligned at origin",
                explanation="JPEG compression grid starts at (0,0) as expected for unedited images"
            ))
        else:
            evidence.append(Evidence(
                finding=f"Block grid offset: ({x_off}, {y_off})",
                explanation=(
                    f"JPEG compression grid is offset by ({x_off}, {y_off}) pixels from origin. "
                    f"This typically indicates the image was cropped."
                ),
                citation="Farid (2009) - JPEG grid offset detection"
            ))

        # Check for spliced regions
        if result.has_spliced_regions:
            n_regions = len(result.spliced_regions)
            evidence.append(Evidence(
                finding=f"Grid inconsistency: {n_regions} region(s) with different alignment",
                explanation=(
                    f"Found {n_regions} region(s) where the JPEG block grid differs from "
                    f"the rest of the image. This strongly suggests image splicing."
                ),
                citation="Bianchi & Piva (2012) - Block-grained JPEG artifact analysis"
            ))

            # Add details about inconsistent regions
            for i, region in enumerate(result.spliced_regions[:3]):
                evidence.append(Evidence(
                    finding=(
                        f"Suspicious region {i+1}: ({region.x}, {region.y}) "
                        f"{region.width}x{region.height}px"
                    ),
                    explanation=(
                        f"Grid offset ({region.primary_offset.x_offset}, "
                        f"{region.primary_offset.y_offset}) differs from image primary "
                        f"({x_off}, {y_off})"
                    )
                ))

            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=evidence,
                methodology="JPEG block grid analysis with regional comparison",
                raw_data={"spliced_regions": [
                    {"x": r.x, "y": r.y, "width": r.width, "height": r.height,
                     "offset_x": r.primary_offset.x_offset,
                     "offset_y": r.primary_offset.y_offset}
                    for r in result.spliced_regions
                ]}
            )

        # Check for double compression
        if result.double_compression_detected:
            evidence.append(Evidence(
                finding="Double JPEG compression detected",
                explanation=(
                    "Image shows artifacts consistent with being saved as JPEG, "
                    "edited, and re-saved as JPEG. "
                    f"Estimated original quality: {result.original_quality or 'unknown'}"
                ),
                citation="Lin et al. (2009) - DCT coefficient analysis"
            ))
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
                evidence=evidence,
                methodology="JPEG block grid and double compression analysis"
            )

        # Cropping detected but no splicing
        if result.is_cropped:
            evidence.append(Evidence(
                finding="Image appears to be cropped",
                explanation="Grid offset suggests cropping, but this alone is not evidence of manipulation"
            ))
            return DimensionResult(
                dimension="blockgrid",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.MEDIUM,
                evidence=evidence,
                methodology="JPEG block grid analysis"
            )

        # Everything looks consistent
        evidence.append(Evidence(
            finding="Block grid consistent",
            explanation="JPEG compression artifacts are consistent throughout the image"
        ))

        return DimensionResult(
            dimension="blockgrid",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="JPEG block grid analysis"
        )

    def _analyze_block_grid(self, image: np.ndarray) -> BlockGridResult:
        """
        Analyze JPEG block grid alignment.

        OPTIMIZE: C - This entire method would benefit from C implementation
        """
        import time
        start_time = time.time()

        # Convert to grayscale
        # OPTIMIZE: CYTHON - Color conversion
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        height, width = gray.shape

        # Detect primary grid offset for whole image
        primary_offset = self._detect_grid_offset(gray)

        # Check if image appears cropped
        is_cropped = (primary_offset.x_offset != 0 or primary_offset.y_offset != 0)

        # Analyze local regions for grid consistency
        # OPTIMIZE: C - Parallel region analysis
        spliced_regions = []

        region_size = self.ANALYSIS_REGION_SIZE
        step = region_size // 2  # 50% overlap

        for y in range(0, height - region_size, step):
            for x in range(0, width - region_size, step):
                region = gray[y:y+region_size, x:x+region_size]

                # Skip low-variance regions (not enough texture for grid detection)
                if np.var(region) < 100:
                    continue

                local_offset = self._detect_grid_offset(region)

                # Check if local grid differs from primary
                if local_offset.confidence > 0.3:
                    x_diff = abs(local_offset.x_offset - primary_offset.x_offset)
                    y_diff = abs(local_offset.y_offset - primary_offset.y_offset)

                    # Handle wraparound (7 and 0 are only 1 apart)
                    x_diff = min(x_diff, 8 - x_diff)
                    y_diff = min(y_diff, 8 - y_diff)

                    if x_diff > 1 or y_diff > 1:
                        spliced_regions.append(GridRegionAnalysis(
                            x=x, y=y,
                            width=region_size, height=region_size,
                            primary_offset=local_offset,
                            inconsistent=True
                        ))

        # Merge adjacent inconsistent regions
        merged_regions = self._merge_regions(spliced_regions)

        # Detect double compression
        double_compression, original_quality = self._detect_double_compression(gray)

        elapsed_ms = (time.time() - start_time) * 1000

        return BlockGridResult(
            primary_offset=primary_offset,
            is_cropped=is_cropped,
            has_spliced_regions=len(merged_regions) > 0,
            spliced_regions=merged_regions,
            double_compression_detected=double_compression,
            original_quality=original_quality,
            processing_time_ms=elapsed_ms
        )

    def _detect_grid_offset(self, image: np.ndarray) -> BlockGridOffset:
        """
        Detect the JPEG block grid offset for an image region.

        Uses the fact that JPEG compression creates discontinuities at
        block boundaries. By looking for these discontinuities, we can
        determine the grid phase.

        OPTIMIZE: ASM - This involves many small convolutions, ideal for SIMD
        """
        height, width = image.shape

        if height < 16 or width < 16:
            return BlockGridOffset(0, 0, 0.0)

        # Compute blockiness measure for each possible offset
        # OPTIMIZE: C - Nested loops with SIMD for blockiness computation
        best_offset = (0, 0)
        best_score = -float('inf')
        scores = np.zeros((8, 8))

        for x_off in range(8):
            for y_off in range(8):
                score = self._compute_blockiness(image, x_off, y_off)
                scores[y_off, x_off] = score
                if score > best_score:
                    best_score = score
                    best_offset = (x_off, y_off)

        # Compute confidence based on score distribution
        scores_flat = scores.flatten()
        scores_sorted = np.sort(scores_flat)[::-1]

        if scores_sorted[0] > 0 and scores_sorted[1] > 0:
            # Confidence is ratio of best to second-best
            confidence = min(1.0, (scores_sorted[0] - scores_sorted[1]) /
                           (scores_sorted[0] + 0.001))
        else:
            confidence = 0.0

        return BlockGridOffset(
            x_offset=best_offset[0],
            y_offset=best_offset[1],
            confidence=confidence
        )

    def _compute_blockiness(
        self,
        image: np.ndarray,
        x_offset: int,
        y_offset: int
    ) -> float:
        """
        Compute blockiness score for a given grid offset.

        Blockiness is measured by the average squared difference
        across block boundaries.

        Uses native SIMD when available for ~5x speedup.
        """
        height, width = image.shape
        block_size = self.BLOCK_SIZE

        # Use native SIMD implementation if available
        if HAS_NATIVE_SIMD:
            return native_simd.compute_blockiness(
                image, x_offset, y_offset, block_size
            )

        # Python fallback
        total_diff = 0.0
        count = 0

        # Check vertical block boundaries
        for x in range(x_offset, width - 1, block_size):
            if x < width - 1:
                col1 = image[:, x]
                col2 = image[:, x + 1]
                diff = np.mean((col1 - col2) ** 2)
                total_diff += diff
                count += 1

        # Check horizontal block boundaries
        for y in range(y_offset, height - 1, block_size):
            if y < height - 1:
                row1 = image[y, :]
                row2 = image[y + 1, :]
                diff = np.mean((row1 - row2) ** 2)
                total_diff += diff
                count += 1

        if count == 0:
            return 0.0

        return total_diff / count

    def _detect_double_compression(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[int]]:
        """
        Detect double JPEG compression.

        When an image is saved as JPEG, edited, then re-saved, it creates
        distinctive artifacts in the DCT coefficient histogram.

        Optimized with batch DCT computation.
        """
        height, width = image.shape

        if height < 64 or width < 64:
            return False, None

        block_size = 8

        # Batch extract all 8x8 blocks using stride tricks
        n_rows = height // block_size
        n_cols = width // block_size

        # Limit to avoid memory issues
        max_blocks = 50000 // 63  # 63 non-DC coefficients per block
        if n_rows * n_cols > max_blocks:
            # Sample a subset
            n_rows = min(n_rows, int(np.sqrt(max_blocks)))
            n_cols = min(n_cols, int(np.sqrt(max_blocks)))

        # Extract blocks using stride tricks (vectorized)
        trimmed = image[:n_rows * block_size, :n_cols * block_size]
        shape = (n_rows, n_cols, block_size, block_size)
        strides = (trimmed.strides[0] * block_size, trimmed.strides[1] * block_size,
                   trimmed.strides[0], trimmed.strides[1])
        blocks = np.lib.stride_tricks.as_strided(trimmed, shape=shape, strides=strides)
        blocks = blocks.reshape(-1, block_size, block_size).copy()

        if len(blocks) < 16:
            return False, None

        # Batch DCT - vectorized over all blocks at once
        dct_blocks = fftpack.dct(fftpack.dct(blocks, axis=2, norm='ortho'), axis=1, norm='ortho')

        # Extract non-DC coefficients (all except [0,0]) - vectorized
        # Flatten each block and skip first element (DC)
        dct_flat = dct_blocks.reshape(len(blocks), -1)[:, 1:]  # Skip DC
        dct_coeffs = dct_flat.ravel()

        if len(dct_coeffs) < 1000:
            return False, None

        # Analyze histogram for double compression artifacts
        # Double compression creates periodic peaks in the histogram
        hist, bins = np.histogram(dct_coeffs, bins=100, range=(-50, 50))

        # Look for periodic peaks (sign of double compression)
        # OPTIMIZE: CYTHON - Peak detection
        fft_hist = np.abs(np.fft.fft(hist))

        # Skip DC component, look for peaks
        fft_mag = fft_hist[1:len(fft_hist)//2]

        if len(fft_mag) < 5:
            return False, None

        # If there's a strong periodic component, double compression likely
        peak_ratio = np.max(fft_mag) / (np.mean(fft_mag) + 0.001)

        if peak_ratio > 5.0:
            # Estimate original quality from peak position
            peak_pos = np.argmax(fft_mag) + 1
            # Rough quality estimate (inverse relationship)
            estimated_quality = max(10, min(95, 100 - peak_pos * 5))
            return True, estimated_quality

        return False, None

    def _merge_regions(
        self,
        regions: List[GridRegionAnalysis]
    ) -> List[GridRegionAnalysis]:
        """
        Merge adjacent inconsistent regions.

        OPTIMIZE: CYTHON - Region merging with spatial indexing
        """
        if not regions:
            return []

        # Sort by position
        regions = sorted(regions, key=lambda r: (r.y, r.x))

        # Simple clustering - merge regions with same offset that are close
        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            cluster = [r1]
            used.add(i)

            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                # Check if same offset and close together
                same_offset = (
                    r1.primary_offset.x_offset == r2.primary_offset.x_offset and
                    r1.primary_offset.y_offset == r2.primary_offset.y_offset
                )

                close = (
                    abs(r1.x - r2.x) < self.ANALYSIS_REGION_SIZE * 2 and
                    abs(r1.y - r2.y) < self.ANALYSIS_REGION_SIZE * 2
                )

                if same_offset and close:
                    cluster.append(r2)
                    used.add(j)

            # Only keep if cluster is significant (not just noise)
            if len(cluster) >= 2:
                # Merge into bounding box
                min_x = min(r.x for r in cluster)
                min_y = min(r.y for r in cluster)
                max_x = max(r.x + r.width for r in cluster)
                max_y = max(r.y + r.height for r in cluster)

                merged.append(GridRegionAnalysis(
                    x=min_x, y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    primary_offset=r1.primary_offset,
                    inconsistent=True
                ))

        return merged


# =============================================================================
# NATIVE IMPLEMENTATION STUBS
# =============================================================================

"""
# cython: language_level=3
# blockgrid_native.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt

# SIMD-accelerated blockiness computation
cpdef float compute_blockiness_fast(
    np.ndarray[np.float64_t, ndim=2] image,
    int x_offset,
    int y_offset,
    int block_size
):
    '''
    Fast blockiness computation.

    Should use:
    - SIMD for boundary difference computation
    - Cache-efficient memory access
    '''
    pass

# Batch DCT for double compression detection
cpdef np.ndarray[np.float64_t, ndim=1] extract_dct_coefficients_fast(
    np.ndarray[np.float64_t, ndim=2] image,
    int block_size,
    int max_samples
):
    '''
    Extract DCT coefficients from many blocks efficiently.

    Should use:
    - FFTW for DCT computation
    - Parallel block processing
    '''
    pass
"""

"""
// blockgrid_native.c - Pure C implementation

#include <immintrin.h>
#include <omp.h>

// AVX2-accelerated blockiness computation
float compute_blockiness_avx2(
    const double* image,
    int width, int height,
    int x_offset, int y_offset,
    int block_size
) {
    double total_diff = 0.0;
    int count = 0;

    // Process 4 pixels at a time with AVX2
    #pragma omp parallel for reduction(+:total_diff,count)
    for (int x = x_offset; x < width - 1; x += block_size) {
        __m256d sum = _mm256_setzero_pd();

        for (int y = 0; y < height - 3; y += 4) {
            __m256d col1 = _mm256_loadu_pd(&image[y * width + x]);
            __m256d col2 = _mm256_loadu_pd(&image[y * width + x + 1]);
            __m256d diff = _mm256_sub_pd(col1, col2);
            sum = _mm256_fmadd_pd(diff, diff, sum);
        }

        // Horizontal sum
        double result[4];
        _mm256_storeu_pd(result, sum);
        total_diff += result[0] + result[1] + result[2] + result[3];
        count += height;
    }

    return count > 0 ? total_diff / count : 0.0f;
}
"""
