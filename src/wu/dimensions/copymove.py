"""
Copy-Move (Clone) forensic detection.

Detects regions of an image that have been copied and pasted within
the same image - a common manipulation technique for removing objects
or duplicating elements.

Approaches implemented:
1. Block-based DCT matching (classic, accurate)
2. Keypoint-based matching (faster for large images)

PERFORMANCE NOTES:
    This module contains several algorithms that would benefit from
    low-level optimization for production use. Areas marked with:

    # OPTIMIZE: CYTHON - Would benefit from Cython compilation
    # OPTIMIZE: C - Should be rewritten in C for production
    # OPTIMIZE: ASM - Critical inner loop, consider SIMD/Assembly

    Current Python implementation is suitable for:
    - Proof of concept
    - Small to medium images (<4MP)
    - Non-real-time analysis

    For production forensic workloads, the marked sections should be
    reimplemented in C/Cython with SIMD intrinsics (AVX2/NEON).

References:
    Fridrich, J., Soukal, D., & Lukáš, J. (2003). Detection of Copy-Move
        Forgery in Digital Images. Proceedings of Digital Forensic
        Research Workshop.
    Popescu, A.C. & Farid, H. (2004). Exposing Digital Forgeries by
        Detecting Duplicated Image Regions. Technical Report, Dartmouth.
    Ryu, S.J., Lee, M.J., & Lee, H.K. (2010). Detection of Copy-Rotate-Move
        Forgery Using Zernike Moments. Information Hiding.
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

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Native SIMD library for accelerated computation
try:
    from ..native import simd as native_simd
    HAS_NATIVE_SIMD = native_simd.is_available()
except ImportError:
    HAS_NATIVE_SIMD = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


@dataclass
class CloneRegion:
    """A detected cloned region pair."""
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    width: int
    height: int
    similarity: float  # 0-1, how similar the regions are

    @property
    def distance(self) -> float:
        """Euclidean distance between source and target."""
        return math.sqrt((self.target_x - self.source_x)**2 +
                        (self.target_y - self.source_y)**2)


@dataclass
class CopyMoveResult:
    """Result of copy-move detection."""
    detected: bool
    clone_regions: List[CloneRegion] = field(default_factory=list)
    method_used: str = ""
    processing_time_ms: float = 0.0


class CopyMoveAnalyzer:
    """
    Detects copy-move (clone) forgery in images.

    Copy-move forgery is when a region of an image is copied and pasted
    elsewhere in the same image, often to hide objects or duplicate elements.

    Detection methods:
    1. Block DCT matching - divides image into blocks, computes DCT,
       finds similar blocks that are spatially separated
    2. Keypoint matching - uses feature descriptors (ORB/SIFT) to find
       matching regions

    Limitations:
    - May miss heavily post-processed clones (blur, noise, rotation)
    - False positives on naturally repeating patterns (tiles, fabric)
    - Computational cost scales with image size
    """

    # Block-based parameters (empirically tuned on synthetic benchmark)
    BLOCK_SIZE = 16          # Size of blocks for DCT analysis
    DCT_COEFFICIENTS = 16    # Number of DCT coefficients to use per block
    SIMILARITY_THRESHOLD = 0.9995  # Minimum DCT similarity (high = fewer false positives)
    MIN_CLONE_DISTANCE = 48  # Minimum distance between clone pairs (pixels)
    MIN_CLONE_AREA = 512     # Minimum area to report (pixels²)
    MIN_BLOCK_VARIANCE = 100.0  # Minimum variance to consider block (skip uniform areas)
    PIXEL_VERIFY_THRESHOLD = 0.98  # Secondary pixel-level verification threshold
    MIN_CLONE_REGIONS = 3    # Minimum matching regions to report as forgery (reduces FPR)

    # Keypoint parameters
    MIN_KEYPOINT_MATCHES = 10  # Minimum matches to consider significant
    MATCH_RATIO_THRESHOLD = 0.75  # Lowe's ratio test threshold

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze image for copy-move forgery.

        Returns DimensionResult with detected clone regions.
        """
        if not HAS_NUMPY or not HAS_PIL:
            return DimensionResult(
                dimension="copymove",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="numpy and PIL required for copy-move detection"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="copymove",
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
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)
        except Exception as e:
            return DimensionResult(
                dimension="copymove",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot load image",
                    explanation=str(e)
                )]
            )

        evidence = []
        clone_result = None

        # Try keypoint method first (faster)
        if HAS_CV2:
            clone_result = self._detect_keypoint_based(image_array)
            if clone_result.detected:
                evidence.append(Evidence(
                    finding=f"Copy-move detected: {len(clone_result.clone_regions)} cloned region(s)",
                    explanation=(
                        f"Keypoint analysis found matching regions that appear to be "
                        f"copied within the image (method: {clone_result.method_used})"
                    ),
                    citation="Amerini et al. (2011) - SIFT-based copy-move detection"
                ))

        # Fall back to block-based if no CV2 or no detection
        if not clone_result or not clone_result.detected:
            if HAS_SCIPY:
                clone_result = self._detect_block_based(image_array)
                if clone_result.detected:
                    evidence.append(Evidence(
                        finding=f"Copy-move detected: {len(clone_result.clone_regions)} cloned region(s)",
                        explanation=(
                            f"Block DCT analysis found {len(clone_result.clone_regions)} "
                            f"region pair(s) with high similarity"
                        ),
                        citation="Fridrich et al. (2003) - Block-based copy-move detection"
                    ))

        # Determine state
        if clone_result and clone_result.detected:
            # Add details about detected regions
            for i, region in enumerate(clone_result.clone_regions[:3]):  # Top 3
                evidence.append(Evidence(
                    finding=f"Clone pair {i+1}: {region.width}x{region.height} pixels",
                    explanation=(
                        f"Source: ({region.source_x}, {region.source_y}), "
                        f"Target: ({region.target_x}, {region.target_y}), "
                        f"Similarity: {region.similarity:.1%}"
                    )
                ))

            return DimensionResult(
                dimension="copymove",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH if len(clone_result.clone_regions) >= 3 else Confidence.MEDIUM,
                evidence=evidence,
                methodology=f"Copy-move detection via {clone_result.method_used}"
            )

        # No clones detected
        evidence.append(Evidence(
            finding="No copy-move forgery detected",
            explanation="Image does not show signs of internal region duplication"
        ))

        return DimensionResult(
            dimension="copymove",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="Block DCT and keypoint analysis"
        )

    def _detect_block_based(self, image: np.ndarray) -> CopyMoveResult:
        """
        Block-based copy-move detection using DCT.

        Optimized with vectorized operations:
        - Batch block extraction using stride tricks
        - Vectorized DCT via matrix operations
        - Matrix multiplication for all-pairs similarity (replaces O(n²) loop)
        """
        import time
        start_time = time.time()

        # Convert to grayscale - vectorized
        gray = np.mean(image, axis=2).astype(np.float32)
        height, width = gray.shape

        if height < self.BLOCK_SIZE * 2 or width < self.BLOCK_SIZE * 2:
            return CopyMoveResult(detected=False, method_used="block_dct")

        step = self.BLOCK_SIZE // 2

        # Batch extract all blocks using stride tricks (vectorized)
        n_rows = (height - self.BLOCK_SIZE) // step + 1
        n_cols = (width - self.BLOCK_SIZE) // step + 1

        # Create view of all blocks without copying
        shape = (n_rows, n_cols, self.BLOCK_SIZE, self.BLOCK_SIZE)
        strides = (gray.strides[0] * step, gray.strides[1] * step,
                   gray.strides[0], gray.strides[1])
        all_blocks = np.lib.stride_tricks.as_strided(gray, shape=shape, strides=strides)

        # Reshape to (n_blocks, block_size, block_size)
        all_blocks = all_blocks.reshape(-1, self.BLOCK_SIZE, self.BLOCK_SIZE).copy()

        # Generate all positions
        ys, xs = np.mgrid[0:n_rows, 0:n_cols]
        all_positions = np.stack([xs.ravel() * step, ys.ravel() * step], axis=1)

        # Compute variance for all blocks at once (vectorized)
        block_variances = np.var(all_blocks, axis=(1, 2))

        # Filter low-variance blocks
        valid_mask = block_variances >= self.MIN_BLOCK_VARIANCE
        blocks = all_blocks[valid_mask]
        positions = all_positions[valid_mask]

        if len(blocks) < 2:
            return CopyMoveResult(detected=False, method_used="block_dct")

        # Batch DCT computation - vectorized over all blocks
        # DCT-II along rows then columns
        dct_blocks = fftpack.dct(fftpack.dct(blocks, axis=2, norm='ortho'), axis=1, norm='ortho')

        # Extract low-frequency features (top-left 4x4)
        features = dct_blocks[:, :4, :4].reshape(len(blocks), -1)[:, :self.DCT_COEFFICIENTS]

        # Also keep flattened raw blocks for verification
        raw_blocks = blocks.reshape(len(blocks), -1)

        # Normalize features (vectorized)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features_norm = features / norms

        raw_norms = np.linalg.norm(raw_blocks, axis=1, keepdims=True)
        raw_norms[raw_norms == 0] = 1
        raw_norm = raw_blocks / raw_norms

        # VECTORIZED SIMILARITY: Matrix multiplication gives all-pairs similarity
        # This replaces the O(n²) nested loop with optimized BLAS
        n_blocks = len(features_norm)

        # For large block counts, process in chunks to manage memory
        chunk_size = min(2000, n_blocks)
        clone_regions = []

        for i_start in range(0, n_blocks, chunk_size):
            i_end = min(i_start + chunk_size, n_blocks)

            # Compute similarities for this chunk against all blocks
            # Shape: (chunk_size, n_blocks)
            similarities = features_norm[i_start:i_end] @ features_norm.T

            # Compute spatial distances (vectorized)
            pos_chunk = positions[i_start:i_end]  # (chunk_size, 2)
            # Broadcast to compute all pairwise distances
            dx = pos_chunk[:, 0:1] - positions[:, 0]  # (chunk_size, n_blocks)
            dy = pos_chunk[:, 1:2] - positions[:, 1]
            distances = np.sqrt(dx**2 + dy**2)

            # Create mask: high similarity AND sufficient distance AND upper triangle
            global_i = np.arange(i_start, i_end)[:, None]
            global_j = np.arange(n_blocks)[None, :]

            mask = (similarities > self.SIMILARITY_THRESHOLD) & \
                   (distances > self.MIN_CLONE_DISTANCE) & \
                   (global_j > global_i)  # Upper triangle only

            # Get candidate pairs
            chunk_indices, j_indices = np.where(mask)
            i_indices = chunk_indices + i_start

            # Verify with pixel correlation (batch)
            if len(i_indices) > 0:
                pixel_sims = np.sum(raw_norm[i_indices] * raw_norm[j_indices], axis=1)
                valid = pixel_sims > self.PIXEL_VERIFY_THRESHOLD

                for idx in np.where(valid)[0]:
                    i, j = i_indices[idx], j_indices[idx]
                    clone_regions.append(CloneRegion(
                        source_x=int(positions[i, 0]), source_y=int(positions[i, 1]),
                        target_x=int(positions[j, 0]), target_y=int(positions[j, 1]),
                        width=self.BLOCK_SIZE,
                        height=self.BLOCK_SIZE,
                        similarity=float(pixel_sims[idx])
                    ))

        # Merge overlapping detections
        # OPTIMIZE: CYTHON - Region merging
        merged_regions = self._merge_clone_regions(clone_regions)

        elapsed_ms = (time.time() - start_time) * 1000

        return CopyMoveResult(
            detected=len(merged_regions) >= self.MIN_CLONE_REGIONS,
            clone_regions=merged_regions,
            method_used="block_dct",
            processing_time_ms=elapsed_ms
        )

    def _detect_keypoint_based(self, image: np.ndarray) -> CopyMoveResult:
        """
        Keypoint-based copy-move detection using ORB features.

        Faster than block-based for large images, but may miss
        small cloned regions.
        """
        import time
        start_time = time.time()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect keypoints and compute descriptors
        # Note: ORB is free, SIFT/SURF may have patent issues
        # OPTIMIZE: The OpenCV ORB is already optimized in C++
        orb = cv2.ORB_create(nfeatures=5000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < self.MIN_KEYPOINT_MATCHES * 2:
            return CopyMoveResult(detected=False, method_used="orb_keypoint")

        # Match features within the same image
        # OPTIMIZE: C - BFMatcher is already C++ but custom matching could be faster
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Find 2 nearest neighbors for ratio test
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        # Apply ratio test and filter self-matches
        good_matches = []
        for match_group in matches:
            if len(match_group) >= 3:
                # Skip first match (self-match)
                m, n = match_group[1], match_group[2]

                # Lowe's ratio test
                if m.distance < self.MATCH_RATIO_THRESHOLD * n.distance:
                    # Ensure minimum spatial distance
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    dist = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

                    if dist > self.MIN_CLONE_DISTANCE:
                        good_matches.append((m, pt1, pt2))

        if len(good_matches) < self.MIN_KEYPOINT_MATCHES:
            elapsed_ms = (time.time() - start_time) * 1000
            return CopyMoveResult(
                detected=False,
                method_used="orb_keypoint",
                processing_time_ms=elapsed_ms
            )

        # Cluster matches to find coherent clone regions
        # OPTIMIZE: CYTHON - Clustering algorithm
        clone_regions = self._cluster_keypoint_matches(good_matches)

        elapsed_ms = (time.time() - start_time) * 1000

        return CopyMoveResult(
            detected=len(clone_regions) > 0,
            clone_regions=clone_regions,
            method_used="orb_keypoint",
            processing_time_ms=elapsed_ms
        )

    def _cluster_keypoint_matches(
        self,
        matches: List[Tuple[Any, Tuple[float, float], Tuple[float, float]]]
    ) -> List[CloneRegion]:
        """
        Cluster keypoint matches into coherent clone regions.

        OPTIMIZE: CYTHON - This clustering could be much faster
        """
        if not matches:
            return []

        # Simple grid-based clustering
        # OPTIMIZE: C - Use proper clustering (DBSCAN) with spatial indexing
        cell_size = 64
        grid: Dict[Tuple[int, int], List] = {}

        for match, pt1, pt2 in matches:
            cell = (int(pt1[0] // cell_size), int(pt1[1] // cell_size))
            if cell not in grid:
                grid[cell] = []
            grid[cell].append((pt1, pt2, match.distance))

        # Find cells with multiple matches
        clone_regions = []
        for cell, cell_matches in grid.items():
            if len(cell_matches) >= 3:  # Minimum matches per cell
                # Calculate bounding box and average similarity
                x1s = [m[0][0] for m in cell_matches]
                y1s = [m[0][1] for m in cell_matches]
                x2s = [m[1][0] for m in cell_matches]
                y2s = [m[1][1] for m in cell_matches]

                # Source region
                src_x = int(min(x1s))
                src_y = int(min(y1s))
                # Target region
                tgt_x = int(min(x2s))
                tgt_y = int(min(y2s))

                width = int(max(max(x1s) - src_x, max(x2s) - tgt_x)) + 1
                height = int(max(max(y1s) - src_y, max(y2s) - tgt_y)) + 1

                # Calculate similarity from match distances
                avg_distance = np.mean([m[2] for m in cell_matches])
                similarity = max(0, 1 - avg_distance / 256)  # ORB distance is 0-256

                if width * height >= self.MIN_CLONE_AREA:
                    clone_regions.append(CloneRegion(
                        source_x=src_x, source_y=src_y,
                        target_x=tgt_x, target_y=tgt_y,
                        width=width, height=height,
                        similarity=similarity
                    ))

        return clone_regions

    def _merge_clone_regions(self, regions: List[CloneRegion]) -> List[CloneRegion]:
        """
        Merge overlapping clone detections.

        OPTIMIZE: CYTHON - Region merging with spatial indexing
        """
        if not regions:
            return []

        # Sort by position
        regions = sorted(regions, key=lambda r: (r.source_x, r.source_y))

        # Simple greedy merging
        # OPTIMIZE: C - Use union-find or R-tree for efficient merging
        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            # Find overlapping regions
            cluster = [r1]
            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                # Check if source regions overlap
                if (abs(r1.source_x - r2.source_x) < self.BLOCK_SIZE and
                    abs(r1.source_y - r2.source_y) < self.BLOCK_SIZE):
                    cluster.append(r2)
                    used.add(j)

            if len(cluster) >= 3:  # Minimum cluster size
                # Merge cluster into single region
                src_x = min(r.source_x for r in cluster)
                src_y = min(r.source_y for r in cluster)
                tgt_x = min(r.target_x for r in cluster)
                tgt_y = min(r.target_y for r in cluster)

                max_src_x = max(r.source_x + r.width for r in cluster)
                max_src_y = max(r.source_y + r.height for r in cluster)

                merged.append(CloneRegion(
                    source_x=src_x, source_y=src_y,
                    target_x=tgt_x, target_y=tgt_y,
                    width=max_src_x - src_x,
                    height=max_src_y - src_y,
                    similarity=np.mean([r.similarity for r in cluster])
                ))

        return merged


# =============================================================================
# NATIVE IMPLEMENTATION STUBS
# =============================================================================
# The following are stub signatures for C/Cython implementations that would
# replace the pure Python code above for production use.

"""
# cython: language_level=3
# copymove_native.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt

# SIMD-accelerated DCT block extraction
cpdef np.ndarray[np.float32_t, ndim=2] extract_dct_features_fast(
    np.ndarray[np.float32_t, ndim=2] gray_image,
    int block_size,
    int n_coefficients
):
    '''
    Extract DCT features from all blocks in image.

    This should use:
    - OpenMP for parallelization across blocks
    - AVX2/NEON intrinsics for DCT computation
    - Cache-friendly memory access patterns
    '''
    pass

# SIMD-accelerated similarity search
cpdef list find_similar_blocks_fast(
    np.ndarray[np.float32_t, ndim=2] features,
    float threshold,
    float min_distance
):
    '''
    Find pairs of similar feature vectors.

    This should use:
    - Locality-sensitive hashing for approximate nearest neighbor
    - SIMD for distance computation
    - Parallel reduction for candidate filtering
    '''
    pass
"""

"""
// copymove_native.c - Pure C implementation for maximum performance

#include <immintrin.h>  // AVX2 intrinsics
#include <omp.h>        // OpenMP

// DCT-II for 16x16 block using AVX2
void dct_16x16_avx2(const float* input, float* output) {
    // Implementation using AVX2 SIMD intrinsics
    // Would achieve ~10x speedup over numpy

    __m256 row, col, dct_coeff;
    // ... AVX2 implementation
}

// Parallel block matching with OpenMP
typedef struct {
    int src_x, src_y;
    int tgt_x, tgt_y;
    float similarity;
} match_t;

int find_matches_parallel(
    const float* features,      // [n_blocks, n_features]
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    match_t* matches,           // Output buffer
    int max_matches
) {
    int n_matches = 0;

    #pragma omp parallel for reduction(+:n_matches) schedule(dynamic)
    for (int i = 0; i < n_blocks; i++) {
        for (int j = i + 1; j < n_blocks; j++) {
            // SIMD dot product for similarity
            float sim = dot_product_avx2(
                &features[i * n_features],
                &features[j * n_features],
                n_features
            );

            if (sim > threshold) {
                // Thread-safe match recording
                #pragma omp critical
                {
                    if (n_matches < max_matches) {
                        matches[n_matches++] = (match_t){...};
                    }
                }
            }
        }
    }

    return n_matches;
}
"""
