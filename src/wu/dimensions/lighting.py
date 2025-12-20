"""
Lighting direction consistency forensic analysis. Zanes notes:

Detects manipulation by analyzing whether lighting is consistent across
different regions of an image. Inconsistent lighting suggests compositing.

Analysis methods:
1. Gradient-based light direction estimation
2. Specular highlight analysis
3. Shadow direction consistency
4. Regional illumination comparison

Court relevance:
    "The lighting direction on the subject's face is inconsistent with
    the lighting on the background, indicating the face was composited
    from a different photograph."

PERFORMANCE NOTES:
    This module contains algorithms that benefit from optimization.
    Areas marked with:

    # OPTIMIZE: CYTHON - Would benefit from Cython compilation
    # OPTIMIZE: C - Should be rewritten in C for production
    # OPTIMIZE: ASM - Critical inner loop, consider SIMD/Assembly

References:
    Johnson, M.K. & Farid, H. (2005). Exposing Digital Forgeries by
        Detecting Inconsistencies in Lighting. ACM Multimedia and
        Security Workshop.
    Kee, E., O'Brien, J.F. & Farid, H. (2013). Exposing Photo Manipulation
        with Inconsistent Shadows. ACM Transactions on Graphics.
    Riess, C. & Angelopoulou, E. (2010). Scene Illumination as an Indicator
        of Image Manipulation. Information Hiding.
    Fan, W., Wang, K., Cayre, F., & Xiong, Z. (2012). 3D Lighting-Based
        Image Forgery Detection Using Shape-from-Shading.
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
    from scipy import ndimage
    from scipy.signal import convolve2d
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
class LightVector:
    """Estimated light direction vector."""
    azimuth: float  # Horizontal angle in degrees (0-360)
    elevation: float  # Vertical angle in degrees (0-90)
    confidence: float  # 0-1, confidence in this estimate

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to unit vector (x, y, z)."""
        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)
        x = math.cos(el_rad) * math.cos(az_rad)
        y = math.cos(el_rad) * math.sin(az_rad)
        z = math.sin(el_rad)
        return (x, y, z)

    def angle_to(self, other: 'LightVector') -> float:
        """Compute angle between two light vectors in degrees."""
        v1 = self.to_cartesian()
        v2 = other.to_cartesian()
        dot = sum(a * b for a, b in zip(v1, v2))
        # Clamp to avoid numerical issues
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))


@dataclass
class RegionLighting:
    """Lighting analysis for a specific image region."""
    x: int
    y: int
    width: int
    height: int
    light_vector: LightVector
    mean_brightness: float
    specular_strength: float  # Strength of specular highlights


@dataclass
class LightingResult:
    """Result of lighting consistency analysis."""
    global_light: LightVector  # Estimated primary light direction
    region_lights: List[RegionLighting] = field(default_factory=list)
    max_inconsistency_angle: float = 0.0  # Largest angle difference
    inconsistent_regions: List[RegionLighting] = field(default_factory=list)
    processing_time_ms: float = 0.0


class LightingAnalyzer:
    """
    Analyzes lighting consistency across image regions.

    Light direction can be estimated from:
    - Surface shading gradients (shape-from-shading)
    - Specular highlight positions
    - Shadow directions and orientations

    Inconsistent lighting across regions strongly suggests compositing,
    as natural photographs have coherent illumination.

    Limitations:
    - Works best with convex surfaces (faces, spheres)
    - Multiple light sources complicate analysis
    - Very flat lighting reduces detection ability
    - Requires sufficient texture/shading variation
    """

    REGION_SIZE = 64  # Size of analysis regions
    INCONSISTENCY_THRESHOLD = 90.0  # Degrees difference to flag (tuned to reduce FPs)
    MIN_REGION_VARIANCE = 75.0  # Minimum variance for reliable estimation
    SPECULAR_PERCENTILE = 99  # Percentile for specular detection
    MIN_INCONSISTENT_REGIONS = 12  # Minimum regions to flag as suspicious
    MAX_INCONSISTENT_RATIO = 0.12
    MIN_CLUSTER_SIZE = 4  # Minimum cluster size for merged regions

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze image for lighting consistency.

        Returns DimensionResult indicating lighting coherence.
        """
        if not HAS_NUMPY or not HAS_PIL:
            return DimensionResult(
                dimension="lighting",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="numpy and PIL required for lighting analysis"
                )]
            )

        if not HAS_SCIPY:
            return DimensionResult(
                dimension="lighting",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="scipy not available",
                    explanation="scipy required for gradient computation"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="lighting",
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
                dimension="lighting",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot load image",
                    explanation=str(e)
                )]
            )

        evidence = []

        # Perform lighting analysis
        result = self._analyze_lighting(image_array)

        # Report global light estimate
        gl = result.global_light
        evidence.append(Evidence(
            finding=f"Primary light direction: azimuth {gl.azimuth:.0f}°, elevation {gl.elevation:.0f}°",
            explanation=(
                f"Estimated dominant light source direction "
                f"(confidence: {gl.confidence:.0%})"
            )
        ))

        if gl.confidence < 0.3:
            evidence.append(Evidence(
                finding="Low confidence in lighting estimation",
                explanation=(
                    "Image has insufficient shading variation for reliable "
                    "lighting analysis. May be very flat lighting or uniform surfaces."
                )
            ))
            return DimensionResult(
                dimension="lighting",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.LOW,
                evidence=evidence,
                methodology="Gradient-based lighting estimation"
            )

        # Check for inconsistent regions
        if result.inconsistent_regions:
            n_regions = len(result.inconsistent_regions)
            n_total = len(result.region_lights) if result.region_lights else 1
            inconsistent_ratio = n_regions / n_total

            # Sanity check: if too many regions are "inconsistent", it's noise not signal
            if inconsistent_ratio > self.MAX_INCONSISTENT_RATIO:
                evidence.append(Evidence(
                    finding=f"High lighting variance ({n_regions}/{n_total} regions differ)",
                    explanation=(
                        f"Too many regions ({inconsistent_ratio:.0%}) show different lighting. "
                        f"This suggests unreliable gradient estimation rather than manipulation."
                    )
                ))
                return DimensionResult(
                    dimension="lighting",
                    state=DimensionState.UNCERTAIN,
                    confidence=Confidence.LOW,
                    evidence=evidence,
                    methodology="Gradient-based lighting estimation"
                )

            # Require minimum number of inconsistent regions to flag
            if n_regions < self.MIN_INCONSISTENT_REGIONS:
                # Not enough regions - fall through to CONSISTENT
                pass
            else:
                evidence.append(Evidence(
                    finding=f"Lighting inconsistency: {n_regions} region(s) with different lighting",
                    explanation=(
                        f"Found {n_regions} region(s) where the estimated light direction "
                        f"differs significantly from the rest of the image. "
                        f"Maximum deviation: {result.max_inconsistency_angle:.0f}°"
                    ),
                    citation="Johnson & Farid (2005) - Lighting inconsistency detection"
                ))

                for i, region in enumerate(result.inconsistent_regions[:3]):
                    angle_diff = region.light_vector.angle_to(result.global_light)
                    evidence.append(Evidence(
                        finding=(
                            f"Suspicious region {i+1}: ({region.x}, {region.y}) "
                            f"{region.width}x{region.height}px"
                        ),
                        explanation=(
                            f"Light direction differs by {angle_diff:.0f}° from image primary. "
                            f"Region light: azimuth {region.light_vector.azimuth:.0f}°, "
                            f"elevation {region.light_vector.elevation:.0f}°"
                        )
                    ))

                return DimensionResult(
                    dimension="lighting",
                    state=DimensionState.SUSPICIOUS,
                    confidence=Confidence.MEDIUM,
                    evidence=evidence,
                    methodology="Regional lighting direction comparison",
                    raw_data={
                        "global_light": {
                            "azimuth": gl.azimuth,
                            "elevation": gl.elevation,
                            "confidence": gl.confidence
                        },
                        "inconsistent_regions": [
                            {"x": r.x, "y": r.y, "width": r.width, "height": r.height,
                             "azimuth": r.light_vector.azimuth,
                             "elevation": r.light_vector.elevation,
                             "angle_diff": r.light_vector.angle_to(result.global_light)}
                            for r in result.inconsistent_regions
                        ]
                    }
                )

        # Lighting appears consistent
        evidence.append(Evidence(
            finding="Lighting consistent across image",
            explanation=(
                f"Light direction estimates are consistent across analyzed regions "
                f"(maximum deviation: {result.max_inconsistency_angle:.0f}°)"
            )
        ))

        return DimensionResult(
            dimension="lighting",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="Gradient-based lighting consistency analysis"
        )

    def _analyze_lighting(self, image: np.ndarray) -> LightingResult:
        """
        Analyze lighting direction across image.

        OPTIMIZE: C - This entire method benefits from C implementation
        """
        import time
        start_time = time.time()

        # Convert to grayscale (luminance)
        # OPTIMIZE: CYTHON - Color conversion
        gray = self._to_luminance(image)

        height, width = gray.shape

        # Compute image gradients
        # Use native SIMD when available for efficiency
        gx, gy = self._compute_gradients(gray)

        # Estimate global light direction
        global_light = self._estimate_light_direction(gray, gx, gy)

        # Analyze regional lighting
        region_lights = []
        region_size = self.REGION_SIZE
        step = region_size // 2  # 50% overlap

        for y in range(0, height - region_size, step):
            for x in range(0, width - region_size, step):
                region_gray = gray[y:y+region_size, x:x+region_size]
                region_gx = gx[y:y+region_size, x:x+region_size]
                region_gy = gy[y:y+region_size, x:x+region_size]

                # Skip low-variance regions
                if np.var(region_gray) < self.MIN_REGION_VARIANCE:
                    continue

                light_vec = self._estimate_light_direction(
                    region_gray, region_gx, region_gy
                )

                # Compute specular strength
                specular = self._compute_specular_strength(region_gray)

                region_lights.append(RegionLighting(
                    x=x, y=y,
                    width=region_size, height=region_size,
                    light_vector=light_vec,
                    mean_brightness=np.mean(region_gray),
                    specular_strength=specular
                ))

        # Find inconsistent regions
        inconsistent = []
        max_angle = 0.0

        for region in region_lights:
            if region.light_vector.confidence < 0.3:
                continue

            angle = region.light_vector.angle_to(global_light)
            max_angle = max(max_angle, angle)

            if angle > self.INCONSISTENCY_THRESHOLD:
                inconsistent.append(region)

        # Merge adjacent inconsistent regions
        merged_inconsistent = self._merge_inconsistent_regions(inconsistent)

        elapsed_ms = (time.time() - start_time) * 1000

        return LightingResult(
            global_light=global_light,
            region_lights=region_lights,
            max_inconsistency_angle=max_angle,
            inconsistent_regions=merged_inconsistent,
            processing_time_ms=elapsed_ms
        )

    def _to_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to luminance using ITU-R BT.601.

        OPTIMIZE: CYTHON - Simple but frequent operation
        """
        if len(image.shape) == 2:
            return image.astype(np.float64)

        # Standard luminance coefficients
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.float64)

    def _compute_gradients(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image gradients using Sobel operators.

        Uses native SIMD when available for accelerated computation.
        """
        # Use native SIMD implementation (C/ASM optimized)
        if HAS_NATIVE_SIMD:
            return native_simd.sobel_3x3(gray)

        # Python/scipy fallback
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64) / 8.0

        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float64) / 8.0

        gx = convolve2d(gray, sobel_x, mode='same', boundary='symm')
        gy = convolve2d(gray, sobel_y, mode='same', boundary='symm')

        return gx, gy

    def _estimate_light_direction(
        self,
        gray: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray
    ) -> LightVector:
        """
        Estimate light direction from shading gradients.

        Uses the assumption that surfaces are Lambertian (diffuse) and
        the relationship between surface normals and image brightness.

        For Lambertian surfaces: I = L · N
        Where I is intensity, L is light direction, N is surface normal.

        OPTIMIZE: C - Linear algebra operations
        """
        height, width = gray.shape

        # Compute gradient magnitude and direction
        # OPTIMIZE: ASM - Vector operations
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Weight by gradient magnitude (stronger gradients = more reliable)
        weights = grad_mag / (np.max(grad_mag) + 1e-6)

        # Filter out very weak gradients
        mask = grad_mag > np.percentile(grad_mag, 25)

        if np.sum(mask) < 100:
            # Not enough gradient information
            return LightVector(0, 45, 0.0)

        # Estimate light direction using weighted average of gradient directions
        # Light comes from the direction of increasing brightness
        # OPTIMIZE: CYTHON - Weighted statistics
        weighted_gx = np.sum(gx[mask] * weights[mask])
        weighted_gy = np.sum(gy[mask] * weights[mask])

        # Normalize
        magnitude = math.sqrt(weighted_gx**2 + weighted_gy**2)
        if magnitude < 1e-6:
            return LightVector(0, 45, 0.0)

        weighted_gx /= magnitude
        weighted_gy /= magnitude

        # Convert to azimuth angle (light direction)
        # Note: gradient points uphill (toward light)
        azimuth = math.degrees(math.atan2(weighted_gy, weighted_gx))
        if azimuth < 0:
            azimuth += 360

        # Estimate elevation from brightness distribution
        # Brighter regions with low gradients suggest more frontal lighting
        brightness_gradient_ratio = np.mean(gray[mask]) / (magnitude * 100 + 1)
        elevation = min(90, max(0, brightness_gradient_ratio * 45))

        # Confidence based on gradient consistency
        # OPTIMIZE: CYTHON - Variance computation
        gx_var = np.var(gx[mask])
        gy_var = np.var(gy[mask])

        # Lower variance in gradient direction = higher confidence
        total_var = gx_var + gy_var
        confidence = 1.0 / (1.0 + total_var / 1000)

        return LightVector(
            azimuth=azimuth,
            elevation=elevation,
            confidence=confidence
        )

    def _compute_specular_strength(self, region: np.ndarray) -> float:
        """
        Compute strength of specular highlights in region.

        Specular highlights appear as very bright spots that can
        indicate light source position.

        OPTIMIZE: CYTHON - Percentile computation
        """
        threshold = np.percentile(region, self.SPECULAR_PERCENTILE)
        specular_pixels = region > threshold

        if np.sum(specular_pixels) == 0:
            return 0.0

        # Strength based on how much brighter speculars are
        mean_specular = np.mean(region[specular_pixels])
        mean_normal = np.mean(region[~specular_pixels])

        if mean_normal < 1:
            return 0.0

        return (mean_specular - mean_normal) / mean_normal

    def _merge_inconsistent_regions(
        self,
        regions: List[RegionLighting]
    ) -> List[RegionLighting]:
        """
        Merge adjacent inconsistent regions.

        OPTIMIZE: CYTHON - Spatial clustering
        """
        if not regions:
            return []

        # Sort by position
        regions = sorted(regions, key=lambda r: (r.y, r.x))

        # Simple greedy merge
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

                # Check if adjacent and similar lighting
                adjacent = (
                    abs(r1.x - r2.x) < self.REGION_SIZE * 2 and
                    abs(r1.y - r2.y) < self.REGION_SIZE * 2
                )

                similar_light = r1.light_vector.angle_to(r2.light_vector) < 20

                if adjacent and similar_light:
                    cluster.append(r2)
                    used.add(j)

            # Only keep significant clusters
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                # Merge into bounding box, use average light
                min_x = min(r.x for r in cluster)
                min_y = min(r.y for r in cluster)
                max_x = max(r.x + r.width for r in cluster)
                max_y = max(r.y + r.height for r in cluster)

                # Average light direction
                avg_az = np.mean([r.light_vector.azimuth for r in cluster])
                avg_el = np.mean([r.light_vector.elevation for r in cluster])
                avg_conf = np.mean([r.light_vector.confidence for r in cluster])

                merged.append(RegionLighting(
                    x=min_x, y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    light_vector=LightVector(avg_az, avg_el, avg_conf),
                    mean_brightness=np.mean([r.mean_brightness for r in cluster]),
                    specular_strength=np.mean([r.specular_strength for r in cluster])
                ))

        return merged


# =============================================================================
# NATIVE IMPLEMENTATION STUBS
# =============================================================================

"""
# cython: language_level=3
# lighting_native.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt, atan2, cos, sin

# SIMD-accelerated gradient computation
cpdef tuple compute_gradients_fast(
    np.ndarray[np.float64_t, ndim=2] gray
):
    '''
    Fast Sobel gradient computation.

    Should use:
    - Separable convolution (2 1D instead of 2D)
    - SIMD for parallel pixel processing
    - Cache-blocking for large images
    '''
    pass

# Fast light direction estimation
cpdef tuple estimate_light_direction_fast(
    np.ndarray[np.float64_t, ndim=2] gray,
    np.ndarray[np.float64_t, ndim=2] gx,
    np.ndarray[np.float64_t, ndim=2] gy
):
    '''
    Fast light direction estimation.

    Should use:
    - Parallel reduction for weighted sums
    - SIMD for variance computation
    '''
    pass
"""

"""
// lighting_native.c - Pure C implementation

#include <immintrin.h>
#include <omp.h>
#include <math.h>

// AVX2-accelerated Sobel gradient
void compute_sobel_avx2(
    const double* gray,
    double* gx,
    double* gy,
    int width, int height
) {
    // Use separable convolution for speed
    // Horizontal pass
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x += 4) {
            // Process 4 pixels at once with AVX2
            __m256d left = _mm256_loadu_pd(&gray[y * width + x - 1]);
            __m256d right = _mm256_loadu_pd(&gray[y * width + x + 1]);
            __m256d diff = _mm256_sub_pd(right, left);
            _mm256_storeu_pd(&gx[y * width + x], diff);
        }
    }

    // Vertical pass (similar)
    // ... AVX2 implementation
}

// Parallel light direction estimation
typedef struct {
    double azimuth;
    double elevation;
    double confidence;
} light_result_t;

light_result_t estimate_light_parallel(
    const double* gray,
    const double* gx,
    const double* gy,
    int width, int height
) {
    double weighted_gx = 0.0;
    double weighted_gy = 0.0;
    double total_weight = 0.0;

    #pragma omp parallel for reduction(+:weighted_gx,weighted_gy,total_weight)
    for (int i = 0; i < width * height; i++) {
        double mag = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        double weight = mag;  // Weight by gradient magnitude

        weighted_gx += gx[i] * weight;
        weighted_gy += gy[i] * weight;
        total_weight += weight;
    }

    // Compute azimuth from weighted average
    double azimuth = atan2(weighted_gy, weighted_gx) * 180.0 / M_PI;
    if (azimuth < 0) azimuth += 360;

    light_result_t result = {azimuth, 45.0, 0.5};
    return result;
}
"""
