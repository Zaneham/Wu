"""
PRNU (Photo Response Non-Uniformity) forensic analysis.

Each digital camera sensor has a unique noise pattern caused by manufacturing
imperfections - like a fingerprint. PRNU analysis can:
1. Link a photo to a specific camera
2. Verify if two photos came from the same camera
3. Detect manipulated regions (where PRNU pattern is disrupted)

PERFORMANCE NOTES:
    This module contains computationally intensive algorithms. Areas marked with:

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
    Lukas, J., Fridrich, J., & Goljan, M. (2006). Digital Camera Identification
        from Sensor Pattern Noise. IEEE Transactions on Information Forensics
        and Security, 1(2), 205-214.
    Chen, M., Fridrich, J., Goljan, M., & Lukas, J. (2008). Determining Image
        Origin and Integrity Using Sensor Noise. IEEE Transactions on
        Information Forensics and Security, 3(1), 74-90.
    Goljan, M. (2008). Digital Camera Identification from Images - Estimating
        False Acceptance Probability. Digital Watermarking, LNCS 5041.
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
    from scipy.signal import wiener
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence

try:
    from wu.native import simd as native_simd
    HAS_NATIVE_SIMD = native_simd.is_available()
except ImportError:
    HAS_NATIVE_SIMD = False


@dataclass
class PRNUFingerprint:
    """A camera's PRNU fingerprint extracted from reference images."""
    camera_id: str
    width: int
    height: int
    fingerprint: Any  # np.ndarray when available
    n_reference_images: int
    quality_score: float  # 0-1, higher is better


@dataclass
class PRNUMatch:
    """Result of matching an image against a PRNU fingerprint."""
    camera_id: str
    correlation: float  # Peak-to-correlation energy ratio
    p_value: float  # Statistical significance
    matched: bool  # Whether correlation exceeds threshold


@dataclass
class PRNUResult:
    """Result of PRNU analysis."""
    noise_residual_quality: float  # Quality of extracted noise
    matches: List[PRNUMatch] = field(default_factory=list)
    region_consistency: Optional[float] = None  # For manipulation detection
    inconsistent_regions: List[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    processing_time_ms: float = 0.0


class PRNUAnalyzer:
    """
    Analyzes images for PRNU (sensor noise) patterns.

    PRNU is caused by:
    - Pixel response non-uniformity (manufacturing variations)
    - Dark current variations
    - Photo-response variations

    The PRNU pattern is:
    - Unique to each sensor (like a fingerprint)
    - Stable over time
    - Present in every image from that camera
    - Disrupted by image manipulation

    Court relevance:
        "PRNU analysis shows this image was NOT taken by the suspect's
        camera, despite claims to the contrary."

    Limitations:
    - Requires reference images from the same camera for matching
    - Low-texture regions have weak PRNU signal
    - Heavy post-processing can obscure PRNU
    - Resizing destroys PRNU
    """

    # Analysis parameters
    BLOCK_SIZE = 128  # Size of blocks for region analysis
    MIN_CORRELATION_THRESHOLD = 60.0  # PCE threshold for positive match
    NOISE_QUALITY_THRESHOLD = 0.1  # Minimum noise quality to analyze

    def analyze(
        self,
        file_path: str,
        reference_fingerprints: Optional[List[PRNUFingerprint]] = None
    ) -> DimensionResult:
        """
        Analyze image for PRNU patterns.

        If reference fingerprints provided, attempts camera identification.
        Otherwise, performs internal consistency analysis.
        """
        if not HAS_NUMPY or not HAS_PIL:
            return DimensionResult(
                dimension="prnu",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="numpy and PIL required for PRNU analysis"
                )]
            )

        if not HAS_SCIPY:
            return DimensionResult(
                dimension="prnu",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="scipy not available",
                    explanation="scipy required for denoising operations"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="prnu",
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
                image_array = np.array(img, dtype=np.float64)
        except Exception as e:
            return DimensionResult(
                dimension="prnu",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot load image",
                    explanation=str(e)
                )]
            )

        evidence = []

        # Extract noise residual
        noise_residual = self._extract_noise_residual(image_array)
        noise_quality = self._estimate_noise_quality(noise_residual)

        evidence.append(Evidence(
            finding=f"Noise residual quality: {noise_quality:.3f}",
            explanation=(
                "Quality of extractable sensor noise pattern. "
                "Higher values indicate more reliable PRNU analysis."
            )
        ))

        if noise_quality < self.NOISE_QUALITY_THRESHOLD:
            evidence.append(Evidence(
                finding="Insufficient noise for PRNU analysis",
                explanation=(
                    "Image has too little texture or has been heavily processed, "
                    "making PRNU extraction unreliable."
                )
            ))
            return DimensionResult(
                dimension="prnu",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.LOW,
                evidence=evidence,
                methodology="PRNU extraction via Wiener denoising"
            )

        # If reference fingerprints provided, attempt matching
        if reference_fingerprints:
            matches = self._match_against_references(
                noise_residual, reference_fingerprints
            )

            for match in matches:
                if match.matched:
                    evidence.append(Evidence(
                        finding=f"PRNU match: Camera '{match.camera_id}'",
                        explanation=(
                            f"Image shows strong correlation (PCE={match.correlation:.1f}) "
                            f"with sensor fingerprint from camera '{match.camera_id}'. "
                            f"P-value: {match.p_value:.2e}"
                        ),
                        citation="Lukas et al. (2006) - PRNU camera identification"
                    ))

            if any(m.matched for m in matches):
                return DimensionResult(
                    dimension="prnu",
                    state=DimensionState.VERIFIED,
                    confidence=Confidence.HIGH,
                    evidence=evidence,
                    methodology="PRNU fingerprint matching",
                    raw_data={"matches": [
                        {"camera_id": m.camera_id, "correlation": m.correlation,
                         "p_value": m.p_value, "matched": m.matched}
                        for m in matches
                    ]}
                )

        # Perform internal consistency analysis (manipulation detection)
        consistency_result = self._analyze_region_consistency(noise_residual)

        if consistency_result.inconsistent_regions:
            n_regions = len(consistency_result.inconsistent_regions)
            evidence.append(Evidence(
                finding=f"PRNU inconsistency: {n_regions} suspicious region(s)",
                explanation=(
                    f"Image contains {n_regions} region(s) with inconsistent "
                    f"sensor noise patterns, suggesting possible manipulation."
                ),
                citation="Chen et al. (2008) - Image integrity via sensor noise"
            ))

            for i, (x, y, w, h) in enumerate(consistency_result.inconsistent_regions[:3]):
                evidence.append(Evidence(
                    finding=f"Inconsistent region {i+1}: ({x}, {y}) {w}x{h}px",
                    explanation="Region shows different PRNU pattern than surrounding areas"
                ))

            return DimensionResult(
                dimension="prnu",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
                evidence=evidence,
                methodology="PRNU regional consistency analysis"
            )

        # No issues found
        evidence.append(Evidence(
            finding="PRNU consistent across image",
            explanation="Sensor noise pattern is consistent, no signs of compositing detected"
        ))

        return DimensionResult(
            dimension="prnu",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="PRNU consistency analysis"
        )

    def _extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract sensor noise residual from image.

        Noise = Image - Denoised(Image)

        The residual contains:
        - PRNU (what we want)
        - Random shot noise
        - Compression artifacts

        OPTIMIZE: C/ASM - Denoising is computationally expensive
        """
        # Convert to float if needed
        if image.dtype != np.float64:
            image = image.astype(np.float64)

        # Apply Wiener filter for denoising
        # OPTIMIZE: CYTHON - Wiener filter implementation
        # OPTIMIZE: ASM - Convolution in frequency domain with SIMD
        if len(image.shape) == 3:
            # Color image - process each channel
            denoised = np.zeros_like(image)
            for c in range(image.shape[2]):
                denoised[:, :, c] = wiener(image[:, :, c], mysize=5)

            # Average across color channels for PRNU
            # OPTIMIZE: CYTHON - Channel averaging
            noise = np.mean(image - denoised, axis=2)
        else:
            # Grayscale
            denoised = wiener(image, mysize=5)
            noise = image - denoised

        return noise

    def _estimate_noise_quality(self, noise: np.ndarray) -> float:
        """
        Estimate the quality of extracted noise for PRNU analysis.

        Higher variance in textured regions = better PRNU signal.

        OPTIMIZE: CYTHON - Quality estimation
        """
        # Compute local variance
        # OPTIMIZE: C - Local variance computation with integral images
        local_mean = ndimage.uniform_filter(noise, size=16)
        local_sqr_mean = ndimage.uniform_filter(noise**2, size=16)
        local_var = local_sqr_mean - local_mean**2

        # Quality is based on variance distribution
        quality = np.sqrt(np.mean(local_var[local_var > 0]))

        # Normalize to 0-1 range (empirically determined)
        return min(1.0, quality / 10.0)

    def _match_against_references(
        self,
        noise: np.ndarray,
        fingerprints: List[PRNUFingerprint]
    ) -> List[PRNUMatch]:
        """
        Match image noise against reference PRNU fingerprints.

        Uses Peak-to-Correlation Energy (PCE) ratio for robust matching.

        OPTIMIZE: C/ASM - Cross-correlation is O(n²) or O(n log n) with FFT
        """
        matches = []

        for fp in fingerprints:
            # Check size compatibility
            if fp.width != noise.shape[1] or fp.height != noise.shape[0]:
                # Size mismatch - cannot compare
                matches.append(PRNUMatch(
                    camera_id=fp.camera_id,
                    correlation=0.0,
                    p_value=1.0,
                    matched=False
                ))
                continue

            # Compute normalized cross-correlation
            # OPTIMIZE: ASM - FFT-based correlation with SIMD
            pce, p_value = self._compute_pce(noise, fp.fingerprint)

            matches.append(PRNUMatch(
                camera_id=fp.camera_id,
                correlation=pce,
                p_value=p_value,
                matched=pce > self.MIN_CORRELATION_THRESHOLD
            ))

        return matches

    def _compute_pce(
        self,
        noise: np.ndarray,
        fingerprint: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Peak-to-Correlation Energy ratio.

        PCE = peak² / mean(correlation² excluding peak neighborhood)

        Higher PCE = stronger match.

        OPTIMIZE: ASM - FFT and correlation with SIMD intrinsics
        """
        # --- NATIVE OPTIMIZATION IF AVAILABLE ---
        if HAS_NATIVE_SIMD:
            # 1. Compute stats using AVX2
            mean_n, var_n = native_simd.mean_variance_asm(noise)
            mean_f, var_f = native_simd.mean_variance_asm(fingerprint)
            
            # 2. Normalize
            # Note: We still do element-wise op in Python unless we add vector-scalar-op to ASM
            # But we saved the reduction time.
            noise_norm = noise - mean_n
            fp_norm = fingerprint - mean_f

            noise_std = math.sqrt(var_n)
            fp_std = math.sqrt(var_f)

            if noise_std == 0 or fp_std == 0:
                return 0.0, 1.0

            noise_norm /= noise_std
            fp_norm /= fp_std
            
            # 3. FFT (Still Python/NumPy for now)
            fft_noise = np.fft.fft2(noise_norm)
            fft_fp = np.fft.fft2(fp_norm)
            correlation = np.fft.ifft2(fft_noise * np.conj(fft_fp)).real

            # 4. Find Peak using ASM
            # Flatten for scanning
            peak_val, peak_flat_idx = native_simd.find_peak_asm(correlation)
            
            # Convert flat index to (y, x)
            h, w = correlation.shape
            py, px = divmod(int(peak_flat_idx), w)
            peak_idx = (py, px)
            
            # 5. Compute Energy
            # Total energy = sum(x^2)
            total_energy = native_simd.correlation_sum_asm(correlation.ravel(), correlation.ravel())
            
            # Subtract peak neighborhood energy (11x11)
            # Brute force subtraction is faster than creating mask array
            peak_energy_sum = 0.0
            y_start = max(0, py - 5)
            y_end = min(h, py + 6)
            x_start = max(0, px - 5)
            x_end = min(w, px + 6)
            
            # Create a view of the neighborhood (fast)
            neighborhood = correlation[y_start:y_end, x_start:x_end]
            peak_energy_sum = np.sum(neighborhood**2)
            
            noise_energy_sum = total_energy - peak_energy_sum
            
            # Count pixels
            total_pixels = w * h
            peak_pixels = (y_end - y_start) * (x_end - x_start)
            noise_pixels = total_pixels - peak_pixels
            
            if noise_pixels > 0:
                noise_energy_mean = noise_energy_sum / noise_pixels
            else:
                noise_energy_mean = 0.0

            if noise_energy_mean <= 1e-15: # Avoid div by zero
                 pce = float('inf') if peak_val != 0 else 0
            else:
                 pce = (peak_val**2) / noise_energy_mean
                 
            p_value = math.exp(-pce / 2) if pce < 100 else 0.0
            return pce, p_value

        # --- FALLBACK PYTHON IMPLEMENTATION ---
        # Normalize both signals
        # OPTIMIZE: CYTHON - Normalization
        noise_norm = noise - np.mean(noise)
        fp_norm = fingerprint - np.mean(fingerprint)

        noise_std = np.std(noise_norm)
        fp_std = np.std(fp_norm)

        if noise_std == 0 or fp_std == 0:
            return 0.0, 1.0

        noise_norm /= noise_std
        fp_norm /= fp_std

        # Compute cross-correlation via FFT
        # OPTIMIZE: ASM - FFT with AVX2/NEON
        # OPTIMIZE: C - Use FFTW library for production
        fft_noise = np.fft.fft2(noise_norm)
        fft_fp = np.fft.fft2(fp_norm)
        correlation = np.fft.ifft2(fft_noise * np.conj(fft_fp)).real

        # Find peak
        peak_idx = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
        peak_value = correlation[peak_idx]

        # Compute PCE (exclude 11x11 neighborhood around peak)
        # OPTIMIZE: CYTHON - Neighborhood exclusion
        mask = np.ones_like(correlation, dtype=bool)
        y, x = peak_idx
        mask[max(0, y-5):min(mask.shape[0], y+6),
             max(0, x-5):min(mask.shape[1], x+6)] = False

        noise_energy = np.mean(correlation[mask]**2)

        if noise_energy == 0:
            pce = float('inf') if peak_value != 0 else 0
        else:
            pce = (peak_value**2) / noise_energy

        # Approximate p-value using chi-squared distribution
        # OPTIMIZE: Could use more accurate statistical model
        p_value = math.exp(-pce / 2) if pce < 100 else 0.0

        return pce, p_value

    def _analyze_region_consistency(self, noise: np.ndarray) -> PRNUResult:
        """
        Analyze PRNU consistency across image regions.

        Detects manipulation by finding regions with different noise patterns.

        OPTIMIZE: C - Region analysis is O(n) but with large constants
        """
        import time
        start_time = time.time()

        height, width = noise.shape
        block_size = self.BLOCK_SIZE

        # Skip if image too small
        if height < block_size * 2 or width < block_size * 2:
            return PRNUResult(
                noise_residual_quality=0,
                region_consistency=1.0,
                inconsistent_regions=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Extract blocks and compute statistics
        # OPTIMIZE: C - Block extraction with cache-friendly access
        block_stats = []

        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block = noise[y:y+block_size, x:x+block_size]

                # Skip low-variance blocks
                variance = np.var(block)
                if variance < 0.1:
                    continue

                # Compute block statistics
                mean = np.mean(block)
                std = np.std(block)
                skew = self._compute_skewness(block)

                block_stats.append({
                    'x': x, 'y': y,
                    'mean': mean, 'std': std, 'skew': skew,
                    'variance': variance
                })

        if len(block_stats) < 4:
            return PRNUResult(
                noise_residual_quality=0,
                region_consistency=1.0,
                inconsistent_regions=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Find outlier blocks
        # OPTIMIZE: CYTHON - Outlier detection
        means = np.array([b['mean'] for b in block_stats])
        stds = np.array([b['std'] for b in block_stats])

        global_mean = np.median(means)
        global_std = np.median(stds)

        # Mark blocks significantly different from median
        mean_threshold = 3 * np.std(means)
        std_threshold = 3 * np.std(stds)

        inconsistent_regions = []
        for block in block_stats:
            if (abs(block['mean'] - global_mean) > mean_threshold or
                abs(block['std'] - global_std) > std_threshold):
                inconsistent_regions.append(
                    (block['x'], block['y'], block_size, block_size)
                )

        # Compute overall consistency score
        n_consistent = len(block_stats) - len(inconsistent_regions)
        consistency = n_consistent / len(block_stats) if block_stats else 1.0

        elapsed_ms = (time.time() - start_time) * 1000

        return PRNUResult(
            noise_residual_quality=self._estimate_noise_quality(noise),
            region_consistency=consistency,
            inconsistent_regions=inconsistent_regions if inconsistent_regions else None,
            processing_time_ms=elapsed_ms
        )

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def create_fingerprint(
        self,
        image_paths: List[str],
        camera_id: str
    ) -> Optional[PRNUFingerprint]:
        """
        Create a PRNU fingerprint from reference images.

        Requires multiple images from the same camera for reliable fingerprint.
        Recommended: 50+ images with varied content.

        OPTIMIZE: C - Fingerprint creation from many images
        """
        if not HAS_NUMPY or not HAS_PIL or not HAS_SCIPY:
            return None

        if len(image_paths) < 5:
            return None  # Need minimum images

        # Extract and average noise residuals
        # OPTIMIZE: C - Parallel processing of images
        noise_sum = None
        count = 0
        width = height = 0

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image_array = np.array(img, dtype=np.float64)

                    if noise_sum is None:
                        height, width = image_array.shape[:2]
                        noise_sum = np.zeros((height, width), dtype=np.float64)

                    # Check size matches
                    if image_array.shape[:2] != (height, width):
                        continue

                    noise = self._extract_noise_residual(image_array)
                    noise_sum += noise
                    count += 1

            except Exception:
                continue

        if count < 5:
            return None

        # Average noise is the fingerprint
        fingerprint = noise_sum / count

        # Estimate quality based on consistency
        quality = min(1.0, count / 50)  # More images = better quality

        return PRNUFingerprint(
            camera_id=camera_id,
            width=width,
            height=height,
            fingerprint=fingerprint,
            n_reference_images=count,
            quality_score=quality
        )


# =============================================================================
# NATIVE IMPLEMENTATION STUBS
# =============================================================================
# The following are stub signatures for C/Cython implementations.

"""
# cython: language_level=3
# prnu_native.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt, exp

# SIMD-accelerated Wiener filter
cpdef np.ndarray[np.float64_t, ndim=2] wiener_filter_fast(
    np.ndarray[np.float64_t, ndim=2] image,
    int window_size
):
    '''
    Fast Wiener filter implementation.

    Should use:
    - Integral images for local mean/variance
    - SIMD for parallel pixel processing
    - Cache-blocking for memory efficiency
    '''
    pass

# FFT-based cross-correlation with SIMD
cpdef tuple compute_pce_fast(
    np.ndarray[np.float64_t, ndim=2] noise,
    np.ndarray[np.float64_t, ndim=2] fingerprint
):
    '''
    Fast PCE computation.

    Should use:
    - FFTW for FFT (much faster than numpy)
    - SIMD for element-wise operations
    - In-place transforms to reduce memory
    '''
    pass
"""

"""
// prnu_native.c - Pure C implementation

#include <fftw3.h>
#include <immintrin.h>
#include <omp.h>

// Fast Wiener denoising using integral images
void wiener_denoise_fast(
    const double* input,
    double* output,
    int width, int height,
    int window_size
) {
    // Compute integral image for mean
    double* integral = malloc(width * height * sizeof(double));
    double* integral_sq = malloc(width * height * sizeof(double));

    // Parallel integral image computation
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        // ... implementation
    }

    // Wiener filter using precomputed integrals
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // O(1) local mean/variance from integrals
            // AVX2 for parallel pixel processing
        }
    }

    free(integral);
    free(integral_sq);
}

// SIMD-accelerated cross-correlation via FFT
typedef struct {
    double pce;
    double p_value;
} pce_result_t;

pce_result_t compute_pce_fftw(
    const double* noise,
    const double* fingerprint,
    int width, int height
) {
    // Use FFTW for much faster FFT than numpy
    fftw_complex* fft_noise = fftw_alloc_complex(width * height);
    fftw_complex* fft_fp = fftw_alloc_complex(width * height);

    fftw_plan plan_noise = fftw_plan_dft_r2c_2d(
        height, width, (double*)noise, fft_noise, FFTW_ESTIMATE
    );

    // ... FFTW operations with SIMD

    fftw_destroy_plan(plan_noise);
    fftw_free(fft_noise);
    fftw_free(fft_fp);

    return result;
}
"""
