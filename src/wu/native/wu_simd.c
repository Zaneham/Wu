/**
 * Wu Forensics - SIMD-optimized native functions
 *
 * Implementation using AVX2/SSE intrinsics for x86-64.
 * Falls back to scalar code when SIMD not available.
 */

#include "wu_simd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Detect architecture and include appropriate intrinsics */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define WU_X86 1
    #include <immintrin.h>
    #ifdef _MSC_VER
        #include <intrin.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define WU_ARM64 1
    #include <arm_neon.h>
#endif

/* Runtime SIMD detection */
static int g_simd_caps = -1;

static void detect_simd_caps(void) {
    if (g_simd_caps >= 0) return;
    g_simd_caps = 0;

#ifdef WU_X86
    #ifdef _MSC_VER
        int cpuinfo[4];
        __cpuid(cpuinfo, 1);
        if (cpuinfo[3] & (1 << 26)) g_simd_caps |= 1;  /* SSE2 */
        if (cpuinfo[2] & (1 << 28)) g_simd_caps |= 2;  /* AVX */
        __cpuidex(cpuinfo, 7, 0);
        if (cpuinfo[1] & (1 << 5)) g_simd_caps |= 4;   /* AVX2 */
    #else
        __builtin_cpu_init();
        if (__builtin_cpu_supports("sse2")) g_simd_caps |= 1;
        if (__builtin_cpu_supports("avx")) g_simd_caps |= 2;
        if (__builtin_cpu_supports("avx2")) g_simd_caps |= 4;
    #endif
#endif

#ifdef WU_ARM64
    g_simd_caps |= 16;  /* NEON always available on ARM64 */
#endif
}

WU_EXPORT int wu_get_simd_caps(void) {
    detect_simd_caps();
    return g_simd_caps;
}

/* ============================================================================
 * DOT PRODUCT - Critical for block similarity comparison
 * ============================================================================ */

#ifdef WU_X86
/* AVX2 implementation - processes 8 floats at a time */
static double dot_product_f32_avx2(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    /* Main loop: 8 floats per iteration */
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);  /* sum += va * vb */
    }

    /* Horizontal sum of 8 floats */
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    double result = _mm_cvtss_f32(sum128);

    /* Handle remainder */
    for (; i < n; i++) {
        result += (double)a[i] * (double)b[i];
    }

    return result;
}

/* SSE2 implementation - processes 4 floats at a time */
static double dot_product_f32_sse2(const float* a, const float* b, size_t n) {
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    /* Horizontal sum */
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    double result = _mm_cvtss_f32(sum);

    for (; i < n; i++) {
        result += (double)a[i] * (double)b[i];
    }

    return result;
}

/* AVX implementation for doubles - 4 at a time */
static double dot_product_f64_avx(const double* a, const double* b, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }

    /* Horizontal sum */
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d sum128 = _mm_add_pd(hi, lo);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double result = _mm_cvtsd_f64(sum128);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
#endif

#ifdef WU_ARM64
/* NEON implementation for ARM64 */
static double dot_product_f32_neon(const float* a, const float* b, size_t n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }

    float result = vaddvq_f32(sum);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return (double)result;
}
#endif

/* Scalar fallback */
static double dot_product_f32_scalar(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

static double dot_product_f64_scalar(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

WU_EXPORT double wu_dot_product_f32(const float* a, const float* b, size_t n) {
    detect_simd_caps();

#ifdef WU_X86
    if (g_simd_caps & 4) return dot_product_f32_avx2(a, b, n);
    if (g_simd_caps & 1) return dot_product_f32_sse2(a, b, n);
#endif
#ifdef WU_ARM64
    if (g_simd_caps & 16) return dot_product_f32_neon(a, b, n);
#endif

    return dot_product_f32_scalar(a, b, n);
}

WU_EXPORT double wu_dot_product_f64(const double* a, const double* b, size_t n) {
    detect_simd_caps();

#ifdef WU_X86
    if (g_simd_caps & 4) return dot_product_f64_avx(a, b, n);
#endif

    return dot_product_f64_scalar(a, b, n);
}

/* ============================================================================
 * EUCLIDEAN DISTANCE
 * ============================================================================ */

WU_EXPORT double wu_euclidean_distance_f32(const float* a, const float* b, size_t n) {
    detect_simd_caps();

#ifdef WU_X86
    if (g_simd_caps & 4) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        double result = _mm_cvtss_f32(sum128);

        for (; i < n; i++) {
            double diff = a[i] - b[i];
            result += diff * diff;
        }

        return sqrt(result);
    }
#endif

    /* Scalar fallback */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* ============================================================================
 * NORMALIZE
 * ============================================================================ */

WU_EXPORT double wu_normalize_f32(float* arr, size_t n) {
    double norm = sqrt(wu_dot_product_f32(arr, arr, n));
    if (norm < 1e-10) return 0.0;

    float inv_norm = (float)(1.0 / norm);

#ifdef WU_X86
    detect_simd_caps();
    if (g_simd_caps & 4) {
        __m256 scale = _mm256_set1_ps(inv_norm);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(arr + i);
            v = _mm256_mul_ps(v, scale);
            _mm256_storeu_ps(arr + i, v);
        }
        for (; i < n; i++) {
            arr[i] *= inv_norm;
        }
        return norm;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        arr[i] *= inv_norm;
    }
    return norm;
}

WU_EXPORT double wu_normalize_f64(double* arr, size_t n) {
    double norm = sqrt(wu_dot_product_f64(arr, arr, n));
    if (norm < 1e-10) return 0.0;

    double inv_norm = 1.0 / norm;

#ifdef WU_X86
    detect_simd_caps();
    if (g_simd_caps & 4) {
        __m256d scale = _mm256_set1_pd(inv_norm);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(arr + i);
            v = _mm256_mul_pd(v, scale);
            _mm256_storeu_pd(arr + i, v);
        }
        for (; i < n; i++) {
            arr[i] *= inv_norm;
        }
        return norm;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        arr[i] *= inv_norm;
    }
    return norm;
}

/* ============================================================================
 * VARIANCE
 * ============================================================================ */

WU_EXPORT double wu_variance_f64(const double* arr, size_t n) {
    if (n == 0) return 0.0;

    /* Two-pass algorithm for numerical stability */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += arr[i];
    }
    mean /= n;

    double var = 0.0;

#ifdef WU_X86
    detect_simd_caps();
    if (g_simd_caps & 4) {
        __m256d vmean = _mm256_set1_pd(mean);
        __m256d sum = _mm256_setzero_pd();
        size_t i = 0;

        for (; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(arr + i);
            __m256d diff = _mm256_sub_pd(v, vmean);
            sum = _mm256_fmadd_pd(diff, diff, sum);
        }

        __m128d hi = _mm256_extractf128_pd(sum, 1);
        __m128d lo = _mm256_castpd256_pd128(sum);
        __m128d sum128 = _mm_add_pd(hi, lo);
        sum128 = _mm_hadd_pd(sum128, sum128);
        var = _mm_cvtsd_f64(sum128);

        for (; i < n; i++) {
            double diff = arr[i] - mean;
            var += diff * diff;
        }

        return var / n;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        double diff = arr[i] - mean;
        var += diff * diff;
    }
    return var / n;
}

/* ============================================================================
 * SOBEL 3x3 - Gradient computation for lighting analysis
 * ============================================================================ */

WU_EXPORT void wu_sobel_3x3(
    const double* input,
    double* gx,
    double* gy,
    int width,
    int height
) {
    /* Sobel kernels (pre-divided by 8 for normalization) */
    const double kx[9] = {
        -0.125, 0, 0.125,
        -0.250, 0, 0.250,
        -0.125, 0, 0.125
    };
    const double ky[9] = {
        -0.125, -0.250, -0.125,
         0,      0,      0,
         0.125,  0.250,  0.125
    };

    /* Clear borders */
    memset(gx, 0, width * sizeof(double));
    memset(gy, 0, width * sizeof(double));
    memset(gx + (height - 1) * width, 0, width * sizeof(double));
    memset(gy + (height - 1) * width, 0, width * sizeof(double));

    for (int y = 0; y < height; y++) {
        gx[y * width] = 0;
        gx[y * width + width - 1] = 0;
        gy[y * width] = 0;
        gy[y * width + width - 1] = 0;
    }

    /* Main convolution loop */
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double sum_x = 0.0, sum_y = 0.0;

            /* 3x3 convolution */
            for (int ky_idx = 0; ky_idx < 3; ky_idx++) {
                for (int kx_idx = 0; kx_idx < 3; kx_idx++) {
                    double pixel = input[(y + ky_idx - 1) * width + (x + kx_idx - 1)];
                    int k_idx = ky_idx * 3 + kx_idx;
                    sum_x += pixel * kx[k_idx];
                    sum_y += pixel * ky[k_idx];
                }
            }

            gx[y * width + x] = sum_x;
            gy[y * width + x] = sum_y;
        }
    }
}

/* ============================================================================
 * DCT 8x8 - JPEG block analysis
 * ============================================================================ */

/* Precomputed DCT-II coefficients for 8x8 */
static const double DCT_COEFF[8][8] = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545, -0.097545, -0.277785, -0.415735, -0.490393},
    {0.461940, 0.191342, -0.191342, -0.461940, -0.461940, -0.191342, 0.191342, 0.461940},
    {0.415735, -0.097545, -0.490393, -0.277785, 0.277785, 0.490393, 0.097545, -0.415735},
    {0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553, 0.353553},
    {0.277785, -0.490393, 0.097545, 0.415735, -0.415735, -0.097545, 0.490393, -0.277785},
    {0.191342, -0.461940, 0.461940, -0.191342, -0.191342, 0.461940, -0.461940, 0.191342},
    {0.097545, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785, -0.097545}
};

WU_EXPORT void wu_dct_8x8(const float* input, float* output) {
    double temp[64];

    /* Row transform */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double sum = 0.0;
            for (int k = 0; k < 8; k++) {
                sum += input[i * 8 + k] * DCT_COEFF[j][k];
            }
            temp[i * 8 + j] = sum;
        }
    }

    /* Column transform */
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            double sum = 0.0;
            for (int k = 0; k < 8; k++) {
                sum += temp[k * 8 + j] * DCT_COEFF[i][k];
            }
            output[i * 8 + j] = (float)sum;
        }
    }
}

/* ============================================================================
 * BLOCKINESS - JPEG grid detection
 * ============================================================================ */

WU_EXPORT double wu_compute_blockiness(
    const double* image,
    int width,
    int height,
    int x_offset,
    int y_offset,
    int block_size
) {
    double total_diff = 0.0;
    int count = 0;

    /* Vertical boundaries */
    for (int x = x_offset; x < width - 1; x += block_size) {
        for (int y = 0; y < height; y++) {
            double diff = image[y * width + x] - image[y * width + x + 1];
            total_diff += diff * diff;
            count++;
        }
    }

    /* Horizontal boundaries */
    for (int y = y_offset; y < height - 1; y += block_size) {
        for (int x = 0; x < width; x++) {
            double diff = image[y * width + x] - image[(y + 1) * width + x];
            total_diff += diff * diff;
            count++;
        }
    }

    return count > 0 ? total_diff / count : 0.0;
}

/* ============================================================================
 * BLOCK EXTRACTION AND MATCHING - Copy-move detection
 * ============================================================================ */

WU_EXPORT int wu_extract_dct_blocks(
    const float* image,
    int width,
    int height,
    int block_size,
    int step,
    float* features,
    int* positions,
    int max_blocks
) {
    int n_blocks = 0;
    float* block = (float*)malloc(block_size * block_size * sizeof(float));
    float* dct_out = (float*)malloc(block_size * block_size * sizeof(float));

    for (int y = 0; y <= height - block_size && n_blocks < max_blocks; y += step) {
        for (int x = 0; x <= width - block_size && n_blocks < max_blocks; x += step) {
            /* Extract block */
            for (int by = 0; by < block_size; by++) {
                for (int bx = 0; bx < block_size; bx++) {
                    block[by * block_size + bx] = image[(y + by) * width + (x + bx)];
                }
            }

            /* Compute variance to skip uniform blocks */
            double mean = 0.0;
            for (int i = 0; i < block_size * block_size; i++) {
                mean += block[i];
            }
            mean /= (block_size * block_size);

            double var = 0.0;
            for (int i = 0; i < block_size * block_size; i++) {
                double diff = block[i] - mean;
                var += diff * diff;
            }
            var /= (block_size * block_size);

            if (var < 100.0) continue;  /* Skip uniform blocks */

            /* For 8x8 blocks, use DCT; otherwise use raw pixels as features */
            if (block_size == 8) {
                wu_dct_8x8(block, dct_out);
                /* Extract first 16 coefficients */
                for (int i = 0; i < 16 && i < block_size * block_size; i++) {
                    features[n_blocks * 16 + i] = dct_out[i];
                }
            } else {
                /* Use first 16 pixels as features */
                for (int i = 0; i < 16; i++) {
                    features[n_blocks * 16 + i] = block[i];
                }
            }

            positions[n_blocks * 2] = x;
            positions[n_blocks * 2 + 1] = y;
            n_blocks++;
        }
    }

    free(block);
    free(dct_out);
    return n_blocks;
}

WU_EXPORT int wu_find_similar_blocks(
    const float* features,
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    const int* positions,
    float* matches,
    int max_matches
) {
    int n_matches = 0;
    float min_dist_sq = min_distance * min_distance;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < n_blocks && n_matches < max_matches; i++) {
        const float* fi = features + i * n_features;
        int xi = positions[i * 2];
        int yi = positions[i * 2 + 1];

        for (int j = i + 1; j < n_blocks && n_matches < max_matches; j++) {
            int xj = positions[j * 2];
            int yj = positions[j * 2 + 1];

            /* Check spatial distance first (cheaper) */
            float dx = (float)(xj - xi);
            float dy = (float)(yj - yi);
            if (dx * dx + dy * dy < min_dist_sq) continue;

            /* Compute similarity (dot product of normalized vectors) */
            const float* fj = features + j * n_features;
            double sim = wu_dot_product_f32(fi, fj, n_features);

            if (sim > threshold) {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    if (n_matches < max_matches) {
                        matches[n_matches * 3] = (float)i;
                        matches[n_matches * 3 + 1] = (float)j;
                        matches[n_matches * 3 + 2] = (float)sim;
                        n_matches++;
                    }
                }
            }
        }
    }

    return n_matches;
}
