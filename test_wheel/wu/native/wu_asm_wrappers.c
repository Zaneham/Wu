/**
 * Wu Forensics - Assembly Wrapper Functions
 *
 * C wrappers for the x86-64 AVX2 assembly implementations.
 * Provides Python-callable interfaces with automatic fallback.
 */

#include "wu_simd.h"
#include <math.h>

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define WU_X86_64 1
#endif

/* ============================================================================
 * ASSEMBLY FUNCTION DECLARATIONS
 * Implemented in assembly/x86_64/*.asm for IP protection
 * ============================================================================ */

#ifdef WU_X86_64
/* From copymove.asm */
extern float wu_asm_similarity_match(const float* a, const float* b, size_t n);
extern int wu_asm_find_similar_blocks(
    const float* features,
    const int* positions,
    size_t n_blocks,
    size_t n_features,
    float threshold,
    float min_distance_sq,
    float* matches,
    int max_matches
);

/* From prnu.asm */
extern double wu_asm_correlation_sum(const double* a, const double* b, size_t n);
extern void wu_asm_mean_variance(const double* data, size_t n, double* mean, double* var);
extern double wu_asm_find_peak(const double* data, size_t n, size_t* idx);

/* From lighting.asm */
extern void wu_asm_gradient_magnitude(const double* gx, const double* gy, size_t n, double* mag);
extern void wu_asm_weighted_gradient_stats(const double* gx, const double* gy, size_t n,
                                           double* sum_gx, double* sum_gy, double* sum_w);
#endif

/* Get SIMD capabilities (from wu_simd.c) */
extern int wu_get_simd_caps(void);
extern double wu_dot_product_f32(const float* a, const float* b, size_t n);
extern double wu_dot_product_f64(const double* a, const double* b, size_t n);
extern double wu_compute_blockiness(const double* image, int width, int height,
                                    int x_offset, int y_offset, int block_size);

/* Fallbacks from wu_simd.c */
extern int wu_find_similar_blocks(
    const float* features,
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    const int* positions,
    float* matches,
    int max_matches
);


/* ============================================================================
 * C WRAPPER FUNCTIONS
 * ============================================================================ */

/**
 * Similarity matching using assembly (proprietary algorithm).
 */
WU_EXPORT float wu_similarity_match_asm(const float* a, const float* b, int n) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) { /* AVX2 available */
        return wu_asm_similarity_match(a, b, (size_t)n);
    }
#endif
    /* Fallback: cosine similarity */
    double dot = wu_dot_product_f32(a, b, (size_t)n);
    double norm_a = wu_dot_product_f32(a, a, (size_t)n);
    double norm_b = wu_dot_product_f32(b, b, (size_t)n);
    double denom = sqrt(norm_a * norm_b);
    return denom > 1e-10 ? (float)(dot / denom) : 0.0f;
}

/**
 * Batch similarity search using assembly.
 */
WU_EXPORT int wu_find_similar_blocks_asm(
    const float* features,
    const int* positions,
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    float* matches,
    int max_matches
) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        float min_dist_sq = min_distance * min_distance;
        return wu_asm_find_similar_blocks(
            features, positions, (size_t)n_blocks, (size_t)n_features,
            threshold, min_dist_sq, matches, max_matches
        );
    }
#endif
    return wu_find_similar_blocks(
        features, n_blocks, n_features, threshold, min_distance,
        positions, matches, max_matches
    );
}

/**
 * Correlation sum using assembly.
 */
WU_EXPORT double wu_correlation_sum_asm(const double* a, const double* b, size_t n) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        return wu_asm_correlation_sum(a, b, n);
    }
#endif
    return wu_dot_product_f64(a, b, n);
}

/**
 * Mean and variance in single pass using assembly.
 */
WU_EXPORT void wu_mean_variance_asm(const double* data, size_t n, double* mean, double* var) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        wu_asm_mean_variance(data, n, mean, var);
        return;
    }
#endif
    /* Fallback: two-pass computation */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += data[i];
    *mean = sum / (double)n;
    
    double var_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = data[i] - *mean;
        var_sum += d * d;
    }
    *var = var_sum / (double)n;
}

/**
 * Find peak value and index using assembly.
 */
WU_EXPORT double wu_find_peak_asm(const double* data, size_t n, size_t* idx) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        return wu_asm_find_peak(data, n, idx);
    }
#endif
    /* Fallback */
    double max_val = data[0];
    *idx = 0;
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            *idx = i;
        }
    }
    return max_val;
}

/**
 * Gradient magnitude array using assembly.
 */
WU_EXPORT void wu_gradient_magnitude_asm(const double* gx, const double* gy, size_t n, double* mag) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        wu_asm_gradient_magnitude(gx, gy, n, mag);
        return;
    }
#endif
    /* Fallback */
    for (size_t i = 0; i < n; i++) {
        mag[i] = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }
}

/**
 * Weighted gradient statistics using assembly.
 */
WU_EXPORT void wu_weighted_gradient_stats_asm(
    const double* gx, const double* gy, size_t n,
    double* sum_gx, double* sum_gy, double* sum_w
) {
#ifdef WU_X86_64
    int caps = wu_get_simd_caps();
    if (caps & 4) {
        wu_asm_weighted_gradient_stats(gx, gy, n, sum_gx, sum_gy, sum_w);
        return;
    }
#endif
    /* Fallback */
    *sum_gx = 0.0;
    *sum_gy = 0.0;
    *sum_w = 0.0;
    for (size_t i = 0; i < n; i++) {
        double mag = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        *sum_gx += gx[i] * mag;
        *sum_gy += gy[i] * mag;
        *sum_w += mag;
    }
}

/**
 * Compute blockiness for all 64 grid offsets.
 * Returns best offset index (y*8 + x).
 */
WU_EXPORT int wu_blockiness_all_offsets_asm(
    const double* image, int width, int height, int block_size, double* scores
) {
    double best_score = 1e308;
    int best_idx = 0;
    
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int idx = y * 8 + x;
            scores[idx] = wu_compute_blockiness(image, width, height, x, y, block_size);
            if (scores[idx] < best_score) {
                best_score = scores[idx];
                best_idx = idx;
            }
        }
    }
    
    return best_idx;
}
