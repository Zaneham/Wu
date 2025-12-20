/**
 * Wu Forensics - Assembly Wrapper Functions Header
 *
 * Declarations for assembly-accelerated functions with C fallbacks.
 */

#ifndef WU_ASM_H
#define WU_ASM_H

#include "wu_simd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Similarity matching (proprietary algorithm) */
WU_EXPORT float wu_similarity_match_asm(const float* a, const float* b, int n);

/* Batch similarity search */
WU_EXPORT int wu_find_similar_blocks_asm(
    const float* features,
    const int* positions,
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    float* matches,
    int max_matches
);

/* Correlation sum */
WU_EXPORT double wu_correlation_sum_asm(const double* a, const double* b, size_t n);

/* Mean and variance (single pass) */
WU_EXPORT void wu_mean_variance_asm(const double* data, size_t n, double* mean, double* var);

/* Find peak value and index */
WU_EXPORT double wu_find_peak_asm(const double* data, size_t n, size_t* idx);

/* Gradient magnitude array */
WU_EXPORT void wu_gradient_magnitude_asm(const double* gx, const double* gy, size_t n, double* mag);

/* Weighted gradient statistics */
WU_EXPORT void wu_weighted_gradient_stats_asm(
    const double* gx, const double* gy, size_t n,
    double* sum_gx, double* sum_gy, double* sum_w
);

/* Blockiness for all 64 offsets - returns best index */
WU_EXPORT int wu_blockiness_all_offsets_asm(
    const double* image, int width, int height, int block_size, double* scores
);

#ifdef __cplusplus
}
#endif

#endif /* WU_ASM_H */
