/**
 * Wu Forensics - SIMD-optimized native functions
 *
 * High-performance implementations of computationally intensive
 * forensic algorithms using AVX2/SSE intrinsics.
 *
 * Compilation:
 *   Windows (MSVC): cl /O2 /arch:AVX2 /LD wu_simd.c /Fe:wu_simd.dll
 *   Windows (GCC):  gcc -O3 -mavx2 -shared -o wu_simd.dll wu_simd.c
 *   Linux:          gcc -O3 -mavx2 -shared -fPIC -o libwu_simd.so wu_simd.c
 *   macOS (x86):    gcc -O3 -mavx2 -shared -o libwu_simd.dylib wu_simd.c
 *   macOS (ARM):    gcc -O3 -shared -o libwu_simd.dylib wu_simd.c (uses NEON)
 */

#ifndef WU_SIMD_H
#define WU_SIMD_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #define WU_EXPORT __declspec(dllexport)
#else
    #define WU_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute dot product of two float arrays using SIMD.
 * Used for block similarity in copy-move detection.
 *
 * @param a First array (must be 32-byte aligned for best performance)
 * @param b Second array (must be 32-byte aligned for best performance)
 * @param n Length of arrays
 * @return Dot product sum(a[i] * b[i])
 */
WU_EXPORT double wu_dot_product_f32(const float* a, const float* b, size_t n);

/**
 * Compute dot product of two double arrays using SIMD.
 *
 * @param a First array
 * @param b Second array
 * @param n Length of arrays
 * @return Dot product
 */
WU_EXPORT double wu_dot_product_f64(const double* a, const double* b, size_t n);

/**
 * Compute Euclidean distance between two float arrays.
 *
 * @param a First array
 * @param b Second array
 * @param n Length of arrays
 * @return sqrt(sum((a[i] - b[i])^2))
 */
WU_EXPORT double wu_euclidean_distance_f32(const float* a, const float* b, size_t n);

/**
 * Normalize array to unit length in-place.
 *
 * @param arr Array to normalize
 * @param n Length of array
 * @return Original norm (length) of array
 */
WU_EXPORT double wu_normalize_f32(float* arr, size_t n);
WU_EXPORT double wu_normalize_f64(double* arr, size_t n);

/**
 * Compute variance of array.
 *
 * @param arr Input array
 * @param n Length of array
 * @return Variance
 */
WU_EXPORT double wu_variance_f64(const double* arr, size_t n);

/**
 * Apply 3x3 Sobel filter for gradient computation.
 * Used in lighting analysis.
 *
 * @param input Input image (row-major, grayscale float64)
 * @param gx Output horizontal gradient (same size as input)
 * @param gy Output vertical gradient (same size as input)
 * @param width Image width
 * @param height Image height
 */
WU_EXPORT void wu_sobel_3x3(
    const double* input,
    double* gx,
    double* gy,
    int width,
    int height
);

/**
 * Compute 8x8 DCT-II (used in JPEG block analysis).
 *
 * @param input 8x8 block (64 floats, row-major)
 * @param output 8x8 DCT coefficients (64 floats, row-major)
 */
WU_EXPORT void wu_dct_8x8(const float* input, float* output);

/**
 * Extract DCT features from multiple blocks in parallel.
 *
 * @param image Grayscale image (row-major float32)
 * @param width Image width
 * @param height Image height
 * @param block_size Block size (typically 8 or 16)
 * @param step Step between blocks
 * @param features Output feature array (pre-allocated)
 * @param positions Output position array (pre-allocated, x,y pairs)
 * @param max_blocks Maximum blocks to extract
 * @return Number of blocks extracted
 */
WU_EXPORT int wu_extract_dct_blocks(
    const float* image,
    int width,
    int height,
    int block_size,
    int step,
    float* features,
    int* positions,
    int max_blocks
);

/**
 * Find similar block pairs using SIMD-accelerated comparison.
 *
 * @param features Normalized feature vectors (n_blocks x n_features)
 * @param n_blocks Number of blocks
 * @param n_features Features per block
 * @param threshold Similarity threshold (0-1)
 * @param min_distance Minimum spatial distance between matches
 * @param positions Block positions (x,y pairs)
 * @param matches Output match pairs (i,j,similarity triples)
 * @param max_matches Maximum matches to return
 * @return Number of matches found
 */
WU_EXPORT int wu_find_similar_blocks(
    const float* features,
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    const int* positions,
    float* matches,
    int max_matches
);

/**
 * Compute blockiness measure for JPEG grid detection.
 *
 * @param image Grayscale image (row-major float64)
 * @param width Image width
 * @param height Image height
 * @param x_offset Grid X offset (0-7)
 * @param y_offset Grid Y offset (0-7)
 * @param block_size Block size (typically 8)
 * @return Blockiness score
 */
WU_EXPORT double wu_compute_blockiness(
    const double* image,
    int width,
    int height,
    int x_offset,
    int y_offset,
    int block_size
);

/**
 * Get SIMD capability info.
 *
 * @return Bitmask: 1=SSE2, 2=AVX, 4=AVX2, 8=AVX512, 16=NEON
 */
WU_EXPORT int wu_get_simd_caps(void);

#ifdef __cplusplus
}
#endif

#endif /* WU_SIMD_H */
