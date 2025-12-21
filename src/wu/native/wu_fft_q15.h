/**
 * Wu Forensics - Fixed-Point FFT (Q15)
 *
 * Deterministic FFT implementation using Q15 fixed-point arithmetic.
 * Produces bitwise-identical results across all platforms.
 *
 * Q15 format: 16-bit signed integer representing -1.0 to +0.999969
 *             value = integer / 32768
 */

#ifndef WU_FFT_Q15_H
#define WU_FFT_Q15_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed-point types */
typedef int16_t q15_t;
typedef int32_t q31_t;

/* Q15 constants */
#define Q15_ONE       32767
#define Q15_MINUS_ONE (-32768)
#define Q15_HALF      16384

/**
 * In-place radix-2 DIT FFT (Q15)
 *
 * @param real   Real part array (modified in-place)
 * @param imag   Imaginary part array (modified in-place)
 * @param n      FFT size (must be power of 2, max 2048)
 *
 * Note: Output is scaled by 1/n to prevent overflow.
 */
void wu_fft_q15(q15_t* real, q15_t* imag, size_t n);

/**
 * Inverse FFT (Q15)
 */
void wu_ifft_q15(q15_t* real, q15_t* imag, size_t n);

/**
 * Compute magnitude spectrum from complex FFT output.
 *
 * Uses integer approximation of sqrt(re² + im²) to maintain
 * determinism across platforms.
 *
 * @param real      Real part array
 * @param imag      Imaginary part array
 * @param magnitude Output magnitude array (Q15)
 * @param n         Array length
 */
void wu_fft_magnitude_q15(
    const q15_t* real,
    const q15_t* imag,
    q15_t* magnitude,
    size_t n
);

/**
 * Apply Hamming window (Q15 coefficients, precomputed).
 *
 * Window coefficients are stored as compile-time constants
 * to ensure identical behaviour across platforms.
 *
 * @param samples   Input/output samples (modified in-place)
 * @param n         Window size (must be 256 or 512)
 */
void wu_apply_hamming_q15(q15_t* samples, size_t n);

/**
 * Cross-correlation of two Q15 sequences.
 * Returns the lag (in samples) of maximum correlation.
 *
 * Used for temporal alignment between audio and video streams.
 *
 * @param a         First sequence
 * @param b         Second sequence
 * @param n         Sequence length
 * @param max_lag   Maximum lag to search (+/-)
 * @param corr_out  Output: correlation value at best lag (Q15)
 * @return          Lag in samples (positive = a leads b)
 */
int32_t wu_xcorr_q15(
    const q15_t* a,
    const q15_t* b,
    size_t n,
    size_t max_lag,
    q15_t* corr_out
);

/**
 * 3-tap median filter for spectrum smoothing.
 *
 * Removes spurious peaks whilst preserving formant structure.
 *
 * @param in    Input array
 * @param out   Output array
 * @param n     Array length
 */
void wu_median3_q15(const q15_t* in, q15_t* out, size_t n);

/**
 * Find local maxima (peaks) in spectrum.
 *
 * Used for formant extraction - identifies F1/F2 frequencies.
 *
 * @param data       Input spectrum
 * @param n          Spectrum length
 * @param min_idx    Start index for search
 * @param max_idx    End index for search
 * @param threshold  Minimum peak height (Q15)
 * @param peaks_out  Output array of peak indices
 * @param max_peaks  Maximum peaks to return
 * @return           Number of peaks found
 */
size_t wu_find_peaks_q15(
    const q15_t* data,
    size_t n,
    size_t min_idx,
    size_t max_idx,
    q15_t threshold,
    size_t* peaks_out,
    size_t max_peaks
);

#ifdef __cplusplus
}
#endif

#endif /* WU_FFT_Q15_H */
