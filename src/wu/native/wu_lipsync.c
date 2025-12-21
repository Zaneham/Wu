/**
 * Wu Forensics - Lip-Sync Analysis Implementation
 *
 * Deterministic audio-visual synchronisation detection.
 * All arithmetic is integer-only for reproducibility.
 */

#include "wu_lipsync.h"
#include "wu_fft_q15.h"
#include <string.h>

/* ============================================================================
 * PHONEME-VISEME COMPATIBILITY MATRIX
 *
 * Defines which mouth shapes (visemes) are compatible with which
 * sound categories (phonemes). Used for mismatch detection.
 * ============================================================================ */

static const uint8_t COMPAT_MATRIX[7][5] = {
    /*                CLOSED  NARROW  MEDIUM  WIDE  ROUNDED */
    /* SILENCE */   {   1,      1,      0,      0,     0    },
    /* OPEN */      {   0,      0,      1,      1,     0    },
    /* CL_FRONT */  {   0,      1,      1,      0,     0    },
    /* CL_BACK */   {   0,      1,      0,      0,     1    },
    /* MID */       {   0,      0,      1,      1,     1    },
    /* BILABIAL */  {   1,      0,      0,      0,     0    },
    /* OTHER */     {   1,      1,      1,      0,     0    }
};

/* ============================================================================
 * FORMANT EXTRACTION
 * ============================================================================ */

/**
 * Compute frame energy from magnitude spectrum.
 */
static q15_t compute_energy(const q15_t* magnitude, size_t n_bins) {
    q31_t sum = 0;
    for (size_t i = 0; i < n_bins && i < 128; i++) {
        sum += (q31_t)magnitude[i];
    }
    /* Normalise and clamp to Q15 */
    sum = sum / 64;
    if (sum > 32767) return 32767;
    return (q15_t)sum;
}

/**
 * Find strongest peak in a frequency range.
 */
static uint16_t find_formant_peak(
    const q15_t* magnitude,
    size_t min_bin,
    size_t max_bin,
    uint32_t sample_rate,
    size_t fft_size
) {
    q15_t max_val = 0;
    size_t max_idx = min_bin;

    for (size_t i = min_bin; i < max_bin; i++) {
        if (magnitude[i] > max_val) {
            max_val = magnitude[i];
            max_idx = i;
        }
    }

    /* Convert bin index to frequency */
    /* freq = bin * sample_rate / fft_size */
    return (uint16_t)((max_idx * sample_rate) / fft_size);
}

wu_formants_t wu_extract_formants(
    const q15_t* magnitude,
    size_t n_bins,
    uint32_t sample_rate
) {
    wu_formants_t result = {0};

    /* Compute frame energy */
    result.energy = compute_energy(magnitude, n_bins);

    /* Check if voiced (energy above threshold) */
    if (result.energy < WU_VOICED_THRESHOLD) {
        result.voiced = 0;
        return result;
    }
    result.voiced = 1;

    /* Smooth spectrum with median filter */
    q15_t smoothed[256];
    if (n_bins > 256) n_bins = 256;
    wu_median3_q15(magnitude, smoothed, n_bins);

    /* Find F1 (first formant) in 200-1000 Hz range */
    result.f1_hz = find_formant_peak(
        smoothed,
        WU_F1_MIN_BIN,
        WU_F1_MAX_BIN,
        sample_rate,
        WU_FFT_SIZE
    );

    /* Find F2 (second formant) in 800-2500 Hz range */
    result.f2_hz = find_formant_peak(
        smoothed,
        WU_F2_MIN_BIN,
        WU_F2_MAX_BIN,
        sample_rate,
        WU_FFT_SIZE
    );

    return result;
}

/* ============================================================================
 * PHONEME CLASSIFICATION
 *
 * Rule-based classification using F1/F2 formant frequencies.
 * Based on Peterson & Barney (1952) vowel formant data.
 * ============================================================================ */

wu_phoneme_class_t wu_classify_phoneme(const wu_formants_t* formants) {
    if (!formants->voiced) {
        return WU_PHON_SILENCE;
    }

    uint16_t f1 = formants->f1_hz;
    uint16_t f2 = formants->f2_hz;

    /* Bilabial detection: very low energy burst pattern
     * (simplified - real detection would look at temporal pattern) */
    if (formants->energy < WU_VOICED_THRESHOLD * 2 && f1 < 400) {
        return WU_PHON_BILABIAL;
    }

    /* Vowel classification based on F1/F2 space */

    /* Open vowels: high F1 (600-1000 Hz) */
    if (f1 >= 600) {
        return WU_PHON_OPEN;
    }

    /* Close front vowels: low F1, high F2 */
    if (f1 < 400 && f2 > 2000) {
        return WU_PHON_CLOSE_FRONT;
    }

    /* Close back vowels: low F1, low F2 */
    if (f1 < 400 && f2 < 1200) {
        return WU_PHON_CLOSE_BACK;
    }

    /* Mid vowels: everything else in the vowel space */
    if (f1 >= 400 && f1 < 600) {
        return WU_PHON_MID;
    }

    /* Default to other consonant */
    return WU_PHON_OTHER;
}

/* ============================================================================
 * AUDIO PROCESSING PIPELINE
 * ============================================================================ */

size_t wu_audio_to_phonemes(
    const int16_t* samples,
    size_t n_samples,
    wu_phoneme_class_t* phonemes_out,
    uint64_t* times_out,
    size_t max_frames
) {
    if (n_samples < WU_FFT_SIZE) return 0;

    size_t frame_count = 0;
    size_t offset = 0;

    /* Working buffers */
    q15_t real[WU_FFT_SIZE];
    q15_t imag[WU_FFT_SIZE];
    q15_t magnitude[WU_FFT_SIZE / 2];

    while (offset + WU_FFT_SIZE <= n_samples && frame_count < max_frames) {
        /* Copy samples to real buffer (already Q15 compatible) */
        for (size_t i = 0; i < WU_FFT_SIZE; i++) {
            real[i] = samples[offset + i];
            imag[i] = 0;
        }

        /* Apply window */
        wu_apply_hamming_q15(real, WU_FFT_SIZE);

        /* Compute FFT */
        wu_fft_q15(real, imag, WU_FFT_SIZE);

        /* Compute magnitude spectrum */
        wu_fft_magnitude_q15(real, imag, magnitude, WU_FFT_SIZE / 2);

        /* Extract formants */
        wu_formants_t formants = wu_extract_formants(
            magnitude,
            WU_FFT_SIZE / 2,
            WU_AUDIO_SAMPLE_RATE
        );

        /* Classify phoneme */
        phonemes_out[frame_count] = wu_classify_phoneme(&formants);

        /* Compute timestamp (microseconds) */
        /* time_us = offset * 1000000 / sample_rate */
        times_out[frame_count] = (offset * 1000000ULL) / WU_AUDIO_SAMPLE_RATE;

        frame_count++;
        offset += WU_HOP_SIZE;
    }

    return frame_count;
}

/* ============================================================================
 * LIP DETECTION (COLOUR-BASED)
 *
 * Uses YCbCr colour space thresholds for lip segmentation.
 * Based on empirical lip colour ranges from literature.
 * ============================================================================ */

/**
 * Convert RGB to YCbCr (integer arithmetic).
 */
static void rgb_to_ycbcr(uint8_t r, uint8_t g, uint8_t b,
                         int16_t* y, int16_t* cb, int16_t* cr) {
    /* Y  =  0.299*R + 0.587*G + 0.114*B
     * Cb = -0.169*R - 0.331*G + 0.500*B + 128
     * Cr =  0.500*R - 0.419*G - 0.081*B + 128
     *
     * Using fixed-point: multiply by 256, then shift right 8
     */
    *y  = (int16_t)(( 77 * r + 150 * g +  29 * b) >> 8);
    *cb = (int16_t)(((-43 * r -  85 * g + 128 * b) >> 8) + 128);
    *cr = (int16_t)(((128 * r - 107 * g -  21 * b) >> 8) + 128);
}

/**
 * Check if a pixel is likely lip colour.
 *
 * Thresholds based on:
 * - Cr typically higher for lips (red component)
 * - Specific Cb/Cr ratio ranges
 */
static int is_lip_pixel(int16_t y, int16_t cb, int16_t cr) {
    /* Lip detection criteria (empirically tuned) */
    /* Cr should be elevated (red-ish) */
    if (cr < 140 || cr > 180) return 0;

    /* Cb should be moderate */
    if (cb < 100 || cb > 130) return 0;

    /* Luminance should be reasonable (not too dark/bright) */
    if (y < 50 || y > 200) return 0;

    /* Ratio check: Cr/Cb should be > 1 for lips */
    if (cr <= cb) return 0;

    return 1;
}

wu_lip_region_t wu_detect_lips_color(
    const uint8_t* frame_rgb,
    uint32_t width,
    uint32_t height,
    uint32_t face_x,
    uint32_t face_y,
    uint32_t face_w,
    uint32_t face_h
) {
    wu_lip_region_t result = {0};
    result.face_height = (uint16_t)face_h;

    /* Focus on lower third of face (where mouth is) */
    uint32_t mouth_y = face_y + (face_h * 2) / 3;
    uint32_t mouth_h = face_h / 3;
    uint32_t mouth_x = face_x + face_w / 4;
    uint32_t mouth_w = face_w / 2;

    /* Bounds check */
    if (mouth_x + mouth_w > width) mouth_w = width - mouth_x;
    if (mouth_y + mouth_h > height) mouth_h = height - mouth_y;

    /* Scan for lip pixels */
    uint32_t min_x = mouth_x + mouth_w;
    uint32_t max_x = mouth_x;
    uint32_t min_y = mouth_y + mouth_h;
    uint32_t max_y = mouth_y;
    uint32_t lip_pixel_count = 0;

    for (uint32_t y = mouth_y; y < mouth_y + mouth_h; y++) {
        for (uint32_t x = mouth_x; x < mouth_x + mouth_w; x++) {
            size_t idx = (y * width + x) * 3;
            uint8_t r = frame_rgb[idx];
            uint8_t g = frame_rgb[idx + 1];
            uint8_t b = frame_rgb[idx + 2];

            int16_t Y, cb, cr;
            rgb_to_ycbcr(r, g, b, &Y, &cb, &cr);

            if (is_lip_pixel(Y, cb, cr)) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
                lip_pixel_count++;
            }
        }
    }

    /* Check if we found enough lip pixels */
    if (lip_pixel_count < 50) {
        result.valid = 0;
        return result;
    }

    result.valid = 1;
    result.center_x = (uint16_t)((min_x + max_x) / 2);
    result.center_y = (uint16_t)((min_y + max_y) / 2);
    result.width = (uint16_t)(max_x - min_x + 1);
    result.height = (uint16_t)(max_y - min_y + 1);
    result.area = lip_pixel_count;

    return result;
}

/* ============================================================================
 * VISEME CLASSIFICATION
 * ============================================================================ */

wu_viseme_t wu_classify_viseme(const wu_lip_region_t* lips) {
    if (!lips->valid || lips->face_height == 0) {
        return WU_VIS_CLOSED;
    }

    /* Compute aperture ratio (lip height / face height) */
    /* Using fixed-point: multiply by 1000 for per-mille */
    uint32_t aperture_ratio = (lips->height * 1000) / lips->face_height;

    /* Compute aspect ratio (width / height) for roundedness */
    uint32_t aspect_ratio = (lips->width * 100) / (lips->height + 1);

    /* Classification thresholds (per-mille of face height) */
    if (aperture_ratio < 20) {          /* < 2% = closed */
        return WU_VIS_CLOSED;
    }

    if (aperture_ratio < 50) {          /* < 5% = narrow */
        /* Check for rounding (protruded lips) */
        if (aspect_ratio < 200) {       /* Width < 2x height */
            return WU_VIS_ROUNDED;
        }
        return WU_VIS_NARROW;
    }

    if (aperture_ratio < 100) {         /* < 10% = medium */
        if (aspect_ratio < 150) {
            return WU_VIS_ROUNDED;
        }
        return WU_VIS_MEDIUM;
    }

    /* > 10% = wide open */
    return WU_VIS_WIDE;
}

/* ============================================================================
 * SYNCHRONISATION ANALYSIS
 * ============================================================================ */

int wu_phoneme_viseme_compatible(wu_phoneme_class_t phoneme, wu_viseme_t viseme) {
    if (phoneme > WU_PHON_OTHER || viseme > WU_VIS_ROUNDED) {
        return 0;
    }
    return COMPAT_MATRIX[phoneme][viseme];
}

wu_sync_result_t wu_analyze_sync(
    const wu_phoneme_class_t* phonemes,
    const uint64_t* phon_times,
    size_t n_phonemes,
    const wu_viseme_t* visemes,
    const uint64_t* vis_times,
    size_t n_visemes
) {
    wu_sync_result_t result = {0};

    if (n_phonemes == 0 || n_visemes == 0) {
        return result;
    }

    /* Compute duration */
    uint64_t max_phon_time = phon_times[n_phonemes - 1];
    uint64_t max_vis_time = vis_times[n_visemes - 1];
    uint64_t duration = (max_phon_time > max_vis_time) ? max_phon_time : max_vis_time;
    result.analyzed_duration_ms = (uint32_t)(duration / 1000);

    /* Find optimal offset using cross-correlation of voiced segments */
    /* Convert to simple presence vectors for correlation */
    int32_t best_offset_us = 0;
    int32_t best_match_count = 0;

    /* Test offsets from -500ms to +500ms in 10ms steps */
    for (int32_t offset_us = -500000; offset_us <= 500000; offset_us += 10000) {
        int32_t match_count = 0;

        for (size_t pi = 0; pi < n_phonemes; pi++) {
            if (phonemes[pi] == WU_PHON_SILENCE) continue;

            int64_t target_time = (int64_t)phon_times[pi] + offset_us;
            if (target_time < 0) continue;

            /* Find nearest viseme */
            size_t nearest_vi = 0;
            int64_t min_diff = 0x7FFFFFFFFFFFFFFFLL;

            for (size_t vi = 0; vi < n_visemes; vi++) {
                int64_t diff = (int64_t)vis_times[vi] - target_time;
                if (diff < 0) diff = -diff;
                if (diff < min_diff) {
                    min_diff = diff;
                    nearest_vi = vi;
                }
            }

            /* Check compatibility (only if within 50ms) */
            if (min_diff < 50000) {
                if (wu_phoneme_viseme_compatible(phonemes[pi], visemes[nearest_vi])) {
                    match_count++;
                }
            }
        }

        if (match_count > best_match_count) {
            best_match_count = match_count;
            best_offset_us = offset_us;
        }
    }

    result.offset_ms = best_offset_us / 1000;

    /* Count mismatches at best offset */
    uint32_t voiced_count = 0;
    uint32_t mismatch_count = 0;

    for (size_t pi = 0; pi < n_phonemes; pi++) {
        if (phonemes[pi] == WU_PHON_SILENCE) continue;
        voiced_count++;

        int64_t target_time = (int64_t)phon_times[pi] + best_offset_us;
        if (target_time < 0) continue;

        /* Find nearest viseme */
        size_t nearest_vi = 0;
        int64_t min_diff = 0x7FFFFFFFFFFFFFFFLL;

        for (size_t vi = 0; vi < n_visemes; vi++) {
            int64_t diff = (int64_t)vis_times[vi] - target_time;
            if (diff < 0) diff = -diff;
            if (diff < min_diff) {
                min_diff = diff;
                nearest_vi = vi;
            }
        }

        if (min_diff < 50000) {
            if (!wu_phoneme_viseme_compatible(phonemes[pi], visemes[nearest_vi])) {
                mismatch_count++;
            }
        }
    }

    result.total_frames = voiced_count;
    result.mismatch_count = mismatch_count;

    /* Compute correlation score */
    if (voiced_count > 0) {
        int32_t match_rate = ((voiced_count - mismatch_count) * 32767) / voiced_count;
        result.correlation = (q15_t)match_rate;
    }

    return result;
}
