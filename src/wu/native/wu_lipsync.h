/**
 * Wu Forensics - Lip-Sync Analysis (Deterministic)
 *
 * Audio-visual synchronisation detection using fixed-point arithmetic.
 * All operations are integer-only to guarantee reproducibility.
 */

#ifndef WU_LIPSYNC_H
#define WU_LIPSYNC_H

#include "wu_fft_q15.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define WU_AUDIO_SAMPLE_RATE    16000   /* Target sample rate (Hz) */
#define WU_FFT_SIZE             512     /* FFT frame size */
#define WU_HOP_SIZE             160     /* 10ms hop @ 16kHz */
#define WU_FRAME_MS             32      /* Frame duration (ms) */

/* Frequency bin indices for formant ranges @ 16kHz, 512-point FFT */
/* Bin = freq * FFT_SIZE / SAMPLE_RATE */
#define WU_F1_MIN_BIN           6       /* ~200 Hz */
#define WU_F1_MAX_BIN           32      /* ~1000 Hz */
#define WU_F2_MIN_BIN           25      /* ~800 Hz */
#define WU_F2_MAX_BIN           80      /* ~2500 Hz */

/* Energy threshold for voiced speech detection (Q15) */
#define WU_VOICED_THRESHOLD     1000

/* ============================================================================
 * PHONEME CLASSES
 *
 * Broad phoneme categories based on F1/F2 formant positions.
 * These map to viseme groups for synchronisation checking.
 * ============================================================================ */

typedef enum {
    WU_PHON_SILENCE = 0,        /* No speech energy */
    WU_PHON_OPEN,               /* Open vowels: /a/, /æ/, /ɑ/ */
    WU_PHON_CLOSE_FRONT,        /* Close front: /i/, /ɪ/ */
    WU_PHON_CLOSE_BACK,         /* Close back: /u/, /ʊ/ */
    WU_PHON_MID,                /* Mid vowels: /e/, /ə/, /o/ */
    WU_PHON_BILABIAL,           /* Bilabials: /p/, /b/, /m/ */
    WU_PHON_OTHER               /* Other consonants */
} wu_phoneme_class_t;

/* ============================================================================
 * VISEME CLASSES
 *
 * Visual mouth shapes corresponding to phoneme groups.
 * ============================================================================ */

typedef enum {
    WU_VIS_CLOSED = 0,          /* Lips together */
    WU_VIS_NARROW,              /* Small opening */
    WU_VIS_MEDIUM,              /* Medium opening */
    WU_VIS_WIDE,                /* Large opening */
    WU_VIS_ROUNDED              /* Lips protruded/rounded */
} wu_viseme_t;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Formant analysis result for a single audio frame.
 */
typedef struct {
    uint16_t f1_hz;             /* First formant frequency (Hz) */
    uint16_t f2_hz;             /* Second formant frequency (Hz) */
    q15_t    energy;            /* Frame energy (Q15) */
    uint8_t  voiced;            /* 1 = voiced speech, 0 = silence/unvoiced */
} wu_formants_t;

/**
 * Lip region measurements from a video frame.
 */
typedef struct {
    uint16_t center_x;          /* Lip region centre X */
    uint16_t center_y;          /* Lip region centre Y */
    uint16_t width;             /* Horizontal extent (pixels) */
    uint16_t height;            /* Vertical extent / aperture (pixels) */
    uint32_t area;              /* Lip pixel count */
    uint16_t face_height;       /* Reference face height for normalisation */
    uint8_t  valid;             /* 1 = detection successful */
} wu_lip_region_t;

/**
 * Synchronisation analysis result.
 */
typedef struct {
    int32_t  offset_ms;         /* Audio-video offset (positive = audio leads) */
    q15_t    correlation;       /* Peak correlation value (Q15) */
    uint32_t mismatch_count;    /* Frames with phoneme/viseme mismatch */
    uint32_t total_frames;      /* Total voiced frames analysed */
    uint32_t analyzed_duration_ms;  /* Duration of content analysed */
} wu_sync_result_t;

/* ============================================================================
 * AUDIO ANALYSIS FUNCTIONS
 * ============================================================================ */

/**
 * Extract formants from a magnitude spectrum.
 *
 * @param magnitude     FFT magnitude spectrum (Q15, length FFT_SIZE/2)
 * @param n_bins        Number of bins (typically 256)
 * @param sample_rate   Audio sample rate (typically 16000)
 * @return              Formant analysis result
 */
wu_formants_t wu_extract_formants(
    const q15_t* magnitude,
    size_t n_bins,
    uint32_t sample_rate
);

/**
 * Classify phoneme from formant values.
 *
 * Uses rule-based F1/F2 mapping - no machine learning.
 *
 * @param formants      Formant analysis from wu_extract_formants
 * @return              Phoneme class
 */
wu_phoneme_class_t wu_classify_phoneme(const wu_formants_t* formants);

/**
 * Process raw audio samples to extract phoneme sequence.
 *
 * @param samples       16-bit PCM audio (mono, 16kHz)
 * @param n_samples     Number of samples
 * @param phonemes_out  Output array for phoneme classes
 * @param times_out     Output array for timestamps (microseconds)
 * @param max_frames    Maximum frames to output
 * @return              Number of frames processed
 */
size_t wu_audio_to_phonemes(
    const int16_t* samples,
    size_t n_samples,
    wu_phoneme_class_t* phonemes_out,
    uint64_t* times_out,
    size_t max_frames
);

/* ============================================================================
 * VIDEO ANALYSIS FUNCTIONS
 * ============================================================================ */

/**
 * Detect lip region using colour segmentation (YCbCr).
 *
 * Requires a face bounding box as input. Uses deterministic
 * colour thresholds - no neural network.
 *
 * @param frame_rgb     RGB frame data (row-major, 3 bytes per pixel)
 * @param width         Frame width
 * @param height        Frame height
 * @param face_x        Face bounding box X
 * @param face_y        Face bounding box Y
 * @param face_w        Face bounding box width
 * @param face_h        Face bounding box height
 * @return              Lip region measurements
 */
wu_lip_region_t wu_detect_lips_color(
    const uint8_t* frame_rgb,
    uint32_t width,
    uint32_t height,
    uint32_t face_x,
    uint32_t face_y,
    uint32_t face_w,
    uint32_t face_h
);

/**
 * Classify viseme from lip measurements.
 *
 * @param lips          Lip region from wu_detect_lips_color
 * @return              Viseme class
 */
wu_viseme_t wu_classify_viseme(const wu_lip_region_t* lips);

/* ============================================================================
 * SYNCHRONISATION ANALYSIS
 * ============================================================================ */

/**
 * Check if a phoneme and viseme are compatible.
 *
 * @param phoneme       Audio-derived phoneme class
 * @param viseme        Video-derived viseme class
 * @return              1 if compatible, 0 if mismatch
 */
int wu_phoneme_viseme_compatible(wu_phoneme_class_t phoneme, wu_viseme_t viseme);

/**
 * Analyse synchronisation between phoneme and viseme sequences.
 *
 * @param phonemes      Array of phoneme classes
 * @param phon_times    Phoneme timestamps (microseconds)
 * @param n_phonemes    Number of phonemes
 * @param visemes       Array of viseme classes
 * @param vis_times     Viseme timestamps (microseconds)
 * @param n_visemes     Number of visemes
 * @return              Synchronisation analysis result
 */
wu_sync_result_t wu_analyze_sync(
    const wu_phoneme_class_t* phonemes,
    const uint64_t* phon_times,
    size_t n_phonemes,
    const wu_viseme_t* visemes,
    const uint64_t* vis_times,
    size_t n_visemes
);

#ifdef __cplusplus
}
#endif

#endif /* WU_LIPSYNC_H */
