# Deterministic Lip-Sync Analysis

**Status:** Design Draft
**Goal:** Detect audio-visual desynchronization with bitwise reproducibility across platforms.

---

## Design Principles

1. **No floating-point in critical paths** - Use Q15/Q31 fixed-point arithmetic
2. **No ML inference** - Rule-based phoneme/viseme classification
3. **Explicit versioning** - Algorithm version embedded in output for reproducibility
4. **Conservative thresholds** - Prefer false negatives over false positives

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│     AUDIO TRACK       │       │     VIDEO FRAMES      │
│                       │       │                       │
│  16-bit PCM @ 16kHz   │       │  Face region crops    │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   FIXED-POINT FFT     │       │   LIP REGION EXTRACT  │
│                       │       │                       │
│  Q15 Radix-2 DIT      │       │  Geometric detection  │
│  512-sample frames    │       │  No neural landmarks  │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   FORMANT EXTRACT     │       │   LIP MEASUREMENTS    │
│                       │       │                       │
│  F1: 200-1000 Hz      │       │  - Aperture (height)  │
│  F2: 800-2500 Hz      │       │  - Width              │
│  Energy envelope      │       │  - Area ratio         │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   PHONEME CLASSIFY    │       │   VISEME CLASSIFY     │
│                       │       │                       │
│  Rule-based F1/F2     │       │  Rule-based geometry  │
│  mapping to phoneme   │       │  mapping to viseme    │
│  classes              │       │  classes              │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   TEMPORAL ALIGNMENT          │
            │                               │
            │  Cross-correlate phoneme      │
            │  and viseme sequences         │
            │  Detect offset & mismatches   │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   EPISTEMIC OUTPUT            │
            │                               │
            │  CONSISTENT / INCONSISTENT    │
            │  with evidence timestamps     │
            └───────────────────────────────┘
```

---

## Component Details

### 1. Audio Preprocessing

**Input:** Raw audio track, any sample rate
**Output:** 16-bit PCM @ 16kHz mono

```c
// Resample to exactly 16000 Hz using integer-ratio SRC
// This avoids floating-point interpolation variance
typedef struct {
    int32_t num;    // e.g., 16000
    int32_t denom;  // e.g., 44100
} wu_rational_t;

// Polyphase FIR with fixed Q15 coefficients
void wu_resample_rational(
    const int16_t* in, size_t in_len,
    int16_t* out, size_t* out_len,
    wu_rational_t ratio
);
```

### 2. Fixed-Point FFT

**Why:** Floating-point FFT results vary across platforms (x87 vs SSE vs ARM).

**Specification:**
- Radix-2 DIT, 512 samples (32ms @ 16kHz)
- Q15 input, Q15 output (with scaling)
- Hop size: 160 samples (10ms)
- Window: Hamming, precomputed Q15 coefficients

```c
// Q15 format: value = integer / 32768
// Range: -1.0 to +0.999969

typedef int16_t q15_t;
typedef int32_t q31_t;

// In-place FFT, deterministic across all platforms
void wu_fft_q15(q15_t* real, q15_t* imag, size_t n);

// Magnitude spectrum (Q15 output)
void wu_fft_magnitude_q15(
    const q15_t* real,
    const q15_t* imag,
    q15_t* magnitude,
    size_t n
);
```

**Reference implementation:** Use CMSIS-DSP `arm_cfft_q15` algorithm, ported to pure C. This is battle-tested and has known numerical properties.

### 3. Formant Extraction

**Goal:** Extract F1 (first formant) and F2 (second formant) frequencies.

**Method:** Peak picking in smoothed spectrum, not LPC (LPC has numerical stability issues).

```c
typedef struct {
    uint16_t f1_hz;      // First formant frequency
    uint16_t f2_hz;      // Second formant frequency
    q15_t    energy;     // Frame energy (Q15)
    uint8_t  voiced;     // 1 = voiced speech, 0 = unvoiced/silence
} wu_formants_t;

// Extract formants from magnitude spectrum
// Uses fixed bin ranges and deterministic peak picking
wu_formants_t wu_extract_formants(
    const q15_t* magnitude,
    size_t n_bins,
    uint32_t sample_rate
);
```

**Peak picking algorithm:**
1. Smooth spectrum with 3-tap median filter (no floating point)
2. Find local maxima in F1 range (200-1000 Hz)
3. Find local maxima in F2 range (800-2500 Hz)
4. Select strongest peaks

### 4. Phoneme Classification

**Approach:** Rule-based F1/F2 mapping to phoneme *classes* (not individual phonemes).

We don't need to identify "was that an /æ/ or an /ɛ/?" - we need broader categories that map to visible mouth shapes.

**Phoneme Classes (audio-derived):**

| Class | F1 Range | F2 Range | Description |
|-------|----------|----------|-------------|
| `OPEN` | 600-1000 | any | Open vowels (a, æ, ɑ) |
| `CLOSE_FRONT` | 200-400 | 2000-2500 | Close front (i, ɪ) |
| `CLOSE_BACK` | 200-400 | 800-1200 | Close back (u, ʊ) |
| `MID` | 400-600 | 1200-2000 | Mid vowels (e, ə, o) |
| `BILABIAL` | - | - | Detected by energy dip + burst (p, b, m) |
| `SILENCE` | - | - | Energy below threshold |

```c
typedef enum {
    WU_PHON_SILENCE = 0,
    WU_PHON_OPEN,
    WU_PHON_CLOSE_FRONT,
    WU_PHON_CLOSE_BACK,
    WU_PHON_MID,
    WU_PHON_BILABIAL,
    WU_PHON_OTHER_CONSONANT
} wu_phoneme_class_t;

wu_phoneme_class_t wu_classify_phoneme(const wu_formants_t* f);
```

### 5. Lip Region Detection (Video Side)

**Challenge:** Most facial landmark detectors are neural (dlib, mediapipe). These are non-deterministic.

**Deterministic alternative:** Classical CV approach.

**Option A: Color-based lip segmentation**
```c
// Convert to YCbCr (integer arithmetic)
// Lip detection via Cb/Cr thresholds
// Well-documented in literature, deterministic

typedef struct {
    uint16_t x, y;           // Lip region center
    uint16_t width, height;  // Bounding box
    uint16_t aperture;       // Vertical opening (pixels)
    uint16_t spread;         // Horizontal width (pixels)
    uint32_t area;           // Lip pixel count
} wu_lip_region_t;

// Requires face bounding box as input (can use Haar cascade, which is deterministic)
wu_lip_region_t wu_detect_lips_color(
    const uint8_t* frame_rgb,
    uint32_t width, uint32_t height,
    const wu_rect_t* face_bbox
);
```

**Option B: Accept controlled non-determinism**

Use a specific frozen model (e.g., dlib shape predictor 68) with:
- Exact version pinned
- Input preprocessing fixed (resize to 256x256, specific interpolation)
- Document that "Wu lip-sync uses dlib 19.24 shape_predictor_68_face_landmarks.dat"

This is "reproducible given same dependencies" rather than "bitwise identical across all platforms."

**Recommendation:** Option A for maximum determinism, with Option B as fallback when color-based fails (low light, occlusion).

### 6. Viseme Classification

**Visemes:** Visual mouth shapes that correspond to phoneme groups.

| Viseme | Lip State | Corresponds to |
|--------|-----------|----------------|
| `CLOSED` | Lips together | Silence, m, b, p |
| `NARROW` | Small opening | i, u, close vowels |
| `MEDIUM` | Medium opening | e, o, mid vowels |
| `WIDE` | Large opening | a, open vowels |
| `ROUNDED` | Lips protruded | o, u, w |

```c
typedef enum {
    WU_VIS_CLOSED = 0,
    WU_VIS_NARROW,
    WU_VIS_MEDIUM,
    WU_VIS_WIDE,
    WU_VIS_ROUNDED
} wu_viseme_t;

// Thresholds are relative to detected face size
// aperture_ratio = lip_aperture / face_height
wu_viseme_t wu_classify_viseme(
    const wu_lip_region_t* lips,
    uint32_t face_height
);
```

**Threshold table (tunable, but fixed at runtime):**

```c
// These would be calibrated empirically and frozen
#define VIS_CLOSED_MAX_RATIO   0.02   // < 2% = closed
#define VIS_NARROW_MAX_RATIO   0.05   // < 5% = narrow
#define VIS_MEDIUM_MAX_RATIO   0.10   // < 10% = medium
// > 10% = wide
```

### 7. Temporal Alignment

**Goal:** Find if phonemes and visemes are synchronized.

**Method:**
1. Convert both sequences to numeric vectors
2. Cross-correlate to find optimal offset
3. Measure residual mismatches after alignment

```c
typedef struct {
    int32_t  offset_ms;       // Audio leads video if positive
    q15_t    correlation;     // Peak correlation (Q15)
    uint32_t mismatch_count;  // Frames where phoneme/viseme incompatible
    uint32_t total_frames;    // Total voiced frames analyzed
} wu_sync_result_t;

// Expected mapping: which visemes are compatible with which phonemes
static const uint8_t PHONEME_VISEME_COMPAT[7][5] = {
    //             CLOSED NARROW MEDIUM WIDE ROUNDED
    /* SILENCE */  { 1,     1,     0,    0,    0 },
    /* OPEN */     { 0,     0,     1,    1,    0 },
    /* CL_FRONT */ { 0,     1,     1,    0,    0 },
    /* CL_BACK */  { 0,     1,     0,    0,    1 },
    /* MID */      { 0,     0,     1,    1,    1 },
    /* BILABIAL */ { 1,     0,     0,    0,    0 },
    /* OTHER_C */  { 1,     1,     1,    0,    0 },
};

wu_sync_result_t wu_analyze_sync(
    const wu_phoneme_class_t* phonemes,
    const uint64_t* phoneme_times_us,
    size_t n_phonemes,
    const wu_viseme_t* visemes,
    const uint64_t* viseme_times_us,
    size_t n_visemes
);
```

### 8. Decision Logic

```c
typedef struct {
    wu_sync_result_t sync;
    wu_dimension_state_t state;
    wu_confidence_t confidence;
    char evidence[512];
} wu_lipsync_result_t;

wu_lipsync_result_t wu_evaluate_lipsync(const wu_sync_result_t* sync) {
    wu_lipsync_result_t r = {0};
    r.sync = *sync;

    float mismatch_rate = (float)sync->mismatch_count / sync->total_frames;
    int32_t abs_offset = abs(sync->offset_ms);

    // Decision thresholds (forensically conservative)
    if (sync->total_frames < 30) {
        // Less than 1 second of voiced speech
        r.state = WU_STATE_UNCERTAIN;
        r.confidence = WU_CONF_NA;
        snprintf(r.evidence, sizeof(r.evidence),
            "Insufficient voiced frames (%u) for reliable analysis",
            sync->total_frames);
    }
    else if (abs_offset > 200) {
        // More than 200ms offset is clearly wrong
        r.state = WU_STATE_INCONSISTENT;
        r.confidence = WU_CONF_HIGH;
        snprintf(r.evidence, sizeof(r.evidence),
            "Audio-visual offset of %dms detected (threshold: 200ms). "
            "Audio %s video by %dms.",
            abs_offset,
            sync->offset_ms > 0 ? "leads" : "lags",
            abs_offset);
    }
    else if (abs_offset > 80 || mismatch_rate > 0.25) {
        // Subtle but significant desync
        r.state = WU_STATE_SUSPICIOUS;
        r.confidence = WU_CONF_MEDIUM;
        snprintf(r.evidence, sizeof(r.evidence),
            "Offset: %dms, mismatch rate: %.1f%% (%u/%u frames)",
            sync->offset_ms, mismatch_rate * 100,
            sync->mismatch_count, sync->total_frames);
    }
    else {
        r.state = WU_STATE_CONSISTENT;
        r.confidence = WU_CONF_HIGH;
        snprintf(r.evidence, sizeof(r.evidence),
            "Audio-visual sync within normal range. "
            "Offset: %dms, mismatch rate: %.1f%%",
            sync->offset_ms, mismatch_rate * 100);
    }

    return r;
}
```

---

## Fixed-Point Primitives Needed

These would go in your native assembler layer:

| Function | Purpose | Notes |
|----------|---------|-------|
| `wu_fft_q15` | 512-point FFT | Radix-2 DIT, Q15 I/O |
| `wu_magnitude_q15` | Complex to magnitude | `sqrt(re² + im²)` in fixed-point |
| `wu_median3_q15` | 3-tap median filter | For spectrum smoothing |
| `wu_xcorr_q15` | Cross-correlation | For temporal alignment |
| `wu_peak_pick_q15` | Find local maxima | For formant extraction |

**ARM NEON versions** of these would be straightforward - NEON has good Q15 support via `vqadd`, `vqdmulh`, etc.

---

## Limitations to Document

1. **Language dependence** - F1/F2 ranges tuned for English. Other languages may need adjusted thresholds.

2. **Compression artifacts** - Heavy video compression degrades lip region detection. Should report confidence reduction.

3. **Occlusion** - Microphones, hands, beards reduce accuracy. Detector should flag "insufficient lip visibility."

4. **Frame rate** - Below 24fps, temporal resolution may be insufficient. Flag as UNCERTAIN.

5. **Not a deepfake detector** - This detects *desynchronization*, not *generation*. A well-synced deepfake passes this test.

---

## Validation Approach

1. **Synthetic test cases:**
   - Generate videos with known offsets (0ms, 50ms, 100ms, 200ms, 500ms)
   - Verify detection at each level

2. **Cross-platform reproducibility:**
   - Run same video on Windows x64, macOS ARM64, Linux x64
   - Verify identical output (not just "similar" - bitwise identical)

3. **False positive testing:**
   - Test on genuine videos with poor audio (compression, background noise)
   - Ensure we don't flag authentic content

4. **Known deepfake corpus:**
   - Test on FaceForensics++, DFDC datasets where ground truth is known

---

## Implementation Order

1. **Q15 FFT** - Core primitive, test thoroughly
2. **Formant extraction** - Build on FFT
3. **Phoneme classification** - Rule engine
4. **Lip detection (color-based)** - Parallel workstream
5. **Viseme classification** - Build on lip detection
6. **Temporal alignment** - Integration
7. **ARM NEON ports** - After x86 is validated

---

## References

- Owens, A., et al. "Visually Indicated Sounds." CVPR 2016. (Audio-visual correlation)
- Harte, N., Gillen, E. "TCD-TIMIT: An Audio-Visual Corpus." (Validation dataset)
- CMSIS-DSP Library. ARM. (Fixed-point FFT reference)
- Peterson, G., Barney, H. "Control Methods Used in a Study of the Vowels." JASA 1952. (F1/F2 vowel space)
