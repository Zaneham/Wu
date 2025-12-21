"""
Wu Forensics - Lip-Sync Analysis Dimension

Deterministic audio-visual synchronisation detection for forensic analysis.
Uses fixed-point arithmetic and rule-based classification to ensure
reproducibility across platforms.

This module provides a pure Python implementation with optional native
acceleration when the compiled C library is available.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple
import struct

from ..state import DimensionResult, DimensionState, Confidence, Evidence

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# CONSTANTS
# ============================================================================

SAMPLE_RATE = 16000     # Target audio sample rate (Hz)
FFT_SIZE = 512          # FFT frame size
HOP_SIZE = 160          # 10ms hop @ 16kHz
FRAME_MS = 32           # Frame duration (ms)

# Formant frequency ranges
F1_MIN_HZ = 200
F1_MAX_HZ = 1000
F2_MIN_HZ = 800
F2_MAX_HZ = 2500

# Detection thresholds
VOICED_THRESHOLD = 0.03         # Minimum energy for voiced speech
SYNC_OFFSET_THRESHOLD_MS = 200  # Offset above this = INCONSISTENT
SYNC_OFFSET_WARN_MS = 80        # Offset above this = SUSPICIOUS
MISMATCH_RATE_THRESHOLD = 0.25  # Mismatch rate above this = SUSPICIOUS
MIN_VOICED_FRAMES = 30          # Minimum frames for reliable analysis


# ============================================================================
# ENUMS
# ============================================================================

class PhonemeClass(IntEnum):
    """Broad phoneme categories based on formant positions."""
    SILENCE = 0
    OPEN = 1            # Open vowels: /a/, /æ/, /ɑ/
    CLOSE_FRONT = 2     # Close front: /i/, /ɪ/
    CLOSE_BACK = 3      # Close back: /u/, /ʊ/
    MID = 4             # Mid vowels: /e/, /ə/, /o/
    BILABIAL = 5        # Bilabials: /p/, /b/, /m/
    OTHER = 6           # Other consonants


class Viseme(IntEnum):
    """Visual mouth shapes corresponding to phoneme groups."""
    CLOSED = 0          # Lips together
    NARROW = 1          # Small opening
    MEDIUM = 2          # Medium opening
    WIDE = 3            # Large opening
    ROUNDED = 4         # Lips protruded/rounded


# Compatibility matrix: which phonemes match which visemes
PHONEME_VISEME_COMPAT = {
    PhonemeClass.SILENCE:     {Viseme.CLOSED, Viseme.NARROW},
    PhonemeClass.OPEN:        {Viseme.MEDIUM, Viseme.WIDE},
    PhonemeClass.CLOSE_FRONT: {Viseme.NARROW, Viseme.MEDIUM},
    PhonemeClass.CLOSE_BACK:  {Viseme.NARROW, Viseme.ROUNDED},
    PhonemeClass.MID:         {Viseme.MEDIUM, Viseme.WIDE, Viseme.ROUNDED},
    PhonemeClass.BILABIAL:    {Viseme.CLOSED},
    PhonemeClass.OTHER:       {Viseme.CLOSED, Viseme.NARROW, Viseme.MEDIUM},
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Formants:
    """Formant analysis result for a single audio frame."""
    f1_hz: int          # First formant frequency (Hz)
    f2_hz: int          # Second formant frequency (Hz)
    energy: float       # Frame energy (normalised)
    voiced: bool        # True if voiced speech


@dataclass
class LipRegion:
    """Lip region measurements from a video frame."""
    center_x: int
    center_y: int
    width: int
    height: int         # Aperture
    area: int
    face_height: int    # Reference for normalisation
    valid: bool


@dataclass
class SyncResult:
    """Synchronisation analysis result."""
    offset_ms: int              # Audio-video offset (positive = audio leads)
    correlation: float          # Match rate (0-1)
    mismatch_count: int         # Frames with phoneme/viseme mismatch
    total_frames: int           # Total voiced frames analysed
    duration_ms: int            # Content duration analysed


# ============================================================================
# AUDIO ANALYSIS (PURE PYTHON)
# ============================================================================

def _hamming_window(n: int) -> 'np.ndarray':
    """Generate Hamming window coefficients."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required for audio analysis")
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def _extract_formants(magnitude: 'np.ndarray', sample_rate: int) -> Formants:
    """Extract F1/F2 formants from magnitude spectrum."""
    n_bins = len(magnitude)
    fft_size = n_bins * 2

    # Compute energy
    energy = float(np.mean(magnitude[:n_bins // 2]))

    if energy < VOICED_THRESHOLD:
        return Formants(f1_hz=0, f2_hz=0, energy=energy, voiced=False)

    # Frequency resolution
    freq_per_bin = sample_rate / fft_size

    # Find F1 peak (200-1000 Hz)
    f1_min_bin = int(F1_MIN_HZ / freq_per_bin)
    f1_max_bin = int(F1_MAX_HZ / freq_per_bin)
    f1_region = magnitude[f1_min_bin:f1_max_bin]
    f1_bin = f1_min_bin + int(np.argmax(f1_region))
    f1_hz = int(f1_bin * freq_per_bin)

    # Find F2 peak (800-2500 Hz)
    f2_min_bin = int(F2_MIN_HZ / freq_per_bin)
    f2_max_bin = min(int(F2_MAX_HZ / freq_per_bin), n_bins)
    f2_region = magnitude[f2_min_bin:f2_max_bin]
    f2_bin = f2_min_bin + int(np.argmax(f2_region))
    f2_hz = int(f2_bin * freq_per_bin)

    return Formants(f1_hz=f1_hz, f2_hz=f2_hz, energy=energy, voiced=True)


def _classify_phoneme(formants: Formants) -> PhonemeClass:
    """Classify phoneme from formant values using rule-based logic."""
    if not formants.voiced:
        return PhonemeClass.SILENCE

    f1 = formants.f1_hz
    f2 = formants.f2_hz

    # Bilabial detection (simplified)
    if formants.energy < VOICED_THRESHOLD * 2 and f1 < 400:
        return PhonemeClass.BILABIAL

    # Open vowels: high F1
    if f1 >= 600:
        return PhonemeClass.OPEN

    # Close front: low F1, high F2
    if f1 < 400 and f2 > 2000:
        return PhonemeClass.CLOSE_FRONT

    # Close back: low F1, low F2
    if f1 < 400 and f2 < 1200:
        return PhonemeClass.CLOSE_BACK

    # Mid vowels
    if 400 <= f1 < 600:
        return PhonemeClass.MID

    return PhonemeClass.OTHER


def audio_to_phonemes(
    samples: 'np.ndarray',
    sample_rate: int = SAMPLE_RATE
) -> Tuple[List[PhonemeClass], List[int]]:
    """
    Process audio samples to extract phoneme sequence.

    Args:
        samples: Audio samples (mono, float or int16)
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (phoneme_classes, timestamps_us)
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required for audio analysis")

    # Normalise to float
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0

    # Resample if necessary (simple decimation for now)
    if sample_rate != SAMPLE_RATE:
        ratio = sample_rate / SAMPLE_RATE
        indices = (np.arange(len(samples) / ratio) * ratio).astype(int)
        samples = samples[indices]
        sample_rate = SAMPLE_RATE

    phonemes = []
    timestamps = []
    window = _hamming_window(FFT_SIZE)

    offset = 0
    while offset + FFT_SIZE <= len(samples):
        # Extract frame
        frame = samples[offset:offset + FFT_SIZE] * window

        # Compute FFT magnitude
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum) / FFT_SIZE

        # Extract formants and classify
        formants = _extract_formants(magnitude, sample_rate)
        phoneme = _classify_phoneme(formants)

        phonemes.append(phoneme)
        timestamps.append(int(offset * 1000000 / sample_rate))

        offset += HOP_SIZE

    return phonemes, timestamps


# ============================================================================
# VIDEO ANALYSIS
# ============================================================================

def _rgb_to_ycbcr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to YCbCr using integer arithmetic."""
    y = (77 * r + 150 * g + 29 * b) >> 8
    cb = ((-43 * r - 85 * g + 128 * b) >> 8) + 128
    cr = ((128 * r - 107 * g - 21 * b) >> 8) + 128
    return y, cb, cr


def _is_lip_pixel(y: int, cb: int, cr: int) -> bool:
    """Check if pixel is likely lip colour based on YCbCr values."""
    # Cr should be elevated (reddish)
    if cr < 140 or cr > 180:
        return False
    # Cb should be moderate
    if cb < 100 or cb > 130:
        return False
    # Reasonable luminance
    if y < 50 or y > 200:
        return False
    # Ratio check
    if cr <= cb:
        return False
    return True


def detect_lips_color(
    frame_rgb: 'np.ndarray',
    face_bbox: Tuple[int, int, int, int]
) -> LipRegion:
    """
    Detect lip region using colour segmentation.

    Args:
        frame_rgb: RGB frame (H, W, 3)
        face_bbox: Face bounding box (x, y, width, height)

    Returns:
        LipRegion with measurements
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required for video analysis")

    face_x, face_y, face_w, face_h = face_bbox
    height, width = frame_rgb.shape[:2]

    # Focus on lower third of face
    mouth_y = face_y + (face_h * 2) // 3
    mouth_h = face_h // 3
    mouth_x = face_x + face_w // 4
    mouth_w = face_w // 2

    # Bounds check
    mouth_x = max(0, min(mouth_x, width - 1))
    mouth_y = max(0, min(mouth_y, height - 1))
    mouth_w = min(mouth_w, width - mouth_x)
    mouth_h = min(mouth_h, height - mouth_y)

    # Extract mouth region
    region = frame_rgb[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]

    # Create lip mask
    lip_mask = np.zeros((mouth_h, mouth_w), dtype=bool)

    for y in range(mouth_h):
        for x in range(mouth_w):
            r, g, b = region[y, x]
            Y, cb, cr = _rgb_to_ycbcr(int(r), int(g), int(b))
            if _is_lip_pixel(Y, cb, cr):
                lip_mask[y, x] = True

    # Find bounding box of lip pixels
    lip_pixels = np.argwhere(lip_mask)

    if len(lip_pixels) < 50:
        return LipRegion(
            center_x=0, center_y=0, width=0, height=0,
            area=0, face_height=face_h, valid=False
        )

    min_y, min_x = lip_pixels.min(axis=0)
    max_y, max_x = lip_pixels.max(axis=0)

    return LipRegion(
        center_x=mouth_x + (min_x + max_x) // 2,
        center_y=mouth_y + (min_y + max_y) // 2,
        width=max_x - min_x + 1,
        height=max_y - min_y + 1,
        area=len(lip_pixels),
        face_height=face_h,
        valid=True
    )


def classify_viseme(lips: LipRegion) -> Viseme:
    """Classify viseme from lip measurements."""
    if not lips.valid or lips.face_height == 0:
        return Viseme.CLOSED

    # Aperture ratio (per-mille of face height)
    aperture_ratio = (lips.height * 1000) // lips.face_height

    # Aspect ratio (width / height * 100)
    aspect_ratio = (lips.width * 100) // (lips.height + 1)

    if aperture_ratio < 20:
        return Viseme.CLOSED

    if aperture_ratio < 50:
        if aspect_ratio < 200:
            return Viseme.ROUNDED
        return Viseme.NARROW

    if aperture_ratio < 100:
        if aspect_ratio < 150:
            return Viseme.ROUNDED
        return Viseme.MEDIUM

    return Viseme.WIDE


def video_to_visemes(
    frames: List['np.ndarray'],
    face_bboxes: List[Tuple[int, int, int, int]],
    frame_times_us: List[int]
) -> Tuple[List[Viseme], List[int]]:
    """
    Process video frames to extract viseme sequence.

    Args:
        frames: List of RGB frames
        face_bboxes: Face bounding boxes per frame
        frame_times_us: Frame timestamps in microseconds

    Returns:
        Tuple of (visemes, timestamps_us)
    """
    visemes = []
    timestamps = []

    for frame, bbox, time_us in zip(frames, face_bboxes, frame_times_us):
        lips = detect_lips_color(frame, bbox)
        viseme = classify_viseme(lips)
        visemes.append(viseme)
        timestamps.append(time_us)

    return visemes, timestamps


# ============================================================================
# SYNCHRONISATION ANALYSIS
# ============================================================================

def phoneme_viseme_compatible(phoneme: PhonemeClass, viseme: Viseme) -> bool:
    """Check if a phoneme and viseme are compatible."""
    return viseme in PHONEME_VISEME_COMPAT.get(phoneme, set())


def analyze_sync(
    phonemes: List[PhonemeClass],
    phon_times: List[int],
    visemes: List[Viseme],
    vis_times: List[int]
) -> SyncResult:
    """
    Analyse synchronisation between phoneme and viseme sequences.

    Args:
        phonemes: Audio-derived phoneme classes
        phon_times: Phoneme timestamps (microseconds)
        visemes: Video-derived viseme classes
        vis_times: Viseme timestamps (microseconds)

    Returns:
        SyncResult with offset and mismatch information
    """
    if not phonemes or not visemes:
        return SyncResult(
            offset_ms=0, correlation=0.0,
            mismatch_count=0, total_frames=0, duration_ms=0
        )

    # Compute duration
    max_time = max(
        phon_times[-1] if phon_times else 0,
        vis_times[-1] if vis_times else 0
    )
    duration_ms = max_time // 1000

    # Find optimal offset using grid search
    best_offset_us = 0
    best_match_count = 0

    # Test offsets from -500ms to +500ms in 10ms steps
    for offset_us in range(-500000, 500001, 10000):
        match_count = 0

        for pi, (phoneme, phon_time) in enumerate(zip(phonemes, phon_times)):
            if phoneme == PhonemeClass.SILENCE:
                continue

            target_time = phon_time + offset_us
            if target_time < 0:
                continue

            # Find nearest viseme
            min_diff = float('inf')
            nearest_viseme = Viseme.CLOSED

            for viseme, vis_time in zip(visemes, vis_times):
                diff = abs(vis_time - target_time)
                if diff < min_diff:
                    min_diff = diff
                    nearest_viseme = viseme

            # Check compatibility (within 50ms)
            if min_diff < 50000:
                if phoneme_viseme_compatible(phoneme, nearest_viseme):
                    match_count += 1

        if match_count > best_match_count:
            best_match_count = match_count
            best_offset_us = offset_us

    # Count mismatches at best offset
    voiced_count = 0
    mismatch_count = 0

    for phoneme, phon_time in zip(phonemes, phon_times):
        if phoneme == PhonemeClass.SILENCE:
            continue
        voiced_count += 1

        target_time = phon_time + best_offset_us
        if target_time < 0:
            continue

        # Find nearest viseme
        min_diff = float('inf')
        nearest_viseme = Viseme.CLOSED

        for viseme, vis_time in zip(visemes, vis_times):
            diff = abs(vis_time - target_time)
            if diff < min_diff:
                min_diff = diff
                nearest_viseme = viseme

        if min_diff < 50000:
            if not phoneme_viseme_compatible(phoneme, nearest_viseme):
                mismatch_count += 1

    # Compute correlation
    correlation = 0.0
    if voiced_count > 0:
        correlation = (voiced_count - mismatch_count) / voiced_count

    return SyncResult(
        offset_ms=best_offset_us // 1000,
        correlation=correlation,
        mismatch_count=mismatch_count,
        total_frames=voiced_count,
        duration_ms=duration_ms
    )


# ============================================================================
# DIMENSION INTERFACE
# ============================================================================

def analyze_lipsync(
    audio_samples: 'np.ndarray',
    audio_sample_rate: int,
    video_frames: List['np.ndarray'],
    face_bboxes: List[Tuple[int, int, int, int]],
    video_fps: float
) -> DimensionResult:
    """
    Perform lip-sync analysis on audio/video content.

    Args:
        audio_samples: Audio samples (mono)
        audio_sample_rate: Audio sample rate (Hz)
        video_frames: List of RGB video frames
        face_bboxes: Face bounding boxes per frame
        video_fps: Video frame rate

    Returns:
        DimensionResult with synchronisation findings
    """
    # Extract phonemes from audio
    phonemes, phon_times = audio_to_phonemes(audio_samples, audio_sample_rate)

    # Generate video frame timestamps
    frame_times = [int(i * 1000000 / video_fps) for i in range(len(video_frames))]

    # Extract visemes from video
    visemes, vis_times = video_to_visemes(video_frames, face_bboxes, frame_times)

    # Analyse synchronisation
    sync = analyze_sync(phonemes, phon_times, visemes, vis_times)

    # Generate result
    evidence_list = []
    abs_offset = abs(sync.offset_ms)

    if sync.total_frames < MIN_VOICED_FRAMES:
        return DimensionResult(
            dimension="lipsync",
            state=DimensionState.UNCERTAIN,
            confidence=Confidence.NA,
            evidence=[Evidence(
                finding="Insufficient voiced content",
                explanation=f"Only {sync.total_frames} voiced frames detected "
                           f"(minimum {MIN_VOICED_FRAMES} required for reliable analysis)"
            )],
            methodology="Deterministic phoneme-viseme correlation (Wu v1.3)",
            raw_data={"sync": sync.__dict__}
        )

    if abs_offset > SYNC_OFFSET_THRESHOLD_MS:
        mismatch_pct = (sync.mismatch_count / sync.total_frames * 100) if sync.total_frames else 0

        return DimensionResult(
            dimension="lipsync",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding=f"Audio-visual offset of {sync.offset_ms}ms detected",
                explanation=f"Audio {'leads' if sync.offset_ms > 0 else 'lags'} "
                           f"video by {abs_offset}ms. Threshold for inconsistency: "
                           f"{SYNC_OFFSET_THRESHOLD_MS}ms. "
                           f"Mismatch rate: {mismatch_pct:.1f}%",
                contradiction="Temporal alignment outside normal recording tolerance"
            )],
            methodology="Deterministic phoneme-viseme correlation (Wu v1.3)",
            raw_data={"sync": sync.__dict__}
        )

    mismatch_rate = sync.mismatch_count / sync.total_frames if sync.total_frames else 0

    if abs_offset > SYNC_OFFSET_WARN_MS or mismatch_rate > MISMATCH_RATE_THRESHOLD:
        return DimensionResult(
            dimension="lipsync",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.MEDIUM,
            evidence=[Evidence(
                finding=f"Subtle audio-visual anomalies detected",
                explanation=f"Offset: {sync.offset_ms}ms, "
                           f"mismatch rate: {mismatch_rate * 100:.1f}% "
                           f"({sync.mismatch_count}/{sync.total_frames} frames)"
            )],
            methodology="Deterministic phoneme-viseme correlation (Wu v1.3)",
            raw_data={"sync": sync.__dict__}
        )

    return DimensionResult(
        dimension="lipsync",
        state=DimensionState.CONSISTENT,
        confidence=Confidence.HIGH,
        evidence=[Evidence(
            finding="Audio-visual synchronisation within normal range",
            explanation=f"Offset: {sync.offset_ms}ms, "
                       f"mismatch rate: {mismatch_rate * 100:.1f}%. "
                       f"Analysed {sync.total_frames} voiced frames "
                       f"over {sync.duration_ms}ms"
        )],
        methodology="Deterministic phoneme-viseme correlation (Wu v1.3)",
        raw_data={"sync": sync.__dict__}
    )


# ============================================================================
# ANALYZER CLASS
# ============================================================================

class LipSyncAnalyzer:
    """
    Lip-sync forensic analyzer for audio-visual synchronisation detection.

    This analyzer requires both audio and video streams, so it cannot be
    used with the standard single-file analyze() pattern. Instead, use
    analyze_streams() with extracted audio and video data.

    The analysis is fully deterministic - identical inputs will always
    produce identical outputs regardless of platform.
    """

    def __init__(self):
        """Initialise the lip-sync analyzer."""
        self._face_cascade = None

    def _get_face_cascade(self):
        """Lazy-load Haar cascade for face detection."""
        if self._face_cascade is None and HAS_CV2:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def _detect_faces(self, frame: 'np.ndarray') -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame using Haar cascade (deterministic)."""
        if not HAS_CV2:
            return []

        cascade = self._get_face_cascade()
        if cascade is None:
            return []

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Detect faces
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def analyze_streams(
        self,
        audio_samples: 'np.ndarray',
        audio_sample_rate: int,
        video_frames: List['np.ndarray'],
        video_fps: float,
        face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> DimensionResult:
        """
        Analyse lip-sync between audio and video streams.

        Args:
            audio_samples: Mono audio samples (int16 or float)
            audio_sample_rate: Audio sample rate in Hz
            video_frames: List of RGB video frames
            video_fps: Video frame rate
            face_bboxes: Optional pre-detected face bounding boxes.
                         If None, faces will be detected automatically.

        Returns:
            DimensionResult with synchronisation findings
        """
        if not HAS_NUMPY:
            return DimensionResult(
                dimension="lipsync",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="NumPy not available",
                    explanation="Lip-sync analysis requires NumPy"
                )],
                methodology="Deterministic phoneme-viseme correlation (Wu v1.3)"
            )

        if len(video_frames) == 0:
            return DimensionResult(
                dimension="lipsync",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="No video frames provided",
                    explanation="Lip-sync analysis requires video frames"
                )],
                methodology="Deterministic phoneme-viseme correlation (Wu v1.3)"
            )

        # Detect faces if not provided
        if face_bboxes is None:
            face_bboxes = []
            for frame in video_frames:
                faces = self._detect_faces(frame)
                if faces:
                    face_bboxes.append(faces[0])  # Use first face
                else:
                    # Use centre of frame as fallback
                    h, w = frame.shape[:2]
                    face_bboxes.append((w // 4, h // 4, w // 2, h // 2))

        return analyze_lipsync(
            audio_samples=audio_samples,
            audio_sample_rate=audio_sample_rate,
            video_frames=video_frames,
            face_bboxes=face_bboxes,
            video_fps=video_fps
        )

    def is_available(self) -> bool:
        """Check if lip-sync analysis is available."""
        return HAS_NUMPY
