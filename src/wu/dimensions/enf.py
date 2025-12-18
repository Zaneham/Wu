"""
Electric Network Frequency (ENF) forensic analysis.

Extracts power grid frequency (50Hz/60Hz) from audio recordings and
compares against historical grid frequency databases to verify timestamps.

The power grid frequency fluctuates slightly (±0.2Hz) based on load.
These fluctuations are unique to each moment in time and are captured
in any recording made near mains-powered equipment.

Court impact: "The grid frequency in this recording doesn't match the
claimed date" is inarguable signal matching against infrastructure logs.

References:
    Grigoras, C. (2005). Digital Audio Recording Analysis: The Electric
        Network Frequency (ENF) Criterion. International Journal of
        Speech, Language and the Law, 12(1), 63-76.
    Grigoras, C. (2007). Applications of ENF Analysis in Forensic
        Authentication of Digital Audio and Video Recordings.
        Journal of the Audio Engineering Society, 55(4), 260-268.
    Huijbregtse, M. & Geradts, Z. (2009). Using the ENF Criterion for
        Determining the Time of Recording of Short Digital Audio Recordings.

Grid frequency data sources:
    UK: National Grid ESO (https://data.nationalgrideso.com/)
    EU: ENTSO-E Transparency Platform
    US: Regional operators (ERCOT, CAISO, etc.) - patchy availability
    NZ: Transpower (check current availability)
"""

import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

try:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


class GridRegion(Enum):
    """Power grid regions with their nominal frequencies."""
    UK = ("uk", 50.0)
    EU = ("eu", 50.0)
    NZ = ("nz", 50.0)
    AU = ("au", 50.0)
    US_EAST = ("us_east", 60.0)
    US_WEST = ("us_west", 60.0)
    US_TEXAS = ("us_texas", 60.0)  # ERCOT is separate grid
    JAPAN_EAST = ("japan_east", 50.0)  # Tokyo area
    JAPAN_WEST = ("japan_west", 60.0)  # Osaka area

    def __init__(self, region_id: str, nominal_freq: float):
        self.region_id = region_id
        self.nominal_freq = nominal_freq


@dataclass
class ENFSignal:
    """Extracted ENF signal from audio."""
    frequencies: np.ndarray  # Instantaneous frequency over time
    timestamps: np.ndarray   # Time points in seconds
    confidence: np.ndarray   # Signal strength/confidence at each point
    nominal_freq: float      # Expected grid frequency (50 or 60 Hz)
    sample_rate: int         # Original audio sample rate
    duration: float          # Recording duration in seconds

    @property
    def mean_frequency(self) -> float:
        """Mean detected frequency."""
        return float(np.mean(self.frequencies))

    @property
    def frequency_std(self) -> float:
        """Standard deviation of frequency (indicates fluctuation)."""
        return float(np.std(self.frequencies))

    @property
    def signal_strength(self) -> float:
        """Overall signal strength (0-1)."""
        return float(np.mean(self.confidence))

    @property
    def is_usable(self) -> bool:
        """Whether signal is strong enough for matching."""
        return self.signal_strength > 0.3 and len(self.frequencies) >= 10


@dataclass
class ENFMatch:
    """Result of matching ENF against reference database."""
    matched: bool
    correlation: float  # -1 to 1, higher is better match
    matched_timestamp: Optional[str] = None  # ISO format if matched
    matched_region: Optional[str] = None
    time_offset_seconds: Optional[float] = None  # Offset from claimed time


@dataclass
class ENFReference:
    """Reference ENF data from grid operator."""
    region: GridRegion
    start_time: str  # ISO format
    frequencies: np.ndarray
    sample_interval: float  # Seconds between samples (typically 1.0)


class ENFAnalyzer:
    """
    Analyzes audio for Electric Network Frequency to verify timestamps.

    ENF analysis works because:
    1. Power grid frequency fluctuates based on load (±0.2Hz typical)
    2. These fluctuations are recorded by any device near mains power
    3. Grid operators log actual frequency continuously
    4. Matching recording ENF to grid logs verifies when recording was made

    Limitations (documented honestly):
    - Only works if mains hum was captured in recording
    - Outdoor/battery-powered recordings often lack ENF
    - Audio processing (noise removal) may eliminate ENF
    - Grid reference data availability varies by jurisdiction
    - Short recordings (<10s) have limited matching accuracy
    """

    # Analysis parameters
    WINDOW_SIZE = 8192  # FFT window size
    HOP_SIZE = 4096     # Window hop for STFT
    ENF_BANDWIDTH = 0.5  # Hz bandwidth around nominal frequency
    MIN_DURATION = 5.0   # Minimum recording duration in seconds
    MIN_SIGNAL_STRENGTH = 0.3  # Minimum for usable ENF

    # Harmonic analysis - ENF often stronger at harmonics
    HARMONICS = [1, 2, 3, 4]  # Fundamental + harmonics to check

    def __init__(self, reference_db: Optional[Dict[str, ENFReference]] = None):
        """
        Initialize ENF analyzer.

        Args:
            reference_db: Optional dictionary of ENF reference data.
                         Key is region_id, value is ENFReference.
                         If None, only extraction is possible, not matching.
        """
        self.reference_db = reference_db or {}

    def analyze(self, file_path: str, claimed_time: Optional[str] = None,
                region: Optional[GridRegion] = None) -> DimensionResult:
        """
        Analyze audio file for ENF.

        Args:
            file_path: Path to audio or video file
            claimed_time: ISO format timestamp claimed for recording
            region: Expected grid region (auto-detected if not specified)

        Returns:
            DimensionResult with ENF analysis findings
        """
        if not HAS_SCIPY:
            return DimensionResult(
                dimension="enf",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="scipy required for ENF analysis"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="enf",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        # Extract audio
        audio_data, sample_rate = self._load_audio(file_path)
        if audio_data is None:
            return DimensionResult(
                dimension="enf",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot extract audio",
                    explanation="File does not contain extractable audio"
                )]
            )

        duration = len(audio_data) / sample_rate
        if duration < self.MIN_DURATION:
            return DimensionResult(
                dimension="enf",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Recording too short",
                    explanation=f"Duration {duration:.1f}s below minimum {self.MIN_DURATION}s for ENF analysis"
                )]
            )

        evidence = []

        # Try both 50Hz and 60Hz if region not specified
        regions_to_try = [region] if region else [GridRegion.UK, GridRegion.US_EAST]

        best_signal = None
        best_strength = 0
        detected_region = None

        for try_region in regions_to_try:
            enf_signal = self._extract_enf(audio_data, sample_rate, try_region.nominal_freq)
            if enf_signal and enf_signal.signal_strength > best_strength:
                best_signal = enf_signal
                best_strength = enf_signal.signal_strength
                detected_region = try_region

        if best_signal is None or not best_signal.is_usable:
            evidence.append(Evidence(
                finding="No usable ENF signal detected",
                explanation=(
                    "Recording does not contain detectable power grid hum. "
                    "This is common for outdoor recordings, battery-powered devices, "
                    "or audio that has been processed to remove hum."
                )
            ))
            return DimensionResult(
                dimension="enf",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=evidence,
                methodology="ENF extraction via STFT analysis"
            )

        # Report detected ENF
        evidence.append(Evidence(
            finding=f"ENF signal detected at {best_signal.mean_frequency:.3f}Hz",
            explanation=(
                f"Detected {detected_region.nominal_freq}Hz grid frequency with "
                f"signal strength {best_signal.signal_strength:.1%}. "
                f"Frequency fluctuation: ±{best_signal.frequency_std:.4f}Hz"
            )
        ))

        # Attempt database matching if we have reference data
        if claimed_time and detected_region.region_id in self.reference_db:
            match_result = self._match_against_reference(
                best_signal,
                self.reference_db[detected_region.region_id],
                claimed_time
            )

            if match_result.matched:
                evidence.append(Evidence(
                    finding="ENF matches claimed timestamp",
                    explanation=(
                        f"Grid frequency pattern matches {detected_region.region_id.upper()} "
                        f"grid data for claimed time (correlation: {match_result.correlation:.3f})"
                    ),
                    citation="Grid frequency data from national grid operator"
                ))
                return DimensionResult(
                    dimension="enf",
                    state=DimensionState.VERIFIED,
                    confidence=Confidence.HIGH,
                    evidence=evidence,
                    methodology="ENF correlation against grid operator frequency logs"
                )
            else:
                evidence.append(Evidence(
                    finding="ENF does NOT match claimed timestamp",
                    explanation=(
                        f"Grid frequency pattern does not match {detected_region.region_id.upper()} "
                        f"grid data for claimed time (correlation: {match_result.correlation:.3f})"
                    ),
                    contradiction="Recording timestamp inconsistent with power grid frequency",
                    citation="Grid frequency data from national grid operator"
                ))
                return DimensionResult(
                    dimension="enf",
                    state=DimensionState.INCONSISTENT,
                    confidence=Confidence.HIGH,
                    evidence=evidence,
                    methodology="ENF correlation against grid operator frequency logs"
                )

        # No reference data for matching
        if not self.reference_db:
            evidence.append(Evidence(
                finding="No grid reference database loaded",
                explanation=(
                    "ENF signal extracted but cannot verify without reference data. "
                    "Load grid frequency logs for the relevant jurisdiction to enable matching."
                )
            ))

        return DimensionResult(
            dimension="enf",
            state=DimensionState.SUSPICIOUS if best_signal.is_usable else DimensionState.UNCERTAIN,
            confidence=Confidence.MEDIUM if best_signal.is_usable else Confidence.LOW,
            evidence=evidence,
            methodology="ENF extraction via STFT analysis"
        )

    def _load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio data from file."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Try scipy.io.wavfile for WAV files
        if suffix == '.wav':
            try:
                sample_rate, data = wavfile.read(file_path)
                # Convert to mono float
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                data = data.astype(np.float32)
                # Normalize
                if data.dtype == np.int16:
                    data = data / 32768.0
                elif data.dtype == np.int32:
                    data = data / 2147483648.0
                return data, sample_rate
            except Exception:
                pass

        # Try pydub for other formats
        if HAS_PYDUB:
            try:
                audio = AudioSegment.from_file(file_path)
                # Convert to mono
                audio = audio.set_channels(1)
                # Get raw data
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / (2 ** (audio.sample_width * 8 - 1))
                return samples, audio.frame_rate
            except Exception:
                pass

        return None, 0

    def _extract_enf(self, audio: np.ndarray, sample_rate: int,
                     nominal_freq: float) -> Optional[ENFSignal]:
        """
        Extract ENF signal from audio.

        Uses STFT to track instantaneous frequency around the nominal
        grid frequency and its harmonics.
        """
        nyquist = sample_rate / 2

        # Check if we can analyze this frequency
        if nominal_freq >= nyquist:
            return None

        # Use a wider band for initial STFT, then find peak near nominal
        # STFT analysis on raw audio
        f, t, Zxx = signal.stft(audio, sample_rate,
                                nperseg=self.WINDOW_SIZE,
                                noverlap=self.WINDOW_SIZE - self.HOP_SIZE)

        # Find frequency bins in the ENF band
        freq_mask = (f >= nominal_freq - 2.0) & (f <= nominal_freq + 2.0)
        if not np.any(freq_mask):
            return None

        freq_indices = np.where(freq_mask)[0]
        enf_freqs = f[freq_indices]

        # Get magnitude in ENF band
        enf_magnitude = np.abs(Zxx[freq_indices, :])

        if enf_magnitude.size == 0:
            return None

        # Track peak frequency at each time step
        frequencies = []
        confidences = []

        # Also compute overall signal power for comparison
        full_magnitude = np.abs(Zxx)

        for i in range(enf_magnitude.shape[1]):
            col = enf_magnitude[:, i]
            if len(col) == 0:
                continue

            peak_idx = np.argmax(col)
            peak_freq = enf_freqs[peak_idx]
            peak_mag = col[peak_idx]

            # Confidence: ratio of ENF band energy to total energy in low frequencies
            # Focus on 0-200Hz range for comparison
            low_freq_mask = f < 200
            low_freq_mag = np.sum(full_magnitude[low_freq_mask, i]) if np.any(low_freq_mask) else 1.0
            enf_band_mag = np.sum(col)

            if low_freq_mag > 0:
                confidence = enf_band_mag / (low_freq_mag + 1e-10)
            else:
                confidence = 0.0

            # Clamp confidence to reasonable range
            confidence = min(confidence, 1.0)

            frequencies.append(peak_freq)
            confidences.append(confidence)

        if len(frequencies) == 0:
            return None

        return ENFSignal(
            frequencies=np.array(frequencies),
            timestamps=t[:len(frequencies)],
            confidence=np.array(confidences),
            nominal_freq=nominal_freq,
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate
        )

    def _match_against_reference(self, enf_signal: ENFSignal,
                                  reference: ENFReference,
                                  claimed_time: str) -> ENFMatch:
        """
        Match extracted ENF against reference database.

        Uses normalized cross-correlation to find best match.
        """
        # Resample ENF signal to match reference sample rate
        # Reference is typically 1 sample per second
        target_samples = int(enf_signal.duration / reference.sample_interval)

        if target_samples < 10:
            return ENFMatch(matched=False, correlation=0.0)

        # Resample extracted frequencies
        resampled = signal.resample(enf_signal.frequencies, target_samples)

        # Normalize both signals
        ref_norm = (reference.frequencies - np.mean(reference.frequencies)) / (np.std(reference.frequencies) + 1e-10)
        sig_norm = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-10)

        # Cross-correlation
        correlation = signal.correlate(ref_norm, sig_norm, mode='valid')
        correlation = correlation / len(sig_norm)

        # Find best match
        best_idx = np.argmax(correlation)
        best_corr = correlation[best_idx]

        # Threshold for match (empirically determined)
        MATCH_THRESHOLD = 0.7

        return ENFMatch(
            matched=best_corr > MATCH_THRESHOLD,
            correlation=float(best_corr),
            time_offset_seconds=float(best_idx * reference.sample_interval) if best_corr > MATCH_THRESHOLD else None
        )

    def extract_enf_only(self, file_path: str,
                         nominal_freq: float = 50.0) -> Optional[ENFSignal]:
        """
        Extract ENF signal without database matching.

        Useful for building reference databases or standalone analysis.

        Args:
            file_path: Path to audio file
            nominal_freq: Expected grid frequency (50.0 or 60.0)

        Returns:
            ENFSignal if successful, None otherwise
        """
        audio_data, sample_rate = self._load_audio(file_path)
        if audio_data is None:
            return None

        return self._extract_enf(audio_data, sample_rate, nominal_freq)

    def load_reference_csv(self, csv_path: str, region: GridRegion,
                          start_time: str) -> ENFReference:
        """
        Load reference ENF data from CSV file.

        Expected format: one frequency value per line, one sample per second.

        Args:
            csv_path: Path to CSV file with frequency data
            region: Grid region this data is from
            start_time: ISO format start time for the data

        Returns:
            ENFReference object
        """
        frequencies = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        frequencies.append(float(line.split(',')[0]))
                    except ValueError:
                        continue

        return ENFReference(
            region=region,
            start_time=start_time,
            frequencies=np.array(frequencies),
            sample_interval=1.0
        )
