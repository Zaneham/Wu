"""
Tests for ENF (Electric Network Frequency) forensic analysis.

Tests ENF extraction from audio, signal processing, and database matching.
"""

import pytest
import tempfile
import struct
import wave
from pathlib import Path

try:
    import numpy as np
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from wu.dimensions.enf import (
    ENFAnalyzer, ENFSignal, ENFMatch, ENFReference, GridRegion
)
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return ENFAnalyzer()


def create_test_wav(path: str, duration: float = 10.0, sample_rate: int = 44100,
                    add_enf: bool = False, enf_freq: float = 50.0,
                    enf_amplitude: float = 0.01):
    """Create a test WAV file with optional ENF signal."""
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Base noise
    audio = np.random.randn(num_samples) * 0.1

    if add_enf:
        # Add simulated ENF (50Hz or 60Hz hum with slight variations)
        freq_variation = np.sin(2 * np.pi * 0.1 * t) * 0.05  # Slow frequency drift
        enf_signal = enf_amplitude * np.sin(2 * np.pi * (enf_freq + freq_variation) * t)
        audio += enf_signal

    # Normalize to int16 range
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


class TestGridRegion:
    """Test GridRegion enum."""

    def test_uk_region(self):
        """UK region is 50Hz."""
        assert GridRegion.UK.nominal_freq == 50.0
        assert GridRegion.UK.region_id == "uk"

    def test_us_region(self):
        """US regions are 60Hz."""
        assert GridRegion.US_EAST.nominal_freq == 60.0
        assert GridRegion.US_WEST.nominal_freq == 60.0
        assert GridRegion.US_TEXAS.nominal_freq == 60.0

    def test_japan_split(self):
        """Japan has both 50Hz and 60Hz regions."""
        assert GridRegion.JAPAN_EAST.nominal_freq == 50.0
        assert GridRegion.JAPAN_WEST.nominal_freq == 60.0

    def test_nz_region(self):
        """NZ is 50Hz."""
        assert GridRegion.NZ.nominal_freq == 50.0


class TestENFAnalyzerBasic:
    """Test basic ENFAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyzer_with_reference_db(self):
        """Analyzer can be created with reference database."""
        analyzer = ENFAnalyzer(reference_db={})
        assert analyzer.reference_db == {}

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.wav")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "enf"

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_text("not audio")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN


class TestENFSignal:
    """Test ENFSignal dataclass."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_signal_creation(self):
        """ENFSignal can be created."""
        signal = ENFSignal(
            frequencies=np.array([50.0, 50.01, 49.99]),
            timestamps=np.array([0.0, 1.0, 2.0]),
            confidence=np.array([0.8, 0.7, 0.9]),
            nominal_freq=50.0,
            sample_rate=44100,
            duration=3.0
        )
        assert signal.mean_frequency == pytest.approx(50.0, abs=0.01)
        assert signal.duration == 3.0

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_signal_strength(self):
        """Signal strength is computed correctly."""
        signal = ENFSignal(
            frequencies=np.array([50.0, 50.0, 50.0]),
            timestamps=np.array([0.0, 1.0, 2.0]),
            confidence=np.array([0.5, 0.6, 0.7]),
            nominal_freq=50.0,
            sample_rate=44100,
            duration=3.0
        )
        assert signal.signal_strength == pytest.approx(0.6, abs=0.01)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_is_usable_strong_signal(self):
        """Strong signal is usable."""
        signal = ENFSignal(
            frequencies=np.array([50.0] * 20),
            timestamps=np.arange(20, dtype=float),
            confidence=np.array([0.8] * 20),
            nominal_freq=50.0,
            sample_rate=44100,
            duration=20.0
        )
        assert signal.is_usable is True

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_is_usable_weak_signal(self):
        """Weak signal is not usable."""
        signal = ENFSignal(
            frequencies=np.array([50.0] * 20),
            timestamps=np.arange(20, dtype=float),
            confidence=np.array([0.1] * 20),  # Low confidence
            nominal_freq=50.0,
            sample_rate=44100,
            duration=20.0
        )
        assert signal.is_usable is False

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_is_usable_short_signal(self):
        """Short signal is not usable."""
        signal = ENFSignal(
            frequencies=np.array([50.0] * 5),  # Only 5 samples
            timestamps=np.arange(5, dtype=float),
            confidence=np.array([0.8] * 5),
            nominal_freq=50.0,
            sample_rate=44100,
            duration=5.0
        )
        assert signal.is_usable is False


class TestENFMatch:
    """Test ENFMatch dataclass."""

    def test_match_result(self):
        """ENFMatch can be created."""
        match = ENFMatch(
            matched=True,
            correlation=0.85,
            matched_timestamp="2024-01-15T14:30:00",
            matched_region="uk"
        )
        assert match.matched is True
        assert match.correlation == 0.85


class TestENFExtraction:
    """Test ENF signal extraction."""

    @pytest.fixture
    def temp_wav_with_enf(self, tmp_path):
        """Create WAV with simulated ENF."""
        if not HAS_SCIPY:
            pytest.skip("scipy required")
        path = tmp_path / "with_enf.wav"
        create_test_wav(str(path), duration=15.0, add_enf=True, enf_freq=50.0, enf_amplitude=0.05)
        return str(path)

    @pytest.fixture
    def temp_wav_without_enf(self, tmp_path):
        """Create WAV without ENF."""
        if not HAS_SCIPY:
            pytest.skip("scipy required")
        path = tmp_path / "no_enf.wav"
        create_test_wav(str(path), duration=15.0, add_enf=False)
        return str(path)

    @pytest.fixture
    def temp_wav_short(self, tmp_path):
        """Create short WAV file."""
        if not HAS_SCIPY:
            pytest.skip("scipy required")
        path = tmp_path / "short.wav"
        create_test_wav(str(path), duration=2.0)
        return str(path)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_extract_enf_with_signal(self, analyzer, temp_wav_with_enf):
        """ENF can be extracted from audio with hum."""
        signal = analyzer.extract_enf_only(temp_wav_with_enf, nominal_freq=50.0)
        assert signal is not None
        # Should detect something near 50Hz
        assert abs(signal.mean_frequency - 50.0) < 2.0

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_extract_enf_weak_signal(self, analyzer, temp_wav_without_enf):
        """Weak/missing ENF returns low confidence."""
        signal = analyzer.extract_enf_only(temp_wav_without_enf, nominal_freq=50.0)
        # May return signal but with low confidence
        if signal:
            assert signal.signal_strength < 0.5

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_short_recording_rejected(self, analyzer, temp_wav_short):
        """Short recordings are rejected."""
        result = analyzer.analyze(temp_wav_short)
        assert result.state == DimensionState.UNCERTAIN
        assert any("short" in e.finding.lower() for e in result.evidence)


class TestENFAnalysis:
    """Test full ENF analysis pipeline."""

    @pytest.fixture
    def temp_wav_50hz(self, tmp_path):
        """Create WAV with 50Hz ENF."""
        if not HAS_SCIPY:
            pytest.skip("scipy required")
        path = tmp_path / "50hz.wav"
        create_test_wav(str(path), duration=15.0, add_enf=True, enf_freq=50.0, enf_amplitude=0.1)
        return str(path)

    @pytest.fixture
    def temp_wav_60hz(self, tmp_path):
        """Create WAV with 60Hz ENF."""
        if not HAS_SCIPY:
            pytest.skip("scipy required")
        path = tmp_path / "60hz.wav"
        create_test_wav(str(path), duration=15.0, add_enf=True, enf_freq=60.0, enf_amplitude=0.1)
        return str(path)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_analyze_returns_dimension_result(self, analyzer, temp_wav_50hz):
        """Analysis returns DimensionResult."""
        result = analyzer.analyze(temp_wav_50hz)
        assert result.dimension == "enf"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.UNCERTAIN,
            DimensionState.VERIFIED,
            DimensionState.INCONSISTENT
        ]

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_analyze_with_region_hint(self, analyzer, temp_wav_50hz):
        """Analysis can use region hint."""
        result = analyzer.analyze(temp_wav_50hz, region=GridRegion.UK)
        assert result.dimension == "enf"

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_analyze_has_evidence(self, analyzer, temp_wav_50hz):
        """Analysis includes evidence."""
        result = analyzer.analyze(temp_wav_50hz)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_analyze_has_methodology(self, analyzer, temp_wav_50hz):
        """Analysis includes methodology."""
        result = analyzer.analyze(temp_wav_50hz)
        assert result.methodology is not None


class TestENFReference:
    """Test ENF reference data handling."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_reference_creation(self):
        """ENFReference can be created."""
        ref = ENFReference(
            region=GridRegion.UK,
            start_time="2024-01-15T00:00:00",
            frequencies=np.array([50.0, 50.01, 49.99]),
            sample_interval=1.0
        )
        assert ref.region == GridRegion.UK
        assert len(ref.frequencies) == 3

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_load_reference_csv(self, analyzer, tmp_path):
        """Reference data can be loaded from CSV."""
        csv_path = tmp_path / "ref.csv"
        csv_path.write_text("50.01\n50.02\n49.99\n50.00\n")

        ref = analyzer.load_reference_csv(
            str(csv_path),
            GridRegion.UK,
            "2024-01-15T00:00:00"
        )
        assert ref.region == GridRegion.UK
        assert len(ref.frequencies) == 4


class TestENFNoDependencies:
    """Test behavior when scipy not available."""

    def test_handles_missing_deps(self):
        """Analyzer handles missing deps."""
        analyzer = ENFAnalyzer()
        # Should not crash
        assert analyzer is not None


class TestENFIntegration:
    """Test ENF integration with main analyzer."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_wu_analyzer_with_enf(self, tmp_path):
        """WuAnalyzer can enable ENF analysis."""
        from wu import WuAnalyzer

        # Create test WAV
        wav_path = tmp_path / "test.wav"
        create_test_wav(str(wav_path), duration=10.0, add_enf=True)

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_enf=True
        )
        result = analyzer.analyze(str(wav_path))

        assert result.enf is not None
        assert result.enf.dimension == "enf"

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_wu_analyzer_enf_disabled_by_default(self, tmp_path):
        """WuAnalyzer has ENF disabled by default."""
        from wu import WuAnalyzer

        wav_path = tmp_path / "test.wav"
        create_test_wav(str(wav_path), duration=10.0)

        analyzer = WuAnalyzer()
        result = analyzer.analyze(str(wav_path))

        assert result.enf is None


class TestENFEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_stereo_audio(self, analyzer, tmp_path):
        """Stereo audio is handled."""
        # Create stereo WAV
        path = tmp_path / "stereo.wav"
        num_samples = 44100 * 10
        audio = np.random.randn(num_samples, 2) * 0.1
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

        with wave.open(str(path), 'w') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(audio_int16.tobytes())

        result = analyzer.analyze(str(path))
        assert result.dimension == "enf"

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_low_sample_rate(self, analyzer, tmp_path):
        """Low sample rate audio is handled."""
        path = tmp_path / "low_rate.wav"
        # 8kHz sample rate (telephone quality)
        create_test_wav(str(path), duration=10.0, sample_rate=8000)
        result = analyzer.analyze(str(path))
        assert result.dimension == "enf"

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
    def test_high_sample_rate(self, analyzer, tmp_path):
        """High sample rate audio is handled."""
        path = tmp_path / "high_rate.wav"
        # 96kHz sample rate
        create_test_wav(str(path), duration=10.0, sample_rate=96000)
        result = analyzer.analyze(str(path))
        assert result.dimension == "enf"
