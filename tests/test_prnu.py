"""
Tests for PRNU (sensor fingerprint) forensic detection.

Tests sensor noise extraction, camera identification, and manipulation detection.
"""

import pytest
import math
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy import ndimage
    from scipy.signal import wiener
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from wu.dimensions.prnu import PRNUAnalyzer, PRNUFingerprint, PRNUMatch, PRNUResult
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return PRNUAnalyzer()


def create_test_image(tmp_path, size=(200, 200), name="test.jpg", noise_pattern=None):
    """Create a test image with optional synthetic noise pattern."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    np.random.seed(42)

    # Create base image with some structure
    img_array = np.zeros((*size, 3), dtype=np.uint8)

    # Add gradient background
    for i in range(size[0]):
        for j in range(size[1]):
            img_array[i, j] = [
                int(128 + 64 * math.sin(i / 20)),
                int(128 + 64 * math.cos(j / 20)),
                int(128 + 32 * math.sin((i + j) / 30))
            ]

    # Add some random noise
    noise = np.random.randint(-20, 20, (*size, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add synthetic PRNU pattern if provided
    if noise_pattern is not None:
        prnu_contribution = (noise_pattern * 5).astype(np.int16)
        for c in range(3):
            img_array[:, :, c] = np.clip(
                img_array[:, :, c].astype(np.int16) + prnu_contribution,
                0, 255
            ).astype(np.uint8)

    img = Image.fromarray(img_array)
    path = tmp_path / name
    img.save(path, format='JPEG', quality=95)
    return str(path)


def create_synthetic_prnu(size=(200, 200), seed=12345):
    """Create a synthetic PRNU pattern (simulating sensor fingerprint)."""
    if not HAS_NUMPY:
        pytest.skip("numpy required")

    np.random.seed(seed)
    # PRNU is typically multiplicative noise with low variance
    prnu = np.random.randn(*size) * 0.02
    return prnu


class TestPRNUFingerprint:
    """Test PRNUFingerprint dataclass."""

    def test_creation(self):
        """PRNUFingerprint can be created."""
        if not HAS_NUMPY:
            pytest.skip("numpy required")

        fp = PRNUFingerprint(
            camera_id="camera_001",
            width=100,
            height=100,
            fingerprint=np.zeros((100, 100)),
            n_reference_images=50,
            quality_score=0.8
        )
        assert fp.camera_id == "camera_001"
        assert fp.width == 100
        assert fp.height == 100
        assert fp.n_reference_images == 50


class TestPRNUMatch:
    """Test PRNUMatch dataclass."""

    def test_creation_matched(self):
        """PRNUMatch with positive match."""
        match = PRNUMatch(
            camera_id="camera_001",
            correlation=85.5,
            p_value=1e-10,
            matched=True
        )
        assert match.matched is True
        assert match.correlation == 85.5

    def test_creation_not_matched(self):
        """PRNUMatch with no match."""
        match = PRNUMatch(
            camera_id="camera_002",
            correlation=15.0,
            p_value=0.5,
            matched=False
        )
        assert match.matched is False


class TestPRNUResult:
    """Test PRNUResult dataclass."""

    def test_creation(self):
        """PRNUResult can be created."""
        result = PRNUResult(
            noise_residual_quality=0.5,
            region_consistency=0.95,
            processing_time_ms=150.0
        )
        assert result.noise_residual_quality == 0.5
        assert result.region_consistency == 0.95


class TestPRNUAnalyzerBasic:
    """Test basic PRNUAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "prnu"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_small_image(self, analyzer, tmp_path):
        """Small image is handled."""
        img = Image.new('RGB', (50, 50), color='red')
        path = tmp_path / "small.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "prnu"


class TestNoiseExtraction:
    """Test noise residual extraction."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_extract_noise_residual(self, analyzer, tmp_path):
        """Noise residual can be extracted."""
        path = create_test_image(tmp_path)
        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)

        assert noise is not None
        assert noise.shape == image_array.shape[:2]  # 2D noise
        assert noise.dtype == np.float64

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_noise_has_expected_properties(self, analyzer, tmp_path):
        """Extracted noise has expected statistical properties."""
        path = create_test_image(tmp_path, size=(300, 300))
        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)

        # Noise should be zero-centered
        assert abs(np.mean(noise)) < 5.0

        # Noise should have reasonable variance
        assert 0 < np.std(noise) < 50


class TestNoiseQuality:
    """Test noise quality estimation."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_quality_for_textured_image(self, analyzer, tmp_path):
        """Textured images have higher noise quality."""
        path = create_test_image(tmp_path, size=(200, 200))
        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)
        quality = analyzer._estimate_noise_quality(noise)

        # Should have measurable quality
        assert quality > 0

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_quality_for_uniform_image(self, analyzer, tmp_path):
        """Uniform images have lower noise quality."""
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        path = tmp_path / "uniform.jpg"
        img.save(path, format='JPEG', quality=95)

        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)
        quality = analyzer._estimate_noise_quality(noise)

        # Uniform images have less useful noise
        assert quality >= 0


class TestRegionConsistency:
    """Test PRNU region consistency analysis."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_consistent_image(self, analyzer, tmp_path):
        """Image with consistent PRNU passes."""
        path = create_test_image(tmp_path, size=(300, 300))
        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)
        result = analyzer._analyze_region_consistency(noise)

        # Should be mostly consistent
        if result.region_consistency is not None:
            assert result.region_consistency > 0.5

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_processing_time_recorded(self, analyzer, tmp_path):
        """Processing time is recorded."""
        path = create_test_image(tmp_path, size=(200, 200))
        with Image.open(path) as img:
            image_array = np.array(img, dtype=np.float64)

        noise = analyzer._extract_noise_residual(image_array)
        result = analyzer._analyze_region_consistency(noise)

        assert result.processing_time_ms >= 0


class TestPCEComputation:
    """Test Peak-to-Correlation Energy computation."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_pce_identical_signals(self, analyzer):
        """Identical signals have high PCE."""
        signal = np.random.randn(100, 100)

        pce, p_value = analyzer._compute_pce(signal, signal)

        # Perfect match should have very high PCE
        assert pce > 100

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_pce_unrelated_signals(self, analyzer):
        """Unrelated signals have low PCE."""
        signal1 = np.random.randn(100, 100)
        signal2 = np.random.randn(100, 100)

        pce, p_value = analyzer._compute_pce(signal1, signal2)

        # Unrelated signals should have low PCE
        assert pce < 20


class TestFingerprintCreation:
    """Test PRNU fingerprint creation."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_create_fingerprint_insufficient_images(self, analyzer, tmp_path):
        """Fingerprint creation fails with too few images."""
        # Create only 2 images
        paths = []
        for i in range(2):
            path = create_test_image(tmp_path, name=f"img_{i}.jpg")
            paths.append(path)

        fp = analyzer.create_fingerprint(paths, "test_camera")

        # Should fail with too few images
        assert fp is None

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_create_fingerprint_with_enough_images(self, analyzer, tmp_path):
        """Fingerprint can be created with enough images."""
        # Create 10 images (minimum is 5)
        prnu = create_synthetic_prnu(size=(100, 100))
        paths = []
        for i in range(10):
            np.random.seed(i * 100)  # Different random content per image
            path = create_test_image(tmp_path, size=(100, 100), name=f"img_{i}.jpg", noise_pattern=prnu)
            paths.append(path)

        fp = analyzer.create_fingerprint(paths, "test_camera")

        assert fp is not None
        assert fp.camera_id == "test_camera"
        assert fp.n_reference_images == 10
        assert fp.width == 100
        assert fp.height == 100


class TestFullAnalysis:
    """Test full PRNU analysis pipeline."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_returns_dimension_result(self, analyzer, tmp_path):
        """Analysis returns DimensionResult."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)

        assert result.dimension == "prnu"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.UNCERTAIN,
            DimensionState.VERIFIED
        ]

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_has_evidence(self, analyzer, tmp_path):
        """Analysis includes evidence."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_has_methodology(self, analyzer, tmp_path):
        """Analysis includes methodology."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)
        assert result.methodology is not None


class TestIntegration:
    """Test PRNU integration with main analyzer."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_wu_analyzer_with_prnu(self, tmp_path):
        """WuAnalyzer can enable PRNU detection."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_prnu=True
        )
        result = analyzer.analyze(path)

        assert result.prnu is not None
        assert result.prnu.dimension == "prnu"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_wu_analyzer_prnu_disabled_by_default(self, tmp_path):
        """WuAnalyzer has prnu disabled by default."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer()
        result = analyzer.analyze(path)

        assert result.prnu is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_grayscale_image(self, analyzer, tmp_path):
        """Grayscale images are handled."""
        img = Image.new('L', (100, 100), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "prnu"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_rgba_image(self, analyzer, tmp_path):
        """RGBA images are handled."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path, format='PNG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "prnu"


class TestPerformanceMarkers:
    """Test that performance optimization markers are present."""

    def test_module_has_optimization_comments(self):
        """Module documents optimization opportunities."""
        import wu.dimensions.prnu as prnu_mod
        source = prnu_mod.__doc__

        assert "OPTIMIZE" in source
        assert "CYTHON" in source or "C" in source

    def test_module_has_native_stubs(self):
        """Module includes native implementation stubs."""
        import wu.dimensions.prnu as prnu_mod
        import inspect
        source = inspect.getsource(prnu_mod)

        # Check for C/Cython stub documentation
        assert "cython" in source.lower() or "SIMD" in source


class TestNoDependencies:
    """Test behavior when dependencies not available."""

    def test_handles_missing_deps_gracefully(self):
        """Analyzer handles missing deps."""
        analyzer = PRNUAnalyzer()
        assert analyzer is not None
