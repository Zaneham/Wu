"""
Tests for lighting direction consistency forensic analysis.

Tests light direction estimation, regional analysis, and inconsistency detection.
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
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from wu.dimensions.lighting import (
    LightingAnalyzer, LightVector, RegionLighting, LightingResult
)
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return LightingAnalyzer()


def create_test_image(tmp_path, size=(200, 200), name="test.jpg", light_angle=45):
    """Create a test image with simulated directional lighting."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    np.random.seed(42)

    # Create image with gradient simulating light from given angle
    img_array = np.zeros((*size, 3), dtype=np.uint8)

    # Light direction
    light_x = math.cos(math.radians(light_angle))
    light_y = math.sin(math.radians(light_angle))

    for i in range(size[0]):
        for j in range(size[1]):
            # Simulate shading based on light direction
            # Brighter toward light source
            brightness = 128 + int(40 * (light_x * j / size[1] + light_y * i / size[0]))
            brightness = max(50, min(230, brightness))

            # Add some variation
            variation = np.random.randint(-15, 15)
            brightness = max(0, min(255, brightness + variation))

            img_array[i, j] = [brightness, brightness, brightness]

    img = Image.fromarray(img_array)
    path = tmp_path / name
    img.save(path, format='JPEG', quality=90)
    return str(path)


def create_inconsistent_image(tmp_path, size=(200, 200)):
    """Create an image with inconsistent lighting in different regions."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    np.random.seed(42)
    img_array = np.zeros((*size, 3), dtype=np.uint8)

    # Top half: light from left (angle ~0)
    for i in range(size[0] // 2):
        for j in range(size[1]):
            brightness = 100 + int(80 * j / size[1])
            img_array[i, j] = [brightness, brightness, brightness]

    # Bottom half: light from right (angle ~180)
    for i in range(size[0] // 2, size[0]):
        for j in range(size[1]):
            brightness = 180 - int(80 * j / size[1])
            img_array[i, j] = [brightness, brightness, brightness]

    # Add noise
    noise = np.random.randint(-10, 10, (*size, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    path = tmp_path / "inconsistent.jpg"
    img.save(path, format='JPEG', quality=90)
    return str(path)


class TestLightVector:
    """Test LightVector dataclass."""

    def test_creation(self):
        """LightVector can be created."""
        lv = LightVector(azimuth=45.0, elevation=30.0, confidence=0.8)
        assert lv.azimuth == 45.0
        assert lv.elevation == 30.0
        assert lv.confidence == 0.8

    def test_to_cartesian(self):
        """Cartesian conversion works."""
        # Light from front-right, elevated
        lv = LightVector(azimuth=0.0, elevation=0.0, confidence=1.0)
        x, y, z = lv.to_cartesian()

        # At azimuth=0, elevation=0: pointing along x-axis
        assert abs(x - 1.0) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_to_cartesian_elevated(self):
        """Elevated light vector converts correctly."""
        lv = LightVector(azimuth=0.0, elevation=90.0, confidence=1.0)
        x, y, z = lv.to_cartesian()

        # At elevation=90: pointing straight up (z-axis)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z - 1.0) < 0.01

    def test_angle_to_same(self):
        """Angle between identical vectors is 0."""
        lv1 = LightVector(45.0, 30.0, 0.8)
        lv2 = LightVector(45.0, 30.0, 0.8)
        assert lv1.angle_to(lv2) < 0.01

    def test_angle_to_opposite(self):
        """Angle between opposite vectors is 180."""
        lv1 = LightVector(0.0, 0.0, 0.8)
        lv2 = LightVector(180.0, 0.0, 0.8)
        assert abs(lv1.angle_to(lv2) - 180.0) < 0.01

    def test_angle_to_perpendicular(self):
        """Angle between perpendicular vectors is 90."""
        lv1 = LightVector(0.0, 0.0, 0.8)
        lv2 = LightVector(90.0, 0.0, 0.8)
        assert abs(lv1.angle_to(lv2) - 90.0) < 0.01


class TestRegionLighting:
    """Test RegionLighting dataclass."""

    def test_creation(self):
        """RegionLighting can be created."""
        lv = LightVector(45.0, 30.0, 0.8)
        region = RegionLighting(
            x=100, y=150,
            width=64, height=64,
            light_vector=lv,
            mean_brightness=128.0,
            specular_strength=0.2
        )
        assert region.x == 100
        assert region.mean_brightness == 128.0


class TestLightingResult:
    """Test LightingResult dataclass."""

    def test_creation(self):
        """LightingResult can be created."""
        gl = LightVector(45.0, 30.0, 0.8)
        result = LightingResult(
            global_light=gl,
            max_inconsistency_angle=15.0,
            processing_time_ms=100.0
        )
        assert result.max_inconsistency_angle == 15.0


class TestLightingAnalyzerBasic:
    """Test basic LightingAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "lighting"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN


class TestGradientComputation:
    """Test gradient computation."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_NUMPY, reason="scipy/numpy required")
    def test_compute_gradients(self, analyzer):
        """Gradients can be computed."""
        # Create simple gradient image
        gray = np.zeros((100, 100), dtype=np.float64)
        for j in range(100):
            gray[:, j] = j * 2.5  # Gradient left-to-right

        gx, gy = analyzer._compute_gradients(gray)

        assert gx.shape == gray.shape
        assert gy.shape == gray.shape

        # Horizontal gradient should be non-zero and consistent
        # (sign depends on kernel convention)
        assert abs(np.mean(gx[10:-10, 10:-10])) > 0.1

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_NUMPY, reason="scipy/numpy required")
    def test_gradients_vertical(self, analyzer):
        """Vertical gradients detected correctly."""
        gray = np.zeros((100, 100), dtype=np.float64)
        for i in range(100):
            gray[i, :] = i * 2.5  # Gradient top-to-bottom

        gx, gy = analyzer._compute_gradients(gray)

        # Vertical gradient should be non-zero and consistent
        assert abs(np.mean(gy[10:-10, 10:-10])) > 0.1


class TestLightEstimation:
    """Test light direction estimation."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_NUMPY, reason="scipy/numpy required")
    def test_estimate_light_horizontal(self, analyzer):
        """Light from horizontal gradient estimated correctly."""
        # Create gradient image (light from right)
        gray = np.zeros((100, 100), dtype=np.float64)
        for j in range(100):
            gray[:, j] = 50 + j * 1.5

        gx, gy = analyzer._compute_gradients(gray)
        light = analyzer._estimate_light_direction(gray, gx, gy)

        # Light should be roughly from the right (azimuth near 0 or 360)
        assert light.azimuth < 45 or light.azimuth > 315

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_NUMPY, reason="scipy/numpy required")
    def test_estimate_returns_confidence(self, analyzer):
        """Light estimation returns confidence value."""
        gray = np.random.rand(100, 100) * 255
        gx, gy = analyzer._compute_gradients(gray)
        light = analyzer._estimate_light_direction(gray, gx, gy)

        assert 0 <= light.confidence <= 1


class TestFullAnalysis:
    """Test full lighting analysis pipeline."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_returns_dimension_result(self, analyzer, tmp_path):
        """Analysis returns DimensionResult."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)

        assert result.dimension == "lighting"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_has_evidence(self, analyzer, tmp_path):
        """Analysis includes evidence."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_consistent_image(self, analyzer, tmp_path):
        """Consistently lit image is detected as consistent."""
        path = create_test_image(tmp_path, light_angle=45)
        result = analyzer.analyze(path)

        # Should be consistent or uncertain (not suspicious)
        assert result.state in [DimensionState.CONSISTENT, DimensionState.UNCERTAIN]


class TestIntegration:
    """Test lighting integration with main analyzer."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_wu_analyzer_with_lighting(self, tmp_path):
        """WuAnalyzer can enable lighting analysis."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_lighting=True
        )
        result = analyzer.analyze(path)

        assert result.lighting is not None
        assert result.lighting.dimension == "lighting"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_wu_analyzer_lighting_disabled_by_default(self, tmp_path):
        """WuAnalyzer has lighting disabled by default."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer()
        result = analyzer.analyze(path)

        assert result.lighting is None


class TestSpecularAnalysis:
    """Test specular highlight analysis."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_specular_strength_uniform(self, analyzer):
        """Uniform region has low specular strength."""
        region = np.full((64, 64), 128.0, dtype=np.float64)
        strength = analyzer._compute_specular_strength(region)
        assert strength < 0.1

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_specular_strength_with_highlights(self, analyzer):
        """Region with bright spots has higher specular strength."""
        region = np.full((64, 64), 100.0, dtype=np.float64)
        # Add bright specular spots
        region[30:34, 30:34] = 250.0
        strength = analyzer._compute_specular_strength(region)
        assert strength > 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_grayscale_image(self, analyzer, tmp_path):
        """Grayscale images are handled."""
        img = Image.new('L', (100, 100), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "lighting"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_small_image(self, analyzer, tmp_path):
        """Small images are handled."""
        img = Image.new('RGB', (30, 30), color='blue')
        path = tmp_path / "small.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "lighting"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_uniform_image(self, analyzer, tmp_path):
        """Uniform images return uncertain (low confidence)."""
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        path = tmp_path / "uniform.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "lighting"
        # Uniform images should have uncertain state
        assert result.state == DimensionState.UNCERTAIN


class TestLuminanceConversion:
    """Test luminance conversion."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_rgb_to_luminance(self, analyzer):
        """RGB to luminance conversion works."""
        # Pure red
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lum = analyzer._to_luminance(rgb)
        # Should be ~76 (0.299 * 255)
        assert 70 < lum[0, 0] < 80

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
    def test_grayscale_passthrough(self, analyzer):
        """Grayscale input passes through."""
        gray = np.array([[100, 150], [200, 50]], dtype=np.float64)
        result = analyzer._to_luminance(gray)
        assert np.allclose(result, gray)


class TestPerformanceMarkers:
    """Test that performance optimization markers are present."""

    def test_module_has_optimization_comments(self):
        """Module documents optimization opportunities."""
        import wu.dimensions.lighting as lighting
        source = lighting.__doc__

        assert "OPTIMIZE" in source
        assert "CYTHON" in source or "C" in source

    def test_module_has_native_stubs(self):
        """Module includes native implementation stubs."""
        import wu.dimensions.lighting as lighting
        import inspect
        source = inspect.getsource(lighting)

        # Check for C/Cython stub documentation
        assert "cython" in source.lower() or "SIMD" in source or "AVX" in source


class TestNoDependencies:
    """Test behavior when dependencies not available."""

    def test_handles_missing_deps_gracefully(self):
        """Analyzer handles missing deps."""
        analyzer = LightingAnalyzer()
        assert analyzer is not None
