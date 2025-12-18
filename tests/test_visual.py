"""
Tests for visual forensic analysis.

Tests JPEG artifact analysis, Error Level Analysis,
and manipulation detection.
"""

import io
import pytest
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from wu.dimensions.visual import VisualAnalyzer, ELAResult, QuantizationAnalysis
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return VisualAnalyzer()


@pytest.fixture
def temp_jpeg(tmp_path):
    """Create a temporary JPEG image."""
    if not HAS_DEPS:
        pytest.skip("PIL/numpy required")

    img = Image.new('RGB', (100, 100), color='red')
    path = tmp_path / "test.jpg"
    img.save(path, format='JPEG', quality=85)
    return str(path)


@pytest.fixture
def temp_png(tmp_path):
    """Create a temporary PNG image."""
    if not HAS_DEPS:
        pytest.skip("PIL/numpy required")

    img = Image.new('RGB', (100, 100), color='blue')
    path = tmp_path / "test.png"
    img.save(path, format='PNG')
    return str(path)


class TestVisualAnalyzerBasic:
    """Test basic VisualAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "visual"

    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_analyze_jpeg(self, analyzer, temp_jpeg):
        """JPEG analysis returns result."""
        result = analyzer.analyze(temp_jpeg)
        assert result.dimension == "visual"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.INCONSISTENT,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_analyze_png(self, analyzer, temp_png):
        """PNG analysis returns result."""
        result = analyzer.analyze(temp_png)
        assert result.dimension == "visual"
        # PNG doesn't have JPEG-specific artifacts
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_result_has_evidence(self, analyzer, temp_jpeg):
        """Analysis result has evidence."""
        result = analyzer.analyze(temp_jpeg)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_result_has_methodology(self, analyzer, temp_jpeg):
        """Analysis result has methodology."""
        result = analyzer.analyze(temp_jpeg)
        assert result.methodology is not None
        assert "ELA" in result.methodology


class TestELAAnalysis:
    """Test Error Level Analysis functionality."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_perform_ela_returns_result(self, analyzer, temp_jpeg):
        """ELA returns ELAResult."""
        result = analyzer._perform_ela(temp_jpeg)
        assert result is not None
        assert isinstance(result, ELAResult)

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_ela_has_statistics(self, analyzer, temp_jpeg):
        """ELA result has all statistics."""
        result = analyzer._perform_ela(temp_jpeg)
        assert result.max_difference >= 0
        assert result.mean_difference >= 0
        assert result.std_difference >= 0
        assert result.hotspot_count >= 0

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_ela_clean_image_low_difference(self, tmp_path, analyzer):
        """Clean image has relatively low ELA difference."""
        # Create simple solid color image
        img = Image.new('RGB', (200, 200), color=(128, 128, 128))
        path = tmp_path / "solid.jpg"
        img.save(path, format='JPEG', quality=95)

        result = analyzer._perform_ela(str(path))
        # Solid colors should have low ELA difference
        assert result.mean_difference < 20

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_ela_visualization(self, analyzer, temp_jpeg):
        """ELA visualization can be generated."""
        vis = analyzer.get_ela_visualization(temp_jpeg)
        assert vis is not None
        assert isinstance(vis, Image.Image)
        assert vis.mode == 'RGB'

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_ela_missing_file(self, analyzer):
        """ELA returns None for missing file."""
        result = analyzer._perform_ela("/nonexistent.jpg")
        assert result is None


class TestJPEGAnalysis:
    """Test JPEG-specific analysis."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_analyze_jpeg_returns_evidence(self, analyzer, temp_jpeg):
        """JPEG analysis can return evidence."""
        evidence = analyzer._analyze_jpeg(temp_jpeg)
        # May or may not find issues in a simple test image
        assert isinstance(evidence, list)

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_estimate_jpeg_quality(self, analyzer):
        """Quality estimation works."""
        # Standard table DC coefficient at quality 50 is 16
        assert analyzer._estimate_jpeg_quality([16] + [0]*63) == 50
        # Lower DC = higher quality
        assert analyzer._estimate_jpeg_quality([1] + [0]*63) == 100
        assert analyzer._estimate_jpeg_quality([32] + [0]*63) == 25

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_estimate_quality_invalid_table(self, analyzer):
        """Invalid table returns 0."""
        assert analyzer._estimate_jpeg_quality([]) == 0
        assert analyzer._estimate_jpeg_quality(None) == 0

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_is_standard_table(self, analyzer):
        """Standard table detection works."""
        # The exact standard table should be detected
        assert analyzer._is_standard_table(analyzer.STANDARD_LUMINANCE_QT)
        # Completely different table should not be standard
        assert not analyzer._is_standard_table([1]*64)


class TestQuantizationAnalysis:
    """Test quantization table analysis."""

    def test_quantization_analysis_creation(self):
        """QuantizationAnalysis dataclass works."""
        qa = QuantizationAnalysis(
            estimated_quality=85,
            is_double_compressed=False,
            quality_mismatch=False,
            standard_tables=True
        )
        assert qa.estimated_quality == 85
        assert qa.is_double_compressed is False

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_analyze_quantization_tables(self, analyzer):
        """Quantization table analysis works."""
        # Mock quantization tables
        qt = {
            0: [16]*64,  # Luminance
            1: [16]*64   # Chrominance
        }
        result = analyzer._analyze_quantization_tables(qt)
        assert isinstance(result, QuantizationAnalysis)
        assert result.estimated_quality >= 0
        assert result.estimated_quality <= 100


class TestELAResult:
    """Test ELAResult dataclass."""

    def test_ela_result_creation(self):
        """ELAResult can be created."""
        result = ELAResult(
            max_difference=50.0,
            mean_difference=10.0,
            std_difference=5.0,
            hotspot_count=3
        )
        assert result.max_difference == 50.0
        assert result.hotspot_count == 3

    def test_ela_result_default_regions(self):
        """ELAResult has default empty regions."""
        result = ELAResult(
            max_difference=50.0,
            mean_difference=10.0,
            std_difference=5.0,
            hotspot_count=0
        )
        assert result.hotspot_regions == []


class TestELAInterpretation:
    """Test ELA result interpretation."""

    @pytest.fixture
    def analyzer(self):
        return VisualAnalyzer()

    def test_interpret_clean_ela(self, analyzer):
        """Clean ELA produces no evidence."""
        ela = ELAResult(
            max_difference=5.0,
            mean_difference=2.0,
            std_difference=1.0,
            hotspot_count=0
        )
        evidence = analyzer._interpret_ela(ela)
        assert evidence is None

    def test_interpret_suspicious_ela(self, analyzer):
        """Suspicious ELA produces evidence."""
        ela = ELAResult(
            max_difference=100.0,
            mean_difference=8.0,
            std_difference=15.0,
            hotspot_count=6
        )
        evidence = analyzer._interpret_ela(ela)
        assert evidence is not None
        assert "ELA" in evidence.finding

    def test_interpret_high_mean_difference(self, analyzer):
        """High mean difference produces evidence."""
        ela = ELAResult(
            max_difference=200.0,
            mean_difference=25.0,  # > ELA_SUSPICIOUS_MEAN * 2
            std_difference=10.0,
            hotspot_count=2
        )
        evidence = analyzer._interpret_ela(ela)
        assert evidence is not None
        assert "high" in evidence.finding.lower() or "uniform" in evidence.finding.lower()


class TestThumbnailConsistency:
    """Test thumbnail consistency checking."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_thumbnail_check_no_crash(self, analyzer, temp_jpeg):
        """Thumbnail check doesn't crash."""
        result = analyzer._check_thumbnail_consistency(temp_jpeg)
        # May return None if no thumbnail
        assert result is None or hasattr(result, 'finding')


class TestDoubleCompression:
    """Test double compression detection."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_detect_double_compression(self, analyzer):
        """Double compression detection works."""
        # Very uniform table (all same values) suggests synthetic/edited image
        uniform_qt = [10]*64
        assert analyzer._detect_double_compression(uniform_qt) is True
        # Standard JPEG quantization table pattern should not trigger
        standard_qt = [16, 11, 10, 16, 24, 40, 51, 61] + [12]*56  # Varies naturally
        assert analyzer._detect_double_compression(standard_qt) is False

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_detect_double_compression_empty(self, analyzer):
        """Empty table returns False."""
        assert analyzer._detect_double_compression([]) is False
        assert analyzer._detect_double_compression(None) is False


class TestHotspotCounting:
    """Test hotspot region counting."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_count_no_hotspots(self, analyzer):
        """No hotspots when all pixels are below threshold."""
        mask = np.zeros((100, 100), dtype=bool)
        count = analyzer._count_hotspot_regions(mask)
        assert count == 0

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_count_few_hotspots(self, analyzer):
        """Few hotspots detected."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:15, 10:15] = True  # Small region (25 pixels / 10000 = 0.25%)
        count = analyzer._count_hotspot_regions(mask)
        assert count == 0  # < 1%

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_count_moderate_hotspots(self, analyzer):
        """Moderate hotspots detected."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:30, 0:30] = True  # 9% of pixels
        count = analyzer._count_hotspot_regions(mask)
        assert count > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_grayscale_image(self, analyzer, tmp_path):
        """Grayscale images are handled."""
        img = Image.new('L', (100, 100), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path, format='JPEG', quality=85)

        result = analyzer.analyze(str(path))
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.INCONSISTENT,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_rgba_image(self, analyzer, tmp_path):
        """RGBA images are handled."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path, format='PNG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "visual"

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_very_small_image(self, analyzer, tmp_path):
        """Very small images are handled."""
        img = Image.new('RGB', (5, 5), color='green')
        path = tmp_path / "tiny.jpg"
        img.save(path, format='JPEG', quality=85)

        result = analyzer.analyze(str(path))
        assert result.dimension == "visual"

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_large_image(self, analyzer, tmp_path):
        """Large images are handled."""
        img = Image.new('RGB', (2000, 2000), color='yellow')
        path = tmp_path / "large.jpg"
        img.save(path, format='JPEG', quality=85)

        result = analyzer.analyze(str(path))
        assert result.dimension == "visual"


class TestVisualIntegration:
    """Integration tests for visual analysis."""

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_recompressed_image_detection(self, analyzer, tmp_path):
        """Recompressed images may be flagged."""
        # Create original
        img = Image.new('RGB', (200, 200))
        # Add some texture
        pixels = img.load()
        for i in range(200):
            for j in range(200):
                pixels[i, j] = ((i*j) % 256, (i+j) % 256, (i-j) % 256)

        # Save at high quality
        path1 = tmp_path / "original.jpg"
        img.save(path1, format='JPEG', quality=95)

        # Reload and save at different quality
        with Image.open(path1) as img2:
            path2 = tmp_path / "recompressed.jpg"
            img2.save(path2, format='JPEG', quality=60)

        # Analyze both
        result1 = analyzer.analyze(str(path1))
        result2 = analyzer.analyze(str(path2))

        # Both should return valid results
        assert result1.dimension == "visual"
        assert result2.dimension == "visual"

    @pytest.mark.skipif(not HAS_DEPS, reason="PIL/numpy required")
    def test_modified_region_ela(self, analyzer, tmp_path):
        """Modified region may show in ELA."""
        # Create image with one area modified differently
        img = Image.new('RGB', (200, 200), color=(100, 100, 100))
        path = tmp_path / "modified.jpg"

        # Save original
        img.save(path, format='JPEG', quality=90)

        # Reload, modify a region, save again
        with Image.open(path) as img2:
            img2 = img2.convert('RGB')
            pixels = img2.load()
            # Modify a region
            for i in range(50, 100):
                for j in range(50, 100):
                    pixels[i, j] = (200, 50, 50)
            path2 = tmp_path / "with_modification.jpg"
            img2.save(path2, format='JPEG', quality=90)

        result = analyzer.analyze(str(path2))
        assert result.dimension == "visual"
        # The modification may or may not be detected depending on specifics


class TestNoDependencies:
    """Test behavior when PIL/numpy not available."""

    def test_handles_missing_deps_gracefully(self):
        """Analyzer handles missing deps."""
        # This test documents expected behavior
        # When deps are missing, analyze should return UNCERTAIN
        analyzer = VisualAnalyzer()
        # If deps are available, this passes
        # If deps are missing, analyze() should not crash
        assert analyzer is not None
