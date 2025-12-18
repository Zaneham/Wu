"""
Tests for EXIF thumbnail forensic analysis.

Tests thumbnail extraction, comparison, and mismatch detection.
"""

import pytest
import tempfile
from pathlib import Path
from io import BytesIO

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

from wu.dimensions.thumbnail import (
    ThumbnailAnalyzer,
    ThumbnailComparison,
)
from wu.state import DimensionState, Confidence


pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PIL,
    reason="numpy and PIL required for thumbnail tests"
)


class TestThumbnailComparison:
    """Tests for ThumbnailComparison dataclass."""

    def test_creation(self):
        comp = ThumbnailComparison(
            has_thumbnail=True,
            thumbnail_size=(160, 120),
            main_size=(1920, 1080),
            similarity=0.95,
            mse=0.01,
            significant_difference=False
        )
        assert comp.has_thumbnail is True
        assert comp.similarity == 0.95

    def test_no_thumbnail(self):
        comp = ThumbnailComparison(has_thumbnail=False)
        assert comp.has_thumbnail is False


class TestThumbnailAnalyzerBasic:
    """Basic tests for ThumbnailAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = ThumbnailAnalyzer()
        assert analyzer is not None
        assert analyzer.SIMILARITY_THRESHOLD == 0.85

    def test_analyze_missing_file(self):
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze("/nonexistent/photo.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "thumbnail"

    def test_analyze_invalid_file(self, tmp_path):
        invalid_file = tmp_path / "not_image.txt"
        invalid_file.write_text("This is not an image")

        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(str(invalid_file))
        assert result.state == DimensionState.UNCERTAIN


@pytest.fixture
def jpeg_with_thumbnail(tmp_path):
    """Create a JPEG image with embedded EXIF thumbnail."""
    # Create main image
    main_img = Image.new('RGB', (800, 600), color='blue')

    # Create thumbnail
    thumb = main_img.copy()
    thumb.thumbnail((160, 120))

    # Save with EXIF (PIL doesn't easily add thumbnails, so we'll test without)
    file_path = tmp_path / "with_thumb.jpg"
    main_img.save(str(file_path), "JPEG", quality=85)

    return str(file_path)


@pytest.fixture
def jpeg_without_thumbnail(tmp_path):
    """Create a JPEG image without EXIF thumbnail."""
    img = Image.new('RGB', (800, 600), color='red')
    file_path = tmp_path / "no_thumb.jpg"
    img.save(str(file_path), "JPEG", quality=85)
    return str(file_path)


@pytest.fixture
def png_image(tmp_path):
    """Create a PNG image (never has EXIF thumbnail)."""
    img = Image.new('RGB', (800, 600), color='green')
    file_path = tmp_path / "test.png"
    img.save(str(file_path), "PNG")
    return str(file_path)


class TestThumbnailExtraction:
    """Tests for thumbnail extraction."""

    def test_no_thumbnail_in_new_jpeg(self, jpeg_without_thumbnail):
        """New JPEG without EXIF should have no thumbnail."""
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(jpeg_without_thumbnail)

        # Should report no thumbnail found
        assert result.dimension == "thumbnail"
        # No thumbnail = uncertain (can't determine manipulation)
        assert result.state == DimensionState.UNCERTAIN

    def test_no_thumbnail_in_png(self, png_image):
        """PNG images don't have EXIF thumbnails."""
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(png_image)

        assert result.state == DimensionState.UNCERTAIN
        assert any("No EXIF thumbnail" in e.finding for e in result.evidence)


class TestThumbnailComparisonAlgorithm:
    """Tests for thumbnail comparison algorithm."""

    def test_ssim_identical_images(self):
        """SSIM of identical images should be 1.0."""
        analyzer = ThumbnailAnalyzer()

        img = np.random.rand(100, 100, 3)
        ssim = analyzer._compute_ssim(img, img)

        assert ssim > 0.99

    def test_ssim_different_images(self):
        """SSIM of different images should be low."""
        analyzer = ThumbnailAnalyzer()

        img1 = np.zeros((100, 100, 3))
        img2 = np.ones((100, 100, 3))
        ssim = analyzer._compute_ssim(img1, img2)

        assert ssim < 0.5

    def test_comparison_identical(self, tmp_path):
        """Compare image to itself (via thumbnail) should be similar."""
        # Create simple test image
        img = Image.new('RGB', (800, 600), color='blue')
        thumb = img.copy()
        thumb.thumbnail((160, 120))

        analyzer = ThumbnailAnalyzer()
        comparison = analyzer._compare_images(thumb, img)

        assert comparison.has_thumbnail is True
        assert comparison.similarity > 0.9
        assert bool(comparison.significant_difference) is False


class TestFullAnalysis:
    """Tests for complete thumbnail analysis."""

    def test_analyze_returns_dimension_result(self, jpeg_without_thumbnail):
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(jpeg_without_thumbnail)

        assert result.dimension == "thumbnail"
        assert result.state is not None
        assert result.confidence is not None

    def test_analyze_has_evidence(self, jpeg_without_thumbnail):
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(jpeg_without_thumbnail)

        assert len(result.evidence) > 0

    def test_analyze_has_methodology(self, jpeg_without_thumbnail):
        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(jpeg_without_thumbnail)

        # Should have methodology even when no thumbnail
        assert result.methodology is not None


class TestIntegration:
    """Integration tests with main WuAnalyzer."""

    def test_wu_analyzer_with_thumbnail(self, jpeg_without_thumbnail):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_thumbnail=True
        )
        result = analyzer.analyze(jpeg_without_thumbnail)

        assert result.thumbnail is not None
        assert result.thumbnail.dimension == "thumbnail"

    def test_wu_analyzer_thumbnail_disabled_by_default(self, jpeg_without_thumbnail):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer()
        result = analyzer.analyze(jpeg_without_thumbnail)

        # Thumbnail should be disabled by default
        assert result.thumbnail is None


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_image(self, tmp_path):
        """Very small images should be handled."""
        img = Image.new('RGB', (32, 32), color='red')
        file_path = tmp_path / "tiny.jpg"
        img.save(str(file_path), "JPEG")

        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(str(file_path))

        assert result.dimension == "thumbnail"

    def test_grayscale_image(self, tmp_path):
        """Grayscale images should be handled."""
        img = Image.new('L', (400, 300), color=128)
        file_path = tmp_path / "gray.jpg"
        img.save(str(file_path), "JPEG")

        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(str(file_path))

        assert result.dimension == "thumbnail"


class TestNoDependencies:
    """Test graceful handling when dependencies are missing."""

    def test_handles_missing_deps_gracefully(self, monkeypatch, tmp_path):
        """Should return UNCERTAIN when deps missing."""
        img = Image.new('RGB', (200, 200), color='blue')
        file_path = tmp_path / "test.jpg"
        img.save(str(file_path), "JPEG")

        import wu.dimensions.thumbnail as thumb_module
        monkeypatch.setattr(thumb_module, "HAS_PIL", False)

        analyzer = ThumbnailAnalyzer()
        result = analyzer.analyze(str(file_path))

        assert result.state == DimensionState.UNCERTAIN
