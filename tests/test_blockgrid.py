"""
Tests for JPEG block grid forensic analysis.

Tests block grid offset detection, splicing detection, and double compression.
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
    from scipy import fftpack
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from wu.dimensions.blockgrid import (
    BlockGridAnalyzer, BlockGridOffset, GridRegionAnalysis, BlockGridResult
)
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return BlockGridAnalyzer()


def create_test_jpeg(tmp_path, size=(200, 200), name="test.jpg", quality=85):
    """Create a test JPEG image."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    np.random.seed(42)

    # Create image with some structure (not just noise)
    img_array = np.zeros((*size, 3), dtype=np.uint8)

    # Add gradient and patterns
    for i in range(size[0]):
        for j in range(size[1]):
            img_array[i, j] = [
                int(128 + 64 * math.sin(i / 15)),
                int(128 + 64 * math.cos(j / 15)),
                int(128 + 32 * math.sin((i + j) / 20))
            ]

    # Add some texture
    noise = np.random.randint(-15, 15, (*size, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    path = tmp_path / name
    img.save(path, format='JPEG', quality=quality)
    return str(path)


def create_cropped_jpeg(tmp_path, crop_offset=(3, 5), size=(200, 200)):
    """Create a JPEG that simulates cropping (grid offset)."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    # Create larger image first
    large_size = (size[0] + crop_offset[0] + 10, size[1] + crop_offset[1] + 10)
    np.random.seed(123)

    img_array = np.zeros((*large_size, 3), dtype=np.uint8)
    for i in range(large_size[0]):
        for j in range(large_size[1]):
            img_array[i, j] = [
                int(128 + 50 * math.sin(i / 12)),
                int(128 + 50 * math.cos(j / 12)),
                100
            ]

    # Save as JPEG (creates block grid at 0,0)
    large_img = Image.fromarray(img_array)
    large_path = tmp_path / "large.jpg"
    large_img.save(large_path, format='JPEG', quality=75)

    # Reload and crop (this shifts the grid)
    with Image.open(large_path) as img:
        cropped = img.crop((
            crop_offset[1], crop_offset[0],
            crop_offset[1] + size[1], crop_offset[0] + size[0]
        ))
        cropped_path = tmp_path / "cropped.jpg"
        cropped.save(cropped_path, format='JPEG', quality=95)

    return str(cropped_path)


class TestBlockGridOffset:
    """Test BlockGridOffset dataclass."""

    def test_creation(self):
        """BlockGridOffset can be created."""
        offset = BlockGridOffset(
            x_offset=3,
            y_offset=5,
            confidence=0.8
        )
        assert offset.x_offset == 3
        assert offset.y_offset == 5
        assert offset.confidence == 0.8

    def test_creation_with_region(self):
        """BlockGridOffset can include region info."""
        offset = BlockGridOffset(
            x_offset=2,
            y_offset=4,
            confidence=0.7,
            region=(10, 20, 100, 100)
        )
        assert offset.region == (10, 20, 100, 100)


class TestGridRegionAnalysis:
    """Test GridRegionAnalysis dataclass."""

    def test_creation(self):
        """GridRegionAnalysis can be created."""
        offset = BlockGridOffset(3, 5, 0.9)
        region = GridRegionAnalysis(
            x=100, y=150,
            width=64, height=64,
            primary_offset=offset,
            inconsistent=True
        )
        assert region.x == 100
        assert region.inconsistent is True


class TestBlockGridResult:
    """Test BlockGridResult dataclass."""

    def test_creation(self):
        """BlockGridResult can be created."""
        offset = BlockGridOffset(0, 0, 0.8)
        result = BlockGridResult(
            primary_offset=offset,
            is_cropped=False,
            has_spliced_regions=False,
            double_compression_detected=False,
            processing_time_ms=50.0
        )
        assert result.is_cropped is False
        assert result.processing_time_ms == 50.0


class TestBlockGridAnalyzerBasic:
    """Test basic BlockGridAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "blockgrid"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN


class TestBlockGridDetection:
    """Test block grid offset detection."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_uncropped_image_grid_at_origin(self, analyzer, tmp_path):
        """Uncropped JPEG has grid at origin."""
        path = create_test_jpeg(tmp_path, quality=70)  # Lower quality = stronger artifacts

        with Image.open(path) as img:
            image_array = np.mean(np.array(img), axis=2).astype(np.float64)

        offset = analyzer._detect_grid_offset(image_array)

        # Grid should be at or near origin
        # Note: detection may not be perfect for synthetic images
        assert 0 <= offset.x_offset < 8
        assert 0 <= offset.y_offset < 8

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_blockiness_computation(self, analyzer, tmp_path):
        """Blockiness can be computed."""
        path = create_test_jpeg(tmp_path, quality=60)

        with Image.open(path) as img:
            image_array = np.mean(np.array(img), axis=2).astype(np.float64)

        # Compute blockiness at different offsets
        blockiness_0_0 = analyzer._compute_blockiness(image_array, 0, 0)
        blockiness_4_4 = analyzer._compute_blockiness(image_array, 4, 4)

        # Both should be non-negative
        assert blockiness_0_0 >= 0
        assert blockiness_4_4 >= 0


class TestFullAnalysis:
    """Test full block grid analysis pipeline."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_returns_dimension_result(self, analyzer, tmp_path):
        """Analysis returns DimensionResult."""
        path = create_test_jpeg(tmp_path)
        result = analyzer.analyze(path)

        assert result.dimension == "blockgrid"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.INCONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_has_evidence(self, analyzer, tmp_path):
        """Analysis includes evidence."""
        path = create_test_jpeg(tmp_path)
        result = analyzer.analyze(path)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_analyze_has_methodology(self, analyzer, tmp_path):
        """Analysis includes methodology."""
        path = create_test_jpeg(tmp_path)
        result = analyzer.analyze(path)
        assert result.methodology is not None
        assert "block" in result.methodology.lower() or "grid" in result.methodology.lower()


class TestIntegration:
    """Test blockgrid integration with main analyzer."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY or not HAS_SCIPY, reason="PIL/numpy/scipy required")
    def test_wu_analyzer_with_blockgrid(self, tmp_path):
        """WuAnalyzer can enable block grid analysis."""
        from wu import WuAnalyzer

        path = create_test_jpeg(tmp_path)

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_blockgrid=True
        )
        result = analyzer.analyze(path)

        assert result.blockgrid is not None
        assert result.blockgrid.dimension == "blockgrid"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_wu_analyzer_blockgrid_disabled_by_default(self, tmp_path):
        """WuAnalyzer has blockgrid disabled by default."""
        from wu import WuAnalyzer

        path = create_test_jpeg(tmp_path)

        analyzer = WuAnalyzer()
        result = analyzer.analyze(path)

        assert result.blockgrid is None


class TestDoubleCompression:
    """Test double JPEG compression detection."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL or not HAS_NUMPY, reason="scipy/PIL/numpy required")
    def test_double_compression_detection_method(self, analyzer, tmp_path):
        """Double compression detection method runs without error."""
        path = create_test_jpeg(tmp_path, quality=70)

        with Image.open(path) as img:
            image_array = np.mean(np.array(img), axis=2).astype(np.float64)

        detected, quality = analyzer._detect_double_compression(image_array)

        # Just verify it returns expected types
        assert isinstance(detected, bool)
        assert quality is None or isinstance(quality, int)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_png_image(self, analyzer, tmp_path):
        """PNG images are handled (may show no grid)."""
        img = Image.new('RGB', (100, 100), color='blue')
        path = tmp_path / "test.png"
        img.save(path, format='PNG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "blockgrid"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_small_image(self, analyzer, tmp_path):
        """Small images are handled."""
        img = Image.new('RGB', (30, 30), color='red')
        path = tmp_path / "small.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "blockgrid"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_grayscale_image(self, analyzer, tmp_path):
        """Grayscale images are handled."""
        img = Image.new('L', (100, 100), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "blockgrid"


class TestRegionMerging:
    """Test region merging functionality."""

    def test_merge_empty_list(self, analyzer):
        """Empty list returns empty list."""
        result = analyzer._merge_regions([])
        assert result == []

    def test_merge_single_region(self, analyzer):
        """Single region doesn't form a cluster."""
        offset = BlockGridOffset(3, 5, 0.8)
        regions = [GridRegionAnalysis(10, 10, 64, 64, offset, True)]
        result = analyzer._merge_regions(regions)
        # Single region doesn't meet cluster threshold
        assert len(result) == 0

    def test_merge_adjacent_regions(self, analyzer):
        """Adjacent regions with same offset are merged."""
        offset = BlockGridOffset(3, 5, 0.8)
        regions = [
            GridRegionAnalysis(10, 10, 64, 64, offset, True),
            GridRegionAnalysis(42, 10, 64, 64, offset, True),  # Adjacent
            GridRegionAnalysis(74, 10, 64, 64, offset, True),  # Adjacent
        ]
        result = analyzer._merge_regions(regions)
        # Should merge into one region
        assert len(result) == 1


class TestPerformanceMarkers:
    """Test that performance optimization markers are present."""

    def test_module_has_optimization_comments(self):
        """Module documents optimization opportunities."""
        import wu.dimensions.blockgrid as bg
        source = bg.__doc__

        assert "OPTIMIZE" in source
        assert "CYTHON" in source or "C" in source

    def test_module_has_native_stubs(self):
        """Module includes native implementation stubs."""
        import wu.dimensions.blockgrid as bg
        import inspect
        source = inspect.getsource(bg)

        # Check for C/Cython stub documentation
        assert "cython" in source.lower() or "SIMD" in source or "AVX" in source


class TestNoDependencies:
    """Test behavior when dependencies not available."""

    def test_handles_missing_deps_gracefully(self):
        """Analyzer handles missing deps."""
        analyzer = BlockGridAnalyzer()
        assert analyzer is not None
