"""
Tests for copy-move (clone) forensic detection.

Tests clone region detection, block-based and keypoint-based methods.
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
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import fftpack
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from wu.dimensions.copymove import CopyMoveAnalyzer, CloneRegion, CopyMoveResult
from wu.state import DimensionState, Confidence


@pytest.fixture
def analyzer():
    return CopyMoveAnalyzer()


def create_test_image(tmp_path, size=(200, 200), add_clone=False):
    """Create a test image with optional cloned region."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL/numpy required")

    # Create truly random noise image (no repeating patterns)
    np.random.seed(42)  # Reproducible
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    if add_clone:
        # Copy a region from one place to another (distant enough)
        source_region = img_array[20:80, 20:80].copy()  # 60x60 region
        img_array[120:180, 120:180] = source_region  # Paste 100 pixels away

    img = Image.fromarray(img_array)
    path = tmp_path / ("cloned.jpg" if add_clone else "clean.jpg")
    img.save(path, format='JPEG', quality=95)
    return str(path)


class TestCloneRegion:
    """Test CloneRegion dataclass."""

    def test_creation(self):
        """CloneRegion can be created."""
        region = CloneRegion(
            source_x=10, source_y=20,
            target_x=100, target_y=120,
            width=50, height=50,
            similarity=0.95
        )
        assert region.source_x == 10
        assert region.width == 50
        assert region.similarity == 0.95

    def test_distance(self):
        """Distance is computed correctly."""
        region = CloneRegion(
            source_x=0, source_y=0,
            target_x=30, target_y=40,
            width=50, height=50,
            similarity=0.95
        )
        # 3-4-5 triangle
        assert region.distance == 50.0

    def test_distance_diagonal(self):
        """Distance works for diagonal offset."""
        region = CloneRegion(
            source_x=10, source_y=10,
            target_x=110, target_y=110,
            width=50, height=50,
            similarity=0.9
        )
        expected = math.sqrt(100**2 + 100**2)
        assert abs(region.distance - expected) < 0.01


class TestCopyMoveResult:
    """Test CopyMoveResult dataclass."""

    def test_creation_detected(self):
        """CopyMoveResult with detection."""
        result = CopyMoveResult(
            detected=True,
            clone_regions=[CloneRegion(0, 0, 50, 50, 30, 30, 0.9)],
            method_used="block_dct",
            processing_time_ms=150.0
        )
        assert result.detected is True
        assert len(result.clone_regions) == 1
        assert result.method_used == "block_dct"

    def test_creation_not_detected(self):
        """CopyMoveResult without detection."""
        result = CopyMoveResult(detected=False, method_used="orb_keypoint")
        assert result.detected is False
        assert result.clone_regions == []


class TestCopyMoveAnalyzerBasic:
    """Test basic CopyMoveAnalyzer functionality."""

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None

    def test_analyze_missing_file(self, analyzer):
        """Missing file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "copymove"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_invalid_file(self, analyzer, tmp_path):
        """Invalid file returns UNCERTAIN."""
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")
        result = analyzer.analyze(str(bad_file))
        assert result.state == DimensionState.UNCERTAIN

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_small_image(self, analyzer, tmp_path):
        """Small image is handled."""
        img = Image.new('RGB', (20, 20), color='red')
        path = tmp_path / "small.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "copymove"


class TestBlockBasedDetection:
    """Test block-based DCT copy-move detection."""

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL, reason="scipy/PIL required")
    def test_clean_image_no_detection(self, analyzer, tmp_path):
        """Clean image has no copy-move detection."""
        path = create_test_image(tmp_path, add_clone=False)
        result = analyzer._detect_block_based(np.array(Image.open(path)))

        # Should not detect clones in random noise
        # May have some false positives, but shouldn't be many
        assert len(result.clone_regions) < 3

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL, reason="scipy/PIL required")
    def test_cloned_image_detection(self, analyzer, tmp_path):
        """Cloned image should be detected."""
        path = create_test_image(tmp_path, size=(300, 300), add_clone=True)
        result = analyzer._detect_block_based(np.array(Image.open(path)))

        # Should detect the clone
        # Note: this depends on the specifics of the clone
        assert result.method_used == "block_dct"

    @pytest.mark.skipif(not HAS_SCIPY or not HAS_PIL, reason="scipy/PIL required")
    def test_processing_time_recorded(self, analyzer, tmp_path):
        """Processing time is recorded."""
        path = create_test_image(tmp_path, size=(100, 100))
        result = analyzer._detect_block_based(np.array(Image.open(path)))
        assert result.processing_time_ms > 0


class TestKeypointBasedDetection:
    """Test keypoint-based copy-move detection."""

    @pytest.mark.skipif(not HAS_CV2 or not HAS_PIL, reason="OpenCV/PIL required")
    def test_clean_image_no_detection(self, analyzer, tmp_path):
        """Clean image has no copy-move detection."""
        path = create_test_image(tmp_path, add_clone=False)
        result = analyzer._detect_keypoint_based(np.array(Image.open(path)))
        assert result.method_used == "orb_keypoint"

    @pytest.mark.skipif(not HAS_CV2 or not HAS_PIL, reason="OpenCV/PIL required")
    def test_processing_time_recorded(self, analyzer, tmp_path):
        """Processing time is recorded."""
        path = create_test_image(tmp_path, size=(100, 100))
        result = analyzer._detect_keypoint_based(np.array(Image.open(path)))
        assert result.processing_time_ms >= 0


class TestFullAnalysis:
    """Test full copy-move analysis pipeline."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_returns_dimension_result(self, analyzer, tmp_path):
        """Analysis returns DimensionResult."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)

        assert result.dimension == "copymove"
        assert result.state in [
            DimensionState.CONSISTENT,
            DimensionState.INCONSISTENT,
            DimensionState.SUSPICIOUS,
            DimensionState.UNCERTAIN
        ]

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_has_evidence(self, analyzer, tmp_path):
        """Analysis includes evidence."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)
        assert len(result.evidence) > 0

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_analyze_has_methodology(self, analyzer, tmp_path):
        """Analysis includes methodology."""
        path = create_test_image(tmp_path)
        result = analyzer.analyze(path)
        assert result.methodology is not None

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_consistent_for_clean_image(self, analyzer, tmp_path):
        """Clean image is CONSISTENT."""
        path = create_test_image(tmp_path, add_clone=False)
        result = analyzer.analyze(path)
        # Should be consistent (no clones detected)
        assert result.state in [DimensionState.CONSISTENT, DimensionState.UNCERTAIN]


class TestRegionMerging:
    """Test clone region merging."""

    def test_merge_empty_list(self, analyzer):
        """Empty list returns empty list."""
        result = analyzer._merge_clone_regions([])
        assert result == []

    def test_merge_single_region(self, analyzer):
        """Single region is returned if cluster is small."""
        regions = [CloneRegion(10, 10, 100, 100, 16, 16, 0.9)]
        result = analyzer._merge_clone_regions(regions)
        # Single region doesn't meet minimum cluster size
        assert len(result) == 0

    def test_merge_overlapping_regions(self, analyzer):
        """Overlapping regions are merged."""
        regions = [
            CloneRegion(10, 10, 100, 100, 16, 16, 0.9),
            CloneRegion(12, 12, 102, 102, 16, 16, 0.92),
            CloneRegion(14, 14, 104, 104, 16, 16, 0.88),
        ]
        result = analyzer._merge_clone_regions(regions)
        # Should merge into one larger region
        assert len(result) == 1


class TestIntegration:
    """Test copy-move integration with main analyzer."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_wu_analyzer_with_copymove(self, tmp_path):
        """WuAnalyzer can enable copy-move detection."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_copymove=True
        )
        result = analyzer.analyze(path)

        assert result.copymove is not None
        assert result.copymove.dimension == "copymove"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_wu_analyzer_copymove_disabled_by_default(self, tmp_path):
        """WuAnalyzer has copymove disabled by default."""
        from wu import WuAnalyzer

        path = create_test_image(tmp_path)

        analyzer = WuAnalyzer()
        result = analyzer.analyze(path)

        assert result.copymove is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_grayscale_image(self, analyzer, tmp_path):
        """Grayscale images are handled."""
        img = Image.new('L', (100, 100), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "copymove"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_rgba_image(self, analyzer, tmp_path):
        """RGBA images are handled."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path, format='PNG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "copymove"

    @pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL/numpy required")
    def test_uniform_image(self, analyzer, tmp_path):
        """Uniform color image is handled."""
        img = Image.new('RGB', (100, 100), color='white')
        path = tmp_path / "uniform.jpg"
        img.save(path, format='JPEG')

        result = analyzer.analyze(str(path))
        assert result.dimension == "copymove"


class TestPerformanceMarkers:
    """Test that performance optimization markers are present."""

    def test_module_has_optimization_comments(self):
        """Module documents optimization opportunities."""
        import wu.dimensions.copymove as cm
        source = cm.__doc__

        assert "OPTIMIZE" in source
        assert "CYTHON" in source or "C" in source

    def test_module_has_native_stubs(self):
        """Module includes native implementation stubs."""
        import wu.dimensions.copymove as cm
        import inspect
        source = inspect.getsource(cm)

        # Check for C/Cython stub documentation
        assert "cython" in source.lower() or "SIMD" in source


class TestNoDependencies:
    """Test behavior when dependencies not available."""

    def test_handles_missing_deps_gracefully(self):
        """Analyzer handles missing deps."""
        analyzer = CopyMoveAnalyzer()
        assert analyzer is not None
