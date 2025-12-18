"""
Tests for main WuAnalyzer class.

Tests the orchestration of forensic analysis and
chain of custody features.
"""

import pytest
import tempfile
from pathlib import Path
from wu.analyzer import WuAnalyzer
from wu.state import DimensionState, OverallAssessment


class TestWuAnalyzer:
    """Test WuAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_analyzer_creation(self, analyzer):
        """Analyzer can be created."""
        assert analyzer is not None
        assert analyzer.enable_metadata is True

    def test_analyze_nonexistent_file(self, analyzer):
        """Analyzing nonexistent file returns proper result."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        # Path may be normalized differently on Windows
        assert "nonexistent" in result.file_path
        assert result.metadata is not None
        assert result.metadata.state == DimensionState.UNCERTAIN

    def test_file_hash_computed(self, analyzer):
        """File hash is computed for chain of custody."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test image content")
            temp_path = f.name

        try:
            result = analyzer.analyze(temp_path)
            # SHA256 hex is 64 characters
            assert len(result.file_hash) == 64
            assert result.file_hash.isalnum()
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_file_hash_error(self, analyzer):
        """Nonexistent file has FILE_NOT_FOUND hash."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.file_hash == "FILE_NOT_FOUND"

    def test_analyzed_timestamp_set(self, analyzer):
        """Analysis timestamp is set."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.analyzed_at is not None

    def test_wu_version_set(self, analyzer):
        """Wu version is recorded."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.wu_version == "0.1.0"

    def test_metadata_disabled(self):
        """Analyzer can run without metadata analysis."""
        analyzer = WuAnalyzer(enable_metadata=False)
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.metadata is None
        assert result.overall == OverallAssessment.INSUFFICIENT_DATA


class TestSupportedFormats:
    """Test supported format detection."""

    def test_jpg_supported(self):
        """JPG format is supported."""
        assert WuAnalyzer.is_supported("photo.jpg")
        assert WuAnalyzer.is_supported("photo.jpeg")

    def test_png_supported(self):
        """PNG format is supported."""
        assert WuAnalyzer.is_supported("photo.png")

    def test_raw_formats_supported(self):
        """Raw camera formats are supported."""
        assert WuAnalyzer.is_supported("photo.cr2")
        assert WuAnalyzer.is_supported("photo.nef")
        assert WuAnalyzer.is_supported("photo.arw")

    def test_case_insensitive(self):
        """Format detection is case-insensitive."""
        assert WuAnalyzer.is_supported("PHOTO.JPG")
        assert WuAnalyzer.is_supported("Photo.Png")

    def test_unsupported_format(self):
        """Unsupported formats are detected."""
        assert not WuAnalyzer.is_supported("document.pdf")
        assert not WuAnalyzer.is_supported("spreadsheet.xlsx")
        # Video/audio now supported for ENF analysis
        assert WuAnalyzer.is_supported("video.mp4")
        assert WuAnalyzer.is_supported("audio.wav")

    def test_get_formats_list(self):
        """Supported formats list is returned."""
        formats = WuAnalyzer.get_supported_formats()
        assert ".jpg" in formats
        assert ".png" in formats
        assert ".cr2" in formats


class TestBatchAnalysis:
    """Test batch file analysis."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_batch_empty(self, analyzer):
        """Empty batch returns empty list."""
        results = analyzer.analyze_batch([])
        assert results == []

    def test_batch_multiple(self, analyzer):
        """Multiple files can be analyzed."""
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        results = analyzer.analyze_batch(paths)
        assert len(results) == 2


class TestAnalysisResult:
    """Test WuAnalysis result object."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_to_dict(self, analyzer):
        """Result can be converted to dict."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        d = result.to_dict()
        assert "file_path" in d
        assert "file_hash" in d
        assert "overall_assessment" in d
        assert "dimensions" in d

    def test_to_json(self, analyzer):
        """Result can be converted to JSON."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "file_path" in json_str

    def test_dimensions_property(self, analyzer):
        """Dimensions property returns analyzed dimensions."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        dims = result.dimensions
        assert len(dims) == 3  # Metadata + C2PA + Visual
        dim_names = [d.dimension for d in dims]
        assert "metadata" in dim_names
        assert "c2pa" in dim_names
        assert "visual" in dim_names

    def test_has_inconsistencies(self, analyzer):
        """has_inconsistencies property works."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        # Nonexistent file should be uncertain, not inconsistent
        assert result.has_inconsistencies is False


class TestParallelExecution:
    """Test parallel execution features."""

    def test_parallel_default_enabled(self):
        """Parallel execution is enabled by default."""
        analyzer = WuAnalyzer()
        assert analyzer.parallel is True

    def test_parallel_can_be_disabled(self):
        """Parallel execution can be disabled."""
        analyzer = WuAnalyzer(parallel=False)
        assert analyzer.parallel is False

    def test_max_workers_configurable(self):
        """Max workers can be configured."""
        analyzer = WuAnalyzer(max_workers=2)
        assert analyzer.max_workers == 2

    def test_max_workers_auto_default(self):
        """Max workers defaults to auto-detection."""
        analyzer = WuAnalyzer()
        assert analyzer.max_workers >= 1
        assert analyzer.max_workers <= 8

    def test_parallel_analyze_produces_same_results(self):
        """Parallel analysis produces same results as sequential."""
        parallel_analyzer = WuAnalyzer(parallel=True)
        sequential_analyzer = WuAnalyzer(parallel=False)

        # Use nonexistent file to test basic flow
        parallel_result = parallel_analyzer.analyze("/nonexistent/file.jpg")
        sequential_result = sequential_analyzer.analyze("/nonexistent/file.jpg")

        # Results should have same dimensions
        assert len(parallel_result.dimensions) == len(sequential_result.dimensions)

        # Both should have same overall assessment
        assert parallel_result.overall == sequential_result.overall

        # Both should have same dimension states
        for p_dim, s_dim in zip(
            sorted(parallel_result.dimensions, key=lambda d: d.dimension),
            sorted(sequential_result.dimensions, key=lambda d: d.dimension)
        ):
            assert p_dim.dimension == s_dim.dimension
            assert p_dim.state == s_dim.state

    def test_analyzer_config_built_correctly(self):
        """Analyzer config reflects enabled dimensions."""
        # All defaults
        analyzer1 = WuAnalyzer()
        assert len(analyzer1._analyzer_config) == 3  # metadata, c2pa, visual

        # With extra dimensions
        analyzer2 = WuAnalyzer(enable_thumbnail=True, enable_blockgrid=True)
        assert len(analyzer2._analyzer_config) == 5

        # Minimal
        analyzer3 = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False
        )
        assert len(analyzer3._analyzer_config) == 0

    def test_single_dimension_uses_sequential(self):
        """Single dimension doesn't spawn unnecessary threads."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_c2pa=False,
            enable_visual=False,
            parallel=True
        )
        # With only one dimension, should use sequential path
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.metadata is not None
        assert result.c2pa is None


class TestBatchParallelExecution:
    """Test batch analysis with parallel execution."""

    def test_batch_parallel_default_enabled(self):
        """Batch parallel is enabled by default."""
        analyzer = WuAnalyzer()
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        results = analyzer.analyze_batch(paths, parallel_files=True)
        assert len(results) == 2

    def test_batch_parallel_can_be_disabled(self):
        """Batch parallel can be disabled."""
        analyzer = WuAnalyzer()
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        results = analyzer.analyze_batch(paths, parallel_files=False)
        assert len(results) == 2

    def test_batch_preserves_order(self):
        """Batch results are returned in same order as input."""
        analyzer = WuAnalyzer()
        paths = [
            "/nonexistent/a.jpg",
            "/nonexistent/b.jpg",
            "/nonexistent/c.jpg"
        ]
        results = analyzer.analyze_batch(paths, parallel_files=True)

        # Results should be in same order as input
        assert "a.jpg" in results[0].file_path
        assert "b.jpg" in results[1].file_path
        assert "c.jpg" in results[2].file_path

    def test_batch_max_workers_configurable(self):
        """Batch max workers can be configured."""
        analyzer = WuAnalyzer()
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        results = analyzer.analyze_batch(
            paths,
            parallel_files=True,
            max_file_workers=1
        )
        assert len(results) == 2

    def test_batch_single_file_no_parallel(self):
        """Single file in batch doesn't use parallel overhead."""
        analyzer = WuAnalyzer()
        paths = ["/nonexistent/a.jpg"]
        results = analyzer.analyze_batch(paths, parallel_files=True)
        assert len(results) == 1
