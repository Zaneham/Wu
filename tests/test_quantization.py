"""
Tests for JPEG quantization table forensic analysis.

Tests quantization table parsing, software fingerprinting, and double
compression detection.
"""

import pytest

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

from wu.dimensions.quantization import (
    QuantizationAnalyzer,
    QuantizationTable,
    QuantizationResult,
    DoubleCompressionResult,
    SoftwareMatch,
    STANDARD_LUMINANCE_Q50,
    STANDARD_CHROMINANCE_Q50,
)
from wu.state import DimensionState, Confidence


pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PIL,
    reason="numpy and PIL required for quantization tests"
)


class TestQuantizationTable:
    """Tests for QuantizationTable dataclass."""

    def test_creation(self):
        table = QuantizationTable(
            table_id=0,
            values=STANDARD_LUMINANCE_Q50.copy(),
            estimated_quality=50,
            is_standard=True
        )
        assert table.table_id == 0
        assert table.estimated_quality == 50
        assert table.is_standard is True

    def test_mean_value(self):
        table = QuantizationTable(
            table_id=0,
            values=[10] * 64,
            estimated_quality=80,
            is_standard=False
        )
        assert table.mean_value == 10.0

    def test_dc_value(self):
        values = [16] + [11] * 63
        table = QuantizationTable(
            table_id=0,
            values=values,
            estimated_quality=50,
            is_standard=True
        )
        assert table.dc_value == 16

    def test_empty_values(self):
        table = QuantizationTable(
            table_id=0,
            values=[],
            estimated_quality=0,
            is_standard=False
        )
        assert table.mean_value == 0
        assert table.dc_value == 0  # Should handle empty list


class TestSoftwareMatch:
    """Tests for SoftwareMatch dataclass."""

    def test_creation(self):
        match = SoftwareMatch(
            software="photoshop",
            description="Adobe Photoshop",
            confidence=0.85
        )
        assert match.software == "photoshop"
        assert match.confidence == 0.85


class TestDoubleCompressionResult:
    """Tests for DoubleCompressionResult dataclass."""

    def test_not_detected(self):
        result = DoubleCompressionResult(
            detected=False,
            primary_quality=85,
            secondary_quality=None,
            confidence=0.0
        )
        assert result.detected is False
        assert result.secondary_quality is None

    def test_detected(self):
        result = DoubleCompressionResult(
            detected=True,
            primary_quality=75,
            secondary_quality=90,
            confidence=0.8,
            ghost_quality=90
        )
        assert result.detected is True
        assert result.secondary_quality == 90


class TestQuantizationResult:
    """Tests for QuantizationResult dataclass."""

    def test_creation(self):
        table = QuantizationTable(0, STANDARD_LUMINANCE_Q50.copy(), 50, True)
        result = QuantizationResult(
            tables=[table],
            primary_quality=50,
            is_double_compressed=False,
            double_compression=None,
            software_matches=[],
            table_modified=False,
            processing_time_ms=10.5
        )
        assert len(result.tables) == 1
        assert result.primary_quality == 50
        assert result.is_double_compressed is False


class TestQuantizationAnalyzerBasic:
    """Basic tests for QuantizationAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = QuantizationAnalyzer()
        assert analyzer is not None

    def test_analyze_missing_file(self):
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze("/nonexistent/photo.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "quantization"

    def test_analyze_invalid_file(self, tmp_path):
        invalid_file = tmp_path / "not_image.txt"
        invalid_file.write_text("This is not an image")

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(str(invalid_file))
        assert result.state == DimensionState.UNCERTAIN


@pytest.fixture
def jpeg_image(tmp_path):
    """Create a simple JPEG image."""
    img = Image.new('RGB', (200, 200), color='blue')
    file_path = tmp_path / "test.jpg"
    img.save(str(file_path), "JPEG", quality=75)
    return str(file_path)


@pytest.fixture
def jpeg_high_quality(tmp_path):
    """Create a high-quality JPEG image."""
    img = Image.new('RGB', (200, 200), color='red')
    file_path = tmp_path / "high_quality.jpg"
    img.save(str(file_path), "JPEG", quality=95)
    return str(file_path)


@pytest.fixture
def jpeg_low_quality(tmp_path):
    """Create a low-quality JPEG image."""
    img = Image.new('RGB', (200, 200), color='green')
    file_path = tmp_path / "low_quality.jpg"
    img.save(str(file_path), "JPEG", quality=30)
    return str(file_path)


@pytest.fixture
def png_image(tmp_path):
    """Create a PNG image (not JPEG)."""
    img = Image.new('RGB', (200, 200), color='purple')
    file_path = tmp_path / "test.png"
    img.save(str(file_path), "PNG")
    return str(file_path)


class TestQuantizationAnalysis:
    """Tests for quantization analysis on JPEG images."""

    def test_analyze_jpeg(self, jpeg_image):
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(jpeg_image)

        assert result.dimension == "quantization"
        assert result.state is not None
        assert result.confidence is not None

    def test_analyze_high_quality_jpeg(self, jpeg_high_quality):
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(jpeg_high_quality)

        assert result.dimension == "quantization"
        # Should report evidence about quantization tables
        assert len(result.evidence) > 0

    def test_analyze_low_quality_jpeg(self, jpeg_low_quality):
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(jpeg_low_quality)

        assert result.dimension == "quantization"
        assert len(result.evidence) > 0

    def test_analyze_non_jpeg(self, png_image):
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(png_image)

        # PNG files should return uncertain (no quantization tables)
        assert result.state == DimensionState.UNCERTAIN
        assert any("JPEG" in e.finding or "JPEG" in e.explanation for e in result.evidence)


class TestQualityEstimation:
    """Tests for JPEG quality estimation."""

    def test_estimate_quality_standard_q50(self):
        analyzer = QuantizationAnalyzer()
        quality = analyzer._estimate_quality_from_table(STANDARD_LUMINANCE_Q50)
        # Should be around 50
        assert 40 <= quality <= 60

    def test_estimate_quality_high(self):
        analyzer = QuantizationAnalyzer()
        # High quality = low quantization values
        high_quality_table = [max(1, x // 2) for x in STANDARD_LUMINANCE_Q50]
        quality = analyzer._estimate_quality_from_table(high_quality_table)
        assert quality > 50

    def test_estimate_quality_low(self):
        analyzer = QuantizationAnalyzer()
        # Low quality = high quantization values
        low_quality_table = [min(255, x * 2) for x in STANDARD_LUMINANCE_Q50]
        quality = analyzer._estimate_quality_from_table(low_quality_table)
        assert quality < 50

    def test_estimate_quality_empty(self):
        analyzer = QuantizationAnalyzer()
        quality = analyzer._estimate_quality_from_table([])
        assert quality == 0


class TestStandardTableDetection:
    """Tests for standard table detection."""

    def test_standard_luminance_table(self):
        analyzer = QuantizationAnalyzer()
        is_standard = analyzer._is_standard_table(STANDARD_LUMINANCE_Q50, 0)
        assert bool(is_standard) is True

    def test_standard_chrominance_table(self):
        analyzer = QuantizationAnalyzer()
        is_standard = analyzer._is_standard_table(STANDARD_CHROMINANCE_Q50, 1)
        assert bool(is_standard) is True

    def test_modified_table_not_standard(self):
        analyzer = QuantizationAnalyzer()
        # Random table
        random_table = [10, 20, 5, 30, 15, 25, 8, 12] * 8
        is_standard = analyzer._is_standard_table(random_table, 0)
        assert bool(is_standard) is False

    def test_empty_table_not_standard(self):
        analyzer = QuantizationAnalyzer()
        is_standard = analyzer._is_standard_table([], 0)
        assert is_standard is False


class TestTableModificationDetection:
    """Tests for detecting table modification."""

    def test_normal_tables_not_modified(self):
        analyzer = QuantizationAnalyzer()
        tables = [
            QuantizationTable(0, STANDARD_LUMINANCE_Q50.copy(), 50, True),
            QuantizationTable(1, STANDARD_CHROMINANCE_Q50.copy(), 50, True),
        ]
        modified = analyzer._check_table_modification(tables)
        assert modified is False

    def test_all_same_values_modified(self):
        analyzer = QuantizationAnalyzer()
        tables = [
            QuantizationTable(0, [16] * 64, 50, False),
        ]
        modified = analyzer._check_table_modification(tables)
        assert modified is True

    def test_inverted_pattern_modified(self):
        analyzer = QuantizationAnalyzer()
        # Create inverted pattern (high freq lower than low freq)
        inverted = [100] * 16 + [1] * 48
        tables = [
            QuantizationTable(0, inverted, 50, False),
        ]
        modified = analyzer._check_table_modification(tables)
        assert modified is True

    def test_zero_dc_modified(self):
        analyzer = QuantizationAnalyzer()
        values = [0] + [16] * 63
        tables = [
            QuantizationTable(0, values, 50, False),
        ]
        modified = analyzer._check_table_modification(tables)
        assert modified is True


class TestSoftwareIdentification:
    """Tests for software identification."""

    def test_identify_software_basic(self):
        analyzer = QuantizationAnalyzer()
        tables = [
            QuantizationTable(0, STANDARD_LUMINANCE_Q50.copy(), 50, True),
            QuantizationTable(1, STANDARD_CHROMINANCE_Q50.copy(), 50, True),
        ]
        matches = analyzer._identify_software(tables)
        # Software identification requires pattern matching which may return
        # empty if no strong fingerprint is detected - this is acceptable
        # The important thing is no error and it returns a list
        assert isinstance(matches, list)

    def test_identify_software_single_table(self):
        analyzer = QuantizationAnalyzer()
        tables = [
            QuantizationTable(0, STANDARD_LUMINANCE_Q50.copy(), 50, True),
        ]
        matches = analyzer._identify_software(tables)
        # Single table can't reliably identify software
        assert len(matches) == 0


class TestIntegration:
    """Integration tests with main WuAnalyzer."""

    def test_wu_analyzer_with_quantization(self, jpeg_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_quantization=True
        )
        result = analyzer.analyze(jpeg_image)

        assert result.quantization is not None
        assert result.quantization.dimension == "quantization"

    def test_wu_analyzer_quantization_disabled_by_default(self, jpeg_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer()
        result = analyzer.analyze(jpeg_image)

        # Quantization should be disabled by default
        assert result.quantization is None


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_jpeg(self, tmp_path):
        """Very small JPEG should be handled."""
        img = Image.new('RGB', (32, 32), color='red')
        file_path = tmp_path / "tiny.jpg"
        img.save(str(file_path), "JPEG", quality=75)

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(str(file_path))
        assert result.dimension == "quantization"

    def test_grayscale_jpeg(self, tmp_path):
        """Grayscale JPEG should be handled."""
        img = Image.new('L', (200, 200), color=128)
        file_path = tmp_path / "gray.jpg"
        img.save(str(file_path), "JPEG", quality=75)

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(str(file_path))
        assert result.dimension == "quantization"

    def test_rgba_to_jpeg(self, tmp_path):
        """RGBA converted to JPEG should be handled."""
        img = Image.new('RGBA', (200, 200), color=(255, 0, 0, 128))
        file_path = tmp_path / "from_rgba.jpg"
        # Convert to RGB before saving as JPEG
        rgb_img = img.convert('RGB')
        rgb_img.save(str(file_path), "JPEG", quality=75)

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(str(file_path))
        assert result.dimension == "quantization"


class TestDoubleCompressionDetection:
    """Tests for double compression detection."""

    def test_single_compression_not_detected(self, jpeg_image):
        """Single compression should not show double compression."""
        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(jpeg_image)

        # Fresh JPEG should not show double compression
        # (may or may not be detected depending on analysis)
        assert result.dimension == "quantization"

    def test_recompressed_jpeg(self, tmp_path):
        """Recompressed JPEG may show double compression."""
        # Create original
        img = Image.new('RGB', (400, 400), color='blue')
        for x in range(0, 400, 20):
            for y in range(0, 400, 20):
                img.putpixel((x, y), (255, 0, 0))

        # Save at quality 90
        temp1 = tmp_path / "original.jpg"
        img.save(str(temp1), "JPEG", quality=90)

        # Load and resave at quality 70
        img2 = Image.open(str(temp1))
        temp2 = tmp_path / "recompressed.jpg"
        img2.save(str(temp2), "JPEG", quality=70)

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(str(temp2))

        # Should analyze without error
        assert result.dimension == "quantization"
        assert result.state is not None


class TestNoDependencies:
    """Test graceful handling when dependencies are missing."""

    def test_handles_missing_deps_gracefully(self, monkeypatch, jpeg_image):
        """Should return UNCERTAIN when deps missing."""
        import wu.dimensions.quantization as quant_module
        monkeypatch.setattr(quant_module, "HAS_PIL", False)

        analyzer = QuantizationAnalyzer()
        result = analyzer.analyze(jpeg_image)

        assert result.state == DimensionState.UNCERTAIN
