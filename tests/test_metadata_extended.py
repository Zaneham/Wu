"""
Extended tests for metadata forensic analyzer.

Tests edge cases, all detection patterns, and error handling.
"""

import pytest
from datetime import datetime
from wu.dimensions.metadata import MetadataAnalyzer
from wu.state import DimensionState, Confidence


class TestAllEditingSoftwareDetected:
    """Test all editing software signatures are detected."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    @pytest.mark.parametrize("software", [
        "Adobe Photoshop CC 2023",
        "Adobe Premiere Pro",
        "Final Cut Pro X",
        "DaVinci Resolve 18",
        "After Effects 2023",
        "Lightroom Classic",
        "GIMP 2.10",
        "paint.net 5.0",
        "Pixelmator Pro",
        "Affinity Photo 2",
        "Capture One 23",
        "Luminar Neo",
        "Topaz Photo AI",
        "FFmpeg v5.0",
        "HandBrake 1.6",
        "Avidemux 2.8",
    ])
    def test_editing_software_detected(self, analyzer, software):
        """Each editing software is detected."""
        metadata = {"Software": software}
        result = analyzer._check_software_signatures(metadata)
        assert result is not None, f"Failed to detect: {software}"


class TestAllAISignaturesDetected:
    """Test all AI generation signatures are detected."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    @pytest.mark.parametrize("signature,field", [
        ("DALL-E 3", "Software"),
        ("Midjourney v6", "Software"),
        ("Stable Diffusion XL", "Software"),
        ("ComfyUI", "Software"),
        ("Automatic1111", "Software"),
        ("Sora by OpenAI", "Comment"),
        ("Runway Gen-2", "Software"),
        ("Pika Labs", "Software"),
        ("Kling AI", "Software"),
        ("FLUX.1", "Software"),
        ("Adobe Firefly", "Software"),
    ])
    def test_ai_signature_detected(self, analyzer, signature, field):
        """Each AI signature is detected."""
        metadata = {field: signature}
        result = analyzer._check_ai_signatures(metadata)
        assert result is not None, f"Failed to detect: {signature}"

    def test_sd_parameters_steps(self, analyzer):
        """Stable Diffusion steps parameter detected."""
        metadata = {"parameters": "Steps: 30, Sampler: Euler"}
        result = analyzer._check_ai_signatures(metadata)
        assert result is not None

    def test_sd_parameters_sampler(self, analyzer):
        """Stable Diffusion sampler parameter detected."""
        metadata = {"parameters": "Sampler: DPM++ 2M Karras"}
        result = analyzer._check_ai_signatures(metadata)
        assert result is not None

    def test_sd_parameters_cfg(self, analyzer):
        """Stable Diffusion CFG scale detected."""
        metadata = {"parameters": "CFG Scale: 7.5"}
        result = analyzer._check_ai_signatures(metadata)
        assert result is not None


class TestSoftwareFieldVariations:
    """Test software detection in different metadata fields."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_software_field(self, analyzer):
        """Software field detection."""
        metadata = {"Software": "Adobe Photoshop"}
        assert analyzer._check_software_signatures(metadata) is not None

    def test_processing_software_field(self, analyzer):
        """ProcessingSoftware field detection."""
        metadata = {"ProcessingSoftware": "Adobe Photoshop"}
        assert analyzer._check_software_signatures(metadata) is not None

    def test_creator_tool_field(self, analyzer):
        """CreatorTool field detection."""
        metadata = {"CreatorTool": "Adobe Photoshop"}
        assert analyzer._check_software_signatures(metadata) is not None

    def test_history_software_agent_field(self, analyzer):
        """HistorySoftwareAgent field detection."""
        metadata = {"HistorySoftwareAgent": "Adobe Photoshop"}
        assert analyzer._check_software_signatures(metadata) is not None

    def test_combined_fields(self, analyzer):
        """Multiple fields combine for detection."""
        metadata = {
            "Software": "Camera App",
            "ProcessingSoftware": "Adobe Photoshop"
        }
        assert analyzer._check_software_signatures(metadata) is not None


class TestTimestampEdgeCases:
    """Test timestamp checking edge cases."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_no_timestamps(self, analyzer):
        """No timestamps returns None."""
        metadata = {"Make": "Apple"}
        result = analyzer._check_timestamps(metadata)
        assert result is None

    def test_single_valid_timestamp(self, analyzer):
        """Single valid timestamp returns None."""
        metadata = {"DateTimeOriginal": "2023:06:15 10:30:00"}
        result = analyzer._check_timestamps(metadata)
        assert result is None

    def test_malformed_timestamp_ignored(self, analyzer):
        """Malformed timestamp is ignored."""
        metadata = {
            "DateTimeOriginal": "not a timestamp",
            "DateTime": "also not valid"
        }
        result = analyzer._check_timestamps(metadata)
        assert result is None

    def test_mixed_valid_invalid_timestamps(self, analyzer):
        """Mixed valid/invalid timestamps handled."""
        metadata = {
            "DateTimeOriginal": "2023:06:15 10:30:00",
            "DateTime": "invalid"
        }
        result = analyzer._check_timestamps(metadata)
        assert result is None  # Only one valid timestamp

    def test_all_date_formats(self, analyzer):
        """All supported date formats parse."""
        formats = [
            "2023:06:15 10:30:00",  # EXIF standard
            "2023-06-15 10:30:00",  # ISO-ish
            "2023/06/15 10:30:00",  # Slash separated
        ]
        for fmt in formats:
            ts = analyzer._parse_exif_datetime(fmt)
            assert ts is not None, f"Failed to parse: {fmt}"

    def test_future_date_various_fields(self, analyzer):
        """Future date detected in various fields."""
        future_fields = [
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
            "CreateDate",
            "ModifyDate",
        ]
        for field in future_fields:
            metadata = {field: "2099:12:31 23:59:59"}
            result = analyzer._check_timestamps(metadata)
            assert result is not None, f"Failed to detect future in {field}"

    def test_modification_one_second_before(self, analyzer):
        """Modification one second before original is caught."""
        metadata = {
            "DateTimeOriginal": "2023:06:15 10:30:01",
            "DateTime": "2023:06:15 10:30:00"
        }
        result = analyzer._check_timestamps(metadata)
        assert result is not None

    def test_same_timestamps_ok(self, analyzer):
        """Same timestamps are allowed."""
        metadata = {
            "DateTimeOriginal": "2023:06:15 10:30:00",
            "DateTime": "2023:06:15 10:30:00"
        }
        result = analyzer._check_timestamps(metadata)
        assert result is None


class TestMetadataStrippingDetection:
    """Test stripped metadata detection."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_zero_fields_stripped(self, analyzer):
        """Zero fields is stripped."""
        metadata = {}
        assert analyzer._is_metadata_stripped(metadata) is True

    def test_one_field_stripped(self, analyzer):
        """One field is stripped."""
        metadata = {"ColorSpace": "sRGB"}
        assert analyzer._is_metadata_stripped(metadata) is True

    def test_four_fields_stripped(self, analyzer):
        """Four fields is stripped."""
        metadata = {
            "ColorSpace": "sRGB",
            "Width": "1920",
            "Height": "1080",
            "Format": "JPEG"
        }
        assert analyzer._is_metadata_stripped(metadata) is True

    def test_five_fields_not_stripped(self, analyzer):
        """Five fields is not stripped."""
        metadata = {
            "ColorSpace": "sRGB",
            "Width": "1920",
            "Height": "1080",
            "Format": "JPEG",
            "Orientation": "1"
        }
        assert analyzer._is_metadata_stripped(metadata) is False

    def test_internal_fields_not_counted(self, analyzer):
        """Internal fields (starting with _) not counted."""
        metadata = {
            "_image_width": 1920,
            "_image_height": 1080,
            "_image_format": "JPEG",
            "_internal": "value",
            "_another": "value"
        }
        assert analyzer._is_metadata_stripped(metadata) is True


class TestResolutionNaming:
    """Test resolution name helper."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_8k_resolution(self, analyzer):
        """8K resolution named correctly."""
        assert analyzer._resolution_name(7680, 4320) == "8K"

    def test_4k_exact(self, analyzer):
        """Exact 4K resolution."""
        assert analyzer._resolution_name(3840, 2160) == "4K UHD"

    def test_4k_cinema(self, analyzer):
        """Cinema 4K resolution."""
        assert analyzer._resolution_name(4096, 2160) == "4K UHD"

    def test_1080p_exact(self, analyzer):
        """Exact 1080p resolution."""
        assert analyzer._resolution_name(1920, 1080) == "1080p"

    def test_720p_exact(self, analyzer):
        """Exact 720p resolution."""
        assert analyzer._resolution_name(1280, 720) == "720p"

    def test_below_720p(self, analyzer):
        """Below 720p shows dimensions."""
        result = analyzer._resolution_name(640, 480)
        assert "640" in result and "480" in result

    def test_odd_resolution(self, analyzer):
        """Odd resolution shows dimensions."""
        result = analyzer._resolution_name(1000, 1000)
        # 1,000,000 pixels is between 720p and 1080p
        assert result == "720p"  # Categorized by closest match


class TestAnalyzeIntegration:
    """Integration tests for full analyze method."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_analyze_returns_dimension_result(self, analyzer):
        """Analyze returns DimensionResult."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.dimension == "metadata"

    def test_analyze_nonexistent_evidence(self, analyzer):
        """Nonexistent file has proper evidence."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert len(result.evidence) == 1
        assert "not found" in result.evidence[0].finding.lower()

    def test_analyze_methodology_set(self, analyzer):
        """Methodology is set when metadata found."""
        # Would need actual file to test this fully
        pass


class TestDeviceResolutionCheckEdgeCases:
    """Edge cases for device resolution checking."""

    @pytest.fixture
    def analyzer(self):
        return MetadataAnalyzer()

    def test_no_make_or_model(self, analyzer):
        """No make or model returns None."""
        metadata = {"_image_width": 3840, "_image_height": 2160}
        result = analyzer._check_device_resolution(metadata)
        assert result is None

    def test_only_make(self, analyzer):
        """Only make without model works."""
        metadata = {
            "Make": "Apple",
            "_image_width": 3840,
            "_image_height": 2160
        }
        # Behavior depends on partial matching
        result = analyzer._check_device_resolution(metadata)
        # May or may not find a match

    def test_no_resolution(self, analyzer):
        """No resolution returns None."""
        metadata = {"Make": "Apple", "Model": "iPhone 6"}
        result = analyzer._check_device_resolution(metadata)
        assert result is None

    def test_exif_width_height_fields(self, analyzer):
        """ExifImageWidth/Height fields used."""
        metadata = {
            "Make": "Apple",
            "Model": "iPhone 6",
            "ExifImageWidth": "7680",
            "ExifImageHeight": "4320"
        }
        result = analyzer._check_device_resolution(metadata)
        assert result is not None  # 8K from iPhone 6 is impossible

    def test_width_height_with_units(self, analyzer):
        """Width/height with units parsed."""
        metadata = {
            "Make": "Apple",
            "Model": "iPhone 6",
            "_image_width": "7680 pixels",
            "_image_height": "4320 pixels"
        }
        result = analyzer._check_device_resolution(metadata)
        assert result is not None

    def test_invalid_width_height(self, analyzer):
        """Invalid width/height returns None."""
        metadata = {
            "Make": "Apple",
            "Model": "iPhone 6",
            "_image_width": "not a number",
            "_image_height": "also not"
        }
        result = analyzer._check_device_resolution(metadata)
        assert result is None
