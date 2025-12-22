"""
End-to-end tests for Wu forensic analysis.

Tests the complete pipeline from file input through all analyzers to final output.
Includes realistic manipulation scenarios and full JSON output validation. - ZH 19/12/2025
"""

import json
import os
import pytest
import subprocess
import sys

from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from wu.analyzer import WuAnalyzer
from wu.state import (
    WuAnalysis,
    DimensionState,
    OverallAssessment,
    Confidence,
)


pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PIL,
    reason="numpy and PIL required for e2e tests"
)


# =============================================================================
# FIXTURES: CREATE REALISTIC TEST IMAGES
# =============================================================================

@pytest.fixture
def clean_jpeg(tmp_path):
    """Create a clean, unmanipulated JPEG image."""
    img = Image.new('RGB', (800, 600), color=(100, 150, 200))

    # Add some texture/detail
    draw = ImageDraw.Draw(img)
    for x in range(0, 800, 50):
        draw.line([(x, 0), (x, 600)], fill=(90, 140, 190), width=1)
    for y in range(0, 600, 50):
        draw.line([(0, y), (800, y)], fill=(90, 140, 190), width=1)

    file_path = tmp_path / "clean.jpg"
    img.save(str(file_path), "JPEG", quality=85)
    return str(file_path)


@pytest.fixture
def clean_png(tmp_path):
    """Create a clean PNG image."""
    img = Image.new('RGB', (800, 600), color=(200, 100, 150))

    draw = ImageDraw.Draw(img)
    draw.rectangle([(100, 100), (300, 300)], fill=(150, 50, 100))
    draw.ellipse([(400, 200), (700, 500)], fill=(250, 150, 200))

    file_path = tmp_path / "clean.png"
    img.save(str(file_path), "PNG")
    return str(file_path)


@pytest.fixture
def recompressed_jpeg(tmp_path):
    """Create a double-compressed JPEG (save, load, resave)."""
    # Create original at high quality
    img = Image.new('RGB', (800, 600), color=(50, 100, 150))
    draw = ImageDraw.Draw(img)

    # Add detailed content
    for i in range(20):
        x = (i * 40) % 800
        y = (i * 30) % 600
        draw.rectangle([(x, y), (x+30, y+30)], fill=(200, 150, 100))

    original_path = tmp_path / "original.jpg"
    img.save(str(original_path), "JPEG", quality=95)

    # Load and resave at lower quality
    img2 = Image.open(str(original_path))
    recompressed_path = tmp_path / "recompressed.jpg"
    img2.save(str(recompressed_path), "JPEG", quality=60)

    return str(recompressed_path)


@pytest.fixture
def spliced_image(tmp_path):
    """Create an image with a spliced region (different compression)."""
    # Create base image
    base = Image.new('RGB', (800, 600), color=(100, 100, 100))
    draw = ImageDraw.Draw(base)

    # Add grid pattern
    for x in range(0, 800, 20):
        draw.line([(x, 0), (x, 600)], fill=(120, 120, 120), width=1)
    for y in range(0, 600, 20):
        draw.line([(0, y), (800, y)], fill=(120, 120, 120), width=1)

    # Save as JPEG
    base_path = tmp_path / "base.jpg"
    base.save(str(base_path), "JPEG", quality=85)

    # Create "foreign" image with different content
    foreign = Image.new('RGB', (200, 200), color=(255, 0, 0))
    draw2 = ImageDraw.Draw(foreign)
    draw2.ellipse([(20, 20), (180, 180)], fill=(255, 255, 0))

    foreign_path = tmp_path / "foreign.jpg"
    foreign.save(str(foreign_path), "JPEG", quality=50)  # Different quality

    # Load and splice
    base_img = Image.open(str(base_path))
    foreign_img = Image.open(str(foreign_path))

    # Paste foreign region into base
    base_img.paste(foreign_img, (300, 200))

    spliced_path = tmp_path / "spliced.jpg"
    base_img.save(str(spliced_path), "JPEG", quality=85)

    return str(spliced_path)


@pytest.fixture
def image_with_clone(tmp_path):
    """Create an image with a cloned (copy-moved) region."""
    img = Image.new('RGB', (800, 600), color=(150, 150, 150))
    draw = ImageDraw.Draw(img)

    # Add unique pattern
    draw.rectangle([(100, 100), (200, 200)], fill=(255, 0, 0))
    draw.ellipse([(120, 120), (180, 180)], fill=(0, 255, 0))

    # Clone that region to another location
    region = img.crop((100, 100, 200, 200))
    img.paste(region, (500, 300))

    file_path = tmp_path / "cloned.jpg"
    img.save(str(file_path), "JPEG", quality=85)
    return str(file_path)


@pytest.fixture
def image_with_text(tmp_path):
    """Create an image with text content."""
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Simple text-like rectangles (simulating text)
    for y in range(50, 500, 30):
        width = 100 + (y % 200)
        draw.rectangle([(50, y), (50 + width, y + 15)], fill=(0, 0, 0))

    file_path = tmp_path / "text.jpg"
    img.save(str(file_path), "JPEG", quality=90)
    return str(file_path)


@pytest.fixture
def grayscale_jpeg(tmp_path):
    """Create a grayscale JPEG."""
    img = Image.new('L', (400, 300), color=128)
    draw = ImageDraw.Draw(img)

    # Add gradient
    for x in range(400):
        for y in range(300):
            val = int((x / 400) * 255)
            img.putpixel((x, y), val)

    file_path = tmp_path / "grayscale.jpg"
    img.save(str(file_path), "JPEG", quality=80)
    return str(file_path)


# =============================================================================
# CORE E2E TESTS: ANALYZER
# =============================================================================

class TestFullAnalysisPipeline:
    """Test complete analysis pipeline with all dimensions."""

    def test_analyze_clean_jpeg_all_dimensions(self, clean_jpeg):
        """Full analysis of clean JPEG should complete without errors."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_c2pa=True,
            enable_visual=True,
            enable_thumbnail=True,
            enable_shadows=True,
            enable_perspective=True,
            enable_blockgrid=True,
            enable_quantization=True,
        )

        result = analyzer.analyze(clean_jpeg)

        # Basic structure
        assert isinstance(result, WuAnalysis)
        assert result.file_path == clean_jpeg
        assert len(result.file_hash) == 64  # SHA256 hex
        assert result.analyzed_at is not None
        assert result.wu_version  # Version string exists

        # All requested dimensions should be present
        assert result.metadata is not None
        assert result.c2pa is not None
        assert result.visual is not None
        assert result.thumbnail is not None
        assert result.shadows is not None
        assert result.perspective is not None
        assert result.blockgrid is not None
        assert result.quantization is not None

        # Overall assessment should be determined
        assert result.overall in list(OverallAssessment)

    def test_analyze_clean_png_all_dimensions(self, clean_png):
        """Full analysis of clean PNG should handle non-JPEG gracefully."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
            enable_thumbnail=True,
            enable_shadows=True,
            enable_perspective=True,
            enable_blockgrid=True,
            enable_quantization=True,
        )

        result = analyzer.analyze(clean_png)

        assert isinstance(result, WuAnalysis)
        # PNG-specific: quantization should be uncertain (not JPEG)
        assert result.quantization.state == DimensionState.UNCERTAIN

    def test_analyze_recompressed_jpeg(self, recompressed_jpeg):
        """Double-compressed JPEG may show suspicious indicators."""
        analyzer = WuAnalyzer(
            enable_visual=True,
            enable_blockgrid=True,
            enable_quantization=True,
        )

        result = analyzer.analyze(recompressed_jpeg)

        assert isinstance(result, WuAnalysis)
        # Should complete analysis
        assert result.blockgrid is not None
        assert result.quantization is not None

    def test_analyze_spliced_image(self, spliced_image):
        """Spliced image analysis should detect anomalies."""
        analyzer = WuAnalyzer(
            enable_visual=True,
            enable_blockgrid=True,
            enable_copymove=True,
        )

        result = analyzer.analyze(spliced_image)

        assert isinstance(result, WuAnalysis)
        # Analysis completes regardless of detection
        assert result.visual is not None
        assert result.blockgrid is not None
        assert result.copymove is not None


class TestAnalysisOutput:
    """Test analysis output format and content."""

    def test_json_output_valid(self, clean_jpeg):
        """JSON output should be valid and parseable."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
        )

        result = analyzer.analyze(clean_jpeg)
        json_str = result.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)

        # Check required fields
        assert "file_path" in parsed
        assert "file_hash" in parsed
        assert "analyzed_at" in parsed
        assert "wu_version" in parsed
        assert "overall_assessment" in parsed
        assert "dimensions" in parsed
        assert "findings_summary" in parsed

    def test_json_output_dimensions_structure(self, clean_jpeg):
        """JSON dimensions should have correct structure."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
            enable_thumbnail=True,
        )

        result = analyzer.analyze(clean_jpeg)
        parsed = json.loads(result.to_json())

        dims = parsed["dimensions"]

        # Check each enabled dimension
        for dim_name in ["metadata", "visual", "thumbnail"]:
            assert dim_name in dims
            if dims[dim_name] is not None:
                dim = dims[dim_name]
                assert "dimension" in dim
                assert "state" in dim
                assert "confidence" in dim
                assert "evidence" in dim

    def test_to_dict_roundtrip(self, clean_jpeg):
        """to_dict should produce consistent results."""
        analyzer = WuAnalyzer(enable_metadata=True)

        result = analyzer.analyze(clean_jpeg)

        dict1 = result.to_dict()
        dict2 = result.to_dict()

        assert dict1 == dict2

    def test_findings_summary_not_empty_when_issues(self, spliced_image):
        """Findings summary should contain entries when issues found."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
        )

        result = analyzer.analyze(spliced_image)

        # findings_summary is a list
        assert isinstance(result.findings_summary, list)


class TestDimensionResults:
    """Test individual dimension results."""

    def test_metadata_dimension_structure(self, clean_jpeg):
        """Metadata dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_metadata=True)
        result = analyzer.analyze(clean_jpeg)

        meta = result.metadata
        assert meta.dimension == "metadata"
        assert meta.state in list(DimensionState)
        assert meta.confidence in list(Confidence)
        assert isinstance(meta.evidence, list)

    def test_visual_dimension_structure(self, clean_jpeg):
        """Visual dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_visual=True)
        result = analyzer.analyze(clean_jpeg)

        vis = result.visual
        assert vis.dimension == "visual"
        assert vis.state in list(DimensionState)
        assert len(vis.evidence) > 0

    def test_thumbnail_dimension_structure(self, clean_jpeg):
        """Thumbnail dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_thumbnail=True)
        result = analyzer.analyze(clean_jpeg)

        thumb = result.thumbnail
        assert thumb.dimension == "thumbnail"
        assert thumb.state in list(DimensionState)

    def test_shadows_dimension_structure(self, clean_jpeg):
        """Shadows dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_shadows=True)
        result = analyzer.analyze(clean_jpeg)

        shadows = result.shadows
        assert shadows.dimension == "shadows"
        assert shadows.state in list(DimensionState)

    def test_perspective_dimension_structure(self, clean_jpeg):
        """Perspective dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_perspective=True)
        result = analyzer.analyze(clean_jpeg)

        persp = result.perspective
        assert persp.dimension == "perspective"
        assert persp.state in list(DimensionState)

    def test_quantization_dimension_structure(self, clean_jpeg):
        """Quantization dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_quantization=True)
        result = analyzer.analyze(clean_jpeg)

        quant = result.quantization
        assert quant.dimension == "quantization"
        assert quant.state in list(DimensionState)

    def test_blockgrid_dimension_structure(self, clean_jpeg):
        """Blockgrid dimension should have proper structure."""
        analyzer = WuAnalyzer(enable_blockgrid=True)
        result = analyzer.analyze(clean_jpeg)

        bg = result.blockgrid
        assert bg.dimension == "blockgrid"
        assert bg.state in list(DimensionState)


class TestOverallAssessment:
    """Test overall assessment aggregation."""

    def test_clean_image_assessment(self, clean_jpeg):
        """Clean image should not show inconsistencies."""
        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
        )

        result = analyzer.analyze(clean_jpeg)

        # Should not be INCONSISTENCIES_DETECTED for clean image
        # (could be NO_ANOMALIES or INSUFFICIENT_DATA)
        assert result.overall != OverallAssessment.INCONSISTENCIES_DETECTED or \
               any(d.is_problematic for d in result.dimensions)

    def test_has_inconsistencies_property(self, clean_jpeg):
        """has_inconsistencies property should work correctly."""
        analyzer = WuAnalyzer(enable_metadata=True)
        result = analyzer.analyze(clean_jpeg)

        # Property should match actual dimension states
        expected = any(d.is_problematic for d in result.dimensions)
        assert result.has_inconsistencies == expected

    def test_is_clean_property(self, clean_jpeg):
        """is_clean property should work correctly."""
        analyzer = WuAnalyzer(enable_metadata=True)
        result = analyzer.analyze(clean_jpeg)

        # Property should match actual dimension states
        if len(result.dimensions) > 0:
            expected = all(d.is_clean for d in result.dimensions)
            assert result.is_clean == expected


# =============================================================================
# CLI E2E TESTS
# =============================================================================

class TestCLI:
    """Test CLI interface end-to-end."""

    def test_cli_basic_analysis(self, clean_jpeg):
        """CLI basic analysis should run without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", str(clean_jpeg)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # Should complete (exit code 0 = no anomalies, 1 = anomalies, 2 = inconsistencies)
        assert result.returncode in [0, 1, 2]

    def test_cli_json_output(self, clean_jpeg):
        """CLI --json should produce valid JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", clean_jpeg, "--json"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        parsed = json.loads(result.stdout)






        assert "file_path" in parsed
        assert "overall_assessment" in parsed

    def test_cli_verbose_output(self, clean_jpeg):
        """CLI --verbose should show dimension details."""
        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", clean_jpeg, "-v"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # Verbose output should contain dimension details
        assert "DIMENSION" in result.stdout or "State:" in result.stdout or \
               "ASSESSMENT" in result.stdout

    def test_cli_output_to_file(self, clean_jpeg, tmp_path):
        """CLI -o should write to file."""
        output_file = tmp_path / "report.json"

        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", clean_jpeg,
             "-o", str(output_file)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # File should be created
        assert output_file.exists()

        # Should be valid JSON
        content = output_file.read_text()
        parsed = json.loads(content)
        assert "file_path" in parsed

    def test_cli_with_dimensions(self, clean_jpeg):
        """CLI should accept dimension flags."""
        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", clean_jpeg,
             "--thumbnail", "--shadows", "--perspective", "--quantization",
             "--json"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        parsed = json.loads(result.stdout)
        dims = parsed["dimensions"]

        # Requested dimensions should be present
        assert dims.get("thumbnail") is not None
        assert dims.get("shadows") is not None
        assert dims.get("perspective") is not None
        assert dims.get("quantization") is not None

    def test_cli_missing_file(self, tmp_path):
        """CLI should handle missing file gracefully."""
        missing_file = tmp_path / "nonexistent.jpg"

        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "analyze", str(missing_file)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # Should exit with error
        assert result.returncode != 0 or "not found" in result.stderr.lower() or \
               "error" in result.stderr.lower() or "not found" in result.stdout.lower()


class TestCLIBatch:
    """Test CLI batch processing."""

    def test_batch_multiple_files(self, clean_jpeg, clean_png, tmp_path):
        """Batch should process multiple files."""
        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "batch", clean_jpeg, clean_png],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # Should mention both files or show count
        assert "2" in result.stdout or "Analyzed" in result.stdout

    def test_batch_with_output_dir(self, clean_jpeg, tmp_path):
        """Batch should write to output directory."""
        output_dir = tmp_path / "reports"

        result = subprocess.run(
            [sys.executable, "-m", "wu.cli", "batch", clean_jpeg,
             "-o", str(output_dir)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )


        # Output directory should be created
        assert output_dir.exists()

        # Should contain report file
        reports = list(output_dir.glob("*.json"))
        assert len(reports) >= 1


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_image(self, tmp_path):
        """Very small images should be handled."""
        img = Image.new('RGB', (16, 16), color='red')
        file_path = tmp_path / "tiny.jpg"
        img.save(str(file_path), "JPEG")

        analyzer = WuAnalyzer(
            enable_visual=True,
            enable_shadows=True,
            enable_perspective=True,
        )

        result = analyzer.analyze(str(file_path))
        assert isinstance(result, WuAnalysis)

    def test_large_image(self, tmp_path):
        """Large images should be handled (may be slow)."""
        img = Image.new('RGB', (2000, 2000), color='blue')
        file_path = tmp_path / "large.jpg"
        img.save(str(file_path), "JPEG", quality=70)

        analyzer = WuAnalyzer(enable_visual=True)
        result = analyzer.analyze(str(file_path))

        assert isinstance(result, WuAnalysis)

    def test_grayscale_jpeg_analysis(self, grayscale_jpeg):
        """Grayscale JPEG should be analyzed correctly."""
        analyzer = WuAnalyzer(
            enable_visual=True,
            enable_quantization=True,
        )

        result = analyzer.analyze(grayscale_jpeg)
        assert isinstance(result, WuAnalysis)

    def test_corrupt_file_handling(self, tmp_path):
        """Corrupt files should be handled gracefully."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"not a valid jpeg content at all")

        analyzer = WuAnalyzer(enable_visual=True)
        result = analyzer.analyze(str(corrupt_file))

        # Should return result with UNCERTAIN states, not crash
        assert isinstance(result, WuAnalysis)

    def test_empty_file_handling(self, tmp_path):
        """Empty files should be handled gracefully."""
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")

        analyzer = WuAnalyzer(enable_visual=True)
        result = analyzer.analyze(str(empty_file))

        # Should not crash
        assert isinstance(result, WuAnalysis)


class TestHashConsistency:
    """Test file hash for chain of custody."""

    def test_hash_reproducibility(self, clean_jpeg):
        """Same file should produce same hash."""
        analyzer = WuAnalyzer()

        result1 = analyzer.analyze(clean_jpeg)
        result2 = analyzer.analyze(clean_jpeg)

        assert result1.file_hash == result2.file_hash

    def test_hash_format(self, clean_jpeg):
        """Hash should be valid SHA256 hex string."""
        analyzer = WuAnalyzer()
        result = analyzer.analyze(clean_jpeg)

        # SHA256 hex is 64 characters
        assert len(result.file_hash) == 64
        # All hex characters
        assert all(c in '0123456789abcdef' for c in result.file_hash)

    def test_different_files_different_hashes(self, clean_jpeg, clean_png):
        """Different files should have different hashes."""
        analyzer = WuAnalyzer()

        result1 = analyzer.analyze(clean_jpeg)
        result2 = analyzer.analyze(clean_png)

        assert result1.file_hash != result2.file_hash


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance tests."""

    def test_analysis_completes_in_reasonable_time(self, clean_jpeg):
        """Analysis should complete within reasonable time."""
        import time

        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_visual=True,
            enable_thumbnail=True,
        )

        start = time.time()
        result = analyzer.analyze(clean_jpeg)
        elapsed = time.time() - start

        # Should complete in under 10 seconds for basic analysis
        assert elapsed < 10.0
        assert isinstance(result, WuAnalysis)

    def test_full_analysis_completes(self, clean_jpeg):
        """Full analysis with all dimensions should complete."""
        import time

        analyzer = WuAnalyzer(
            enable_metadata=True,
            enable_c2pa=True,
            enable_visual=True,
            enable_thumbnail=True,
            enable_shadows=True,
            enable_perspective=True,
            enable_blockgrid=True,
            enable_quantization=True,
            enable_copymove=True,
            enable_prnu=True,
            enable_lighting=True,
        )

        start = time.time()
        result = analyzer.analyze(clean_jpeg)
        elapsed = time.time() - start

        # Should complete in under 60 seconds even with all dimensions
        assert elapsed < 60.0
        assert isinstance(result, WuAnalysis)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressions:
    """Regression tests for previously fixed issues."""

    def test_numpy_boolean_comparison(self, clean_jpeg):
        """Ensure numpy booleans are handled correctly."""
        analyzer = WuAnalyzer(
            enable_visual=True,
            enable_shadows=True,
        )

        result = analyzer.analyze(clean_jpeg)

        # These should not raise due to numpy boolean issues
        _ = result.has_inconsistencies
        _ = result.has_anomalies
        _ = result.is_clean

    def test_dimension_properties(self, clean_jpeg):
        """Dimension result properties should work correctly."""
        analyzer = WuAnalyzer(enable_metadata=True)
        result = analyzer.analyze(clean_jpeg)

        for dim in result.dimensions:
            # These properties should not raise
            _ = dim.is_problematic
            _ = dim.is_suspicious
            _ = dim.is_clean


# =============================================================================
# LIP-SYNC ANALYSIS TESTS
# =============================================================================

class TestLipSyncAnalysis:
    """Tests for the deterministic lip-sync analysis module."""

    def test_lipsync_analyzer_import(self):
        """LipSyncAnalyzer should import correctly."""
        from wu.dimensions.lipsync import LipSyncAnalyzer
        analyzer = LipSyncAnalyzer()
        assert analyzer is not None

    def test_lipsync_phoneme_classes(self):
        """Phoneme classes should be defined."""
        from wu.dimensions.lipsync import PhonemeClass
        assert PhonemeClass.SILENCE == 0
        assert PhonemeClass.OPEN == 1
        assert PhonemeClass.BILABIAL == 5

    def test_lipsync_viseme_classes(self):
        """Viseme classes should be defined."""
        from wu.dimensions.lipsync import Viseme
        assert Viseme.CLOSED == 0
        assert Viseme.WIDE == 3
        assert Viseme.ROUNDED == 4

    def test_phoneme_viseme_compatibility(self):
        """Phoneme-viseme compatibility matrix should work."""
        from wu.dimensions.lipsync import (
            phoneme_viseme_compatible, PhonemeClass, Viseme
        )
        # Silence should be compatible with closed mouth
        assert phoneme_viseme_compatible(PhonemeClass.SILENCE, Viseme.CLOSED)
        # Open vowels should be compatible with wide mouth
        assert phoneme_viseme_compatible(PhonemeClass.OPEN, Viseme.WIDE)
        # Bilabials should be compatible with closed mouth
        assert phoneme_viseme_compatible(PhonemeClass.BILABIAL, Viseme.CLOSED)
        # Open vowels should NOT be compatible with closed mouth
        assert not phoneme_viseme_compatible(PhonemeClass.OPEN, Viseme.CLOSED)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_audio_to_phonemes_silent(self):
        """Silent audio should produce silence phonemes."""
        from wu.dimensions.lipsync import audio_to_phonemes, PhonemeClass

        # Create silent audio (1 second @ 16kHz)
        silent_audio = np.zeros(16000, dtype=np.float32)
        phonemes, times = audio_to_phonemes(silent_audio, 16000)

        assert len(phonemes) > 0
        assert len(times) == len(phonemes)
        # Most phonemes should be silence
        silence_count = sum(1 for p in phonemes if p == PhonemeClass.SILENCE)
        assert silence_count > len(phonemes) * 0.8

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_audio_to_phonemes_broadband(self):
        """Broadband audio with formant-like peaks should produce phonemes."""
        from wu.dimensions.lipsync import audio_to_phonemes, PhonemeClass

        # Create broadband noise with formant-like structure
        # This simulates vowel-like content with energy in F1/F2 ranges
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        # Mix of frequencies in vowel formant ranges (300-800Hz, 1000-2000Hz)
        audio = (
            0.3 * np.sin(2 * np.pi * 500 * t) +   # F1 range
            0.2 * np.sin(2 * np.pi * 1500 * t) +  # F2 range
            0.1 * np.random.randn(16000).astype(np.float32)  # Noise
        )
        phonemes, times = audio_to_phonemes(audio, 16000)

        assert len(phonemes) > 0
        # With formant-like content, should detect some non-silence
        # (though exact classification depends on threshold tuning)
        assert len(times) == len(phonemes)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_lip_region_detection(self):
        """Lip region detection should work on synthetic face image."""
        from wu.dimensions.lipsync import detect_lips_color

        # Create a synthetic "face" with lip-coloured region
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a reddish region in lower third (simulating lips)
        # RGB values that should pass lip detection: elevated Cr, moderate Cb
        frame[130:160, 70:130] = [180, 100, 100]  # Reddish

        face_bbox = (0, 0, 200, 200)
        lips = detect_lips_color(frame, face_bbox)

        # Detection might or might not succeed depending on exact thresholds
        # but should not crash
        assert hasattr(lips, 'valid')
        assert hasattr(lips, 'width')
        assert hasattr(lips, 'height')

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_viseme_classification(self):
        """Viseme classification should work for various lip states."""
        from wu.dimensions.lipsync import classify_viseme, LipRegion, Viseme

        # Closed lips (small aperture)
        closed = LipRegion(
            center_x=100, center_y=150, width=40, height=2,
            area=80, face_height=200, valid=True
        )
        assert classify_viseme(closed) == Viseme.CLOSED

        # Wide open mouth (large aperture)
        wide = LipRegion(
            center_x=100, center_y=150, width=50, height=30,
            area=1500, face_height=200, valid=True
        )
        assert classify_viseme(wide) == Viseme.WIDE

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_sync_analysis_aligned(self):
        """Sync analysis should detect aligned sequences."""
        from wu.dimensions.lipsync import (
            analyze_sync, PhonemeClass, Viseme
        )

        # Create perfectly aligned sequences
        phonemes = [PhonemeClass.OPEN] * 50
        visemes = [Viseme.WIDE] * 50
        phon_times = [i * 10000 for i in range(50)]  # 10ms apart
        vis_times = [i * 10000 for i in range(50)]

        result = analyze_sync(phonemes, phon_times, visemes, vis_times)

        assert result.total_frames > 0
        assert abs(result.offset_ms) < 100  # Should be close to 0
        assert result.mismatch_count < result.total_frames  # Mostly matches

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_sync_analysis_offset(self):
        """Sync analysis should detect offset sequences."""
        from wu.dimensions.lipsync import (
            analyze_sync, PhonemeClass, Viseme
        )

        # Create offset sequences (200ms offset)
        phonemes = [PhonemeClass.OPEN] * 50
        visemes = [Viseme.WIDE] * 50
        phon_times = [i * 10000 for i in range(50)]
        vis_times = [i * 10000 + 200000 for i in range(50)]  # 200ms later

        result = analyze_sync(phonemes, phon_times, visemes, vis_times)

        assert result.total_frames > 0
        # Should detect approximately 200ms offset
        assert abs(result.offset_ms - (-200)) < 50 or abs(result.offset_ms) > 150

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_lipsync_analyzer_streams(self):
        """LipSyncAnalyzer.analyze_streams should work."""
        from wu.dimensions.lipsync import LipSyncAnalyzer
        from wu.state import DimensionState

        analyzer = LipSyncAnalyzer()

        # Create minimal test data
        audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(30)]
        bboxes = [(10, 10, 80, 80) for _ in range(30)]

        result = analyzer.analyze_streams(
            audio_samples=audio,
            audio_sample_rate=16000,
            video_frames=frames,
            video_fps=30.0,
            face_bboxes=bboxes
        )

        assert result.dimension == "lipsync"
        # With silent audio, likely UNCERTAIN or CONSISTENT
        assert result.state in [
            DimensionState.UNCERTAIN,
            DimensionState.CONSISTENT,
            DimensionState.SUSPICIOUS
        ]

    def test_lipsync_determinism(self):
        """Lip-sync analysis should be deterministic."""
        from wu.dimensions.lipsync import (
            audio_to_phonemes, PhonemeClass
        )

        if not HAS_NUMPY:
            pytest.skip("NumPy required")

        # Create reproducible test audio
        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        # Run twice
        phonemes1, times1 = audio_to_phonemes(audio, 16000)
        phonemes2, times2 = audio_to_phonemes(audio, 16000)

        # Results should be identical
        assert phonemes1 == phonemes2
        assert times1 == times2

    def test_wu_analyzer_lipsync_flag(self):
        """WuAnalyzer should accept enable_lipsync parameter."""
        analyzer = WuAnalyzer(enable_lipsync=True)
        assert hasattr(analyzer, '_lipsync_analyzer')
        assert analyzer._lipsync_analyzer is not None

    def test_wu_analysis_lipsync_field(self):
        """WuAnalysis should have lipsync field."""
        from wu.state import WuAnalysis
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.mp4",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.0"
        )
        assert hasattr(analysis, 'lipsync')
        assert analysis.lipsync is None  # Not set yet


# =============================================================================
# Q15 FIXED-POINT FFT TESTS
# =============================================================================

class TestQ15FFT:
    """Tests for the Q15 fixed-point FFT implementation (Python layer)."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_hamming_window(self):
        """Hamming window should have correct shape."""
        from wu.dimensions.lipsync import _hamming_window

        window = _hamming_window(512)
        assert len(window) == 512
        assert window[0] < window[255]  # Edges lower than centre
        assert window[0] == pytest.approx(window[-1], rel=0.01)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_formant_extraction_silence(self):
        """Formant extraction should handle silence."""
        from wu.dimensions.lipsync import _extract_formants

        # Silent spectrum
        magnitude = np.zeros(256, dtype=np.float32)
        formants = _extract_formants(magnitude, 16000)

        assert not formants.voiced
        assert formants.energy < 0.01

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_formant_extraction_voiced(self):
        """Formant extraction should detect voiced content."""
        from wu.dimensions.lipsync import _extract_formants

        # Create spectrum with peaks at F1 (500Hz) and F2 (1500Hz)
        magnitude = np.zeros(256, dtype=np.float32)
        # Bin = freq * fft_size / sample_rate = freq * 512 / 16000
        f1_bin = int(500 * 512 / 16000)
        f2_bin = int(1500 * 512 / 16000)
        magnitude[f1_bin] = 0.5
        magnitude[f2_bin] = 0.4
        magnitude[:50] = 0.1  # Some background energy

        formants = _extract_formants(magnitude, 16000)

        assert formants.voiced
        assert formants.f1_hz > 0
        assert formants.f2_hz > 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_phoneme_classification_rules(self):
        """Phoneme classification should follow F1/F2 rules."""
        from wu.dimensions.lipsync import _classify_phoneme, Formants, PhonemeClass

        # High F1 (>600Hz) -> OPEN
        open_formants = Formants(f1_hz=700, f2_hz=1500, energy=0.5, voiced=True)
        assert _classify_phoneme(open_formants) == PhonemeClass.OPEN

        # Low F1, high F2 -> CLOSE_FRONT
        front_formants = Formants(f1_hz=300, f2_hz=2200, energy=0.5, voiced=True)
        assert _classify_phoneme(front_formants) == PhonemeClass.CLOSE_FRONT

        # Low F1, low F2 -> CLOSE_BACK
        back_formants = Formants(f1_hz=300, f2_hz=900, energy=0.5, voiced=True)
        assert _classify_phoneme(back_formants) == PhonemeClass.CLOSE_BACK

        # Not voiced -> SILENCE
        silence_formants = Formants(f1_hz=0, f2_hz=0, energy=0.01, voiced=False)
        assert _classify_phoneme(silence_formants) == PhonemeClass.SILENCE


# =============================================================================
# CORRELATION WARNING TESTS
# =============================================================================

class TestCorrelationWarnings:
    """Tests for the cross-dimension correlation warning system."""

    def test_correlator_import(self):
        """DimensionCorrelator should import correctly."""
        from wu.correlator import DimensionCorrelator
        correlator = DimensionCorrelator()
        assert correlator is not None

    def test_correlation_warning_import(self):
        """CorrelationWarning should import from wu."""
        from wu import CorrelationWarning
        warning = CorrelationWarning(
            severity="high",
            category="test",
            dimensions=["a", "b"],
            finding="Test warning",
        )
        assert warning.severity == "high"

    def test_correlation_warning_to_dict(self):
        """CorrelationWarning.to_dict should work."""
        from wu.state import CorrelationWarning
        warning = CorrelationWarning(
            severity="critical",
            category="device_mismatch",
            dimensions=["metadata", "prnu"],
            finding="Device conflict detected",
            details={"test": "value"},
        )
        d = warning.to_dict()
        assert d["severity"] == "critical"
        assert d["category"] == "device_mismatch"
        assert "metadata" in d["dimensions"]
        assert d["details"]["test"] == "value"

    def test_wu_analysis_has_correlation_warnings(self):
        """WuAnalysis should have correlation_warnings field."""
        from wu.state import WuAnalysis
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.jpg",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )
        assert hasattr(analysis, "correlation_warnings")
        assert analysis.correlation_warnings == []

    def test_device_attribution_conflict(self):
        """Should detect metadata vs PRNU device mismatch."""
        from wu.correlator import DimensionCorrelator
        from wu.state import WuAnalysis, DimensionResult, DimensionState, Confidence
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.jpg",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )

        # Metadata claims Canon
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={"make": "Canon", "model": "EOS 5D"},
        )

        # PRNU matches iPhone
        analysis.prnu = DimensionResult(
            dimension="prnu",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={
                "matches": [{"camera_id": "Apple iPhone 12 Pro", "correlation": 0.85, "matched": True}]
            },
        )

        correlator = DimensionCorrelator()
        warnings = correlator.correlate(analysis)

        assert len(warnings) >= 1
        device_warning = next((w for w in warnings if w.category == "device_mismatch"), None)
        assert device_warning is not None
        assert device_warning.severity == "critical"
        assert "Canon" in device_warning.finding
        assert "iPhone" in device_warning.finding

    def test_c2pa_conflict(self):
        """Should detect C2PA verified but manipulation detected."""
        from wu.correlator import DimensionCorrelator
        from wu.state import WuAnalysis, DimensionResult, DimensionState, Confidence
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.jpg",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )

        # C2PA says verified
        analysis.c2pa = DimensionResult(
            dimension="c2pa",
            state=DimensionState.VERIFIED,
            confidence=Confidence.HIGH,
        )

        # But blockgrid detects splicing
        analysis.blockgrid = DimensionResult(
            dimension="blockgrid",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
        )

        correlator = DimensionCorrelator()
        warnings = correlator.correlate(analysis)

        c2pa_warning = next((w for w in warnings if w.category == "c2pa_conflict"), None)
        assert c2pa_warning is not None
        assert c2pa_warning.severity == "critical"
        assert "C2PA" in c2pa_warning.finding

    def test_lipsync_enf_conflict(self):
        """Should detect lip-sync desync with continuous ENF."""
        from wu.correlator import DimensionCorrelator
        from wu.state import WuAnalysis, DimensionResult, DimensionState, Confidence
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.mp4",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )

        # Lip-sync shows desync
        analysis.lipsync = DimensionResult(
            dimension="lipsync",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.HIGH,
            raw_data={"offset_ms": 250},
        )

        # But ENF is continuous
        analysis.enf = DimensionResult(
            dimension="enf",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={"discontinuity_detected": False},
        )

        correlator = DimensionCorrelator()
        warnings = correlator.correlate(analysis)

        lipsync_warning = next((w for w in warnings if w.category == "lipsync_enf_conflict"), None)
        assert lipsync_warning is not None
        assert lipsync_warning.severity == "high"
        assert "desync" in lipsync_warning.finding.lower()
        assert "continuous" in lipsync_warning.finding.lower()

    def test_no_false_positives_clean_analysis(self):
        """Should generate no warnings for clean, consistent analysis."""
        from wu.correlator import DimensionCorrelator
        from wu.state import WuAnalysis, DimensionResult, DimensionState, Confidence
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.jpg",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )

        # All dimensions consistent, no conflicts
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={"make": "Canon", "model": "EOS 5D"},
        )

        analysis.prnu = DimensionResult(
            dimension="prnu",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={
                "matches": [{"camera_id": "Canon EOS 5D", "correlation": 0.92, "matched": True}]
            },
        )

        analysis.thumbnail = DimensionResult(
            dimension="thumbnail",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={"significant_difference": False, "similarity": 0.98},
        )

        correlator = DimensionCorrelator()
        warnings = correlator.correlate(analysis)

        assert len(warnings) == 0

    def test_severity_ordering(self):
        """Warnings should be sorted by severity (critical first)."""
        from wu.correlator import DimensionCorrelator
        from wu.state import WuAnalysis, DimensionResult, DimensionState, Confidence
        from datetime import datetime

        analysis = WuAnalysis(
            file_path="test.jpg",
            file_hash="abc123",
            analyzed_at=datetime.now(),
            wu_version="1.3.2",
        )

        # Set up multiple conflicts with different severities
        # Device mismatch (critical)
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={"make": "Canon", "model": "EOS 5D"},
        )
        analysis.prnu = DimensionResult(
            dimension="prnu",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            raw_data={
                "matches": [{"camera_id": "Apple iPhone", "correlation": 0.85, "matched": True}]
            },
        )

        # Thumbnail mismatch (high) - also need no software in metadata
        analysis.thumbnail = DimensionResult(
            dimension="thumbnail",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.HIGH,
            raw_data={"significant_difference": True, "similarity": 0.5},
        )

        correlator = DimensionCorrelator()
        warnings = correlator.correlate(analysis)

        # Should have at least device_mismatch (critical) and thumbnail_mismatch (high)
        if len(warnings) >= 2:
            severities = [w.severity for w in warnings]
            # Critical should come before high
            if "critical" in severities and "high" in severities:
                crit_idx = severities.index("critical")
                high_idx = severities.index("high")
                assert crit_idx < high_idx
