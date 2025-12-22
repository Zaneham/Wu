"""
Tests for Wu CLI interface.

Tests command-line interface functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner

from wu.cli import cli, format_assessment, colorize, Colors
from wu.state import OverallAssessment


class TestCLIBasics:
    """Test basic CLI functionality."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        """CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Wu - Epistemic Media Forensics" in result.output

    def test_version(self, runner):
        """CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "wu" in result.output.lower() or "version" in result.output.lower()  # Version info shown


class TestAnalyzeCommand:
    """Test analyze command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpg content")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_analyze_help(self, runner):
        """Analyze command shows help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze a media file" in result.output

    def test_analyze_file(self, runner, temp_file):
        """Analyze command works on file."""
        result = runner.invoke(cli, ["analyze", temp_file])
        # Exit code 0 = no anomalies, 1 = anomalies, 2 = inconsistencies
        assert result.exit_code in [0, 1, 2]
        assert "WU FORENSIC ANALYSIS" in result.output

    def test_analyze_json_output(self, runner, temp_file):
        """Analyze command produces JSON."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json"])
        assert result.exit_code in [0, 1, 2]

        # Should be valid JSON
        parsed = json.loads(result.output)
        assert "file_path" in parsed
        assert "overall_assessment" in parsed

    def test_analyze_verbose(self, runner, temp_file):
        """Analyze command verbose mode."""
        result = runner.invoke(cli, ["analyze", temp_file, "-v"])
        assert result.exit_code in [0, 1, 2]
        assert "DIMENSION DETAILS" in result.output

    def test_analyze_output_to_file(self, runner, temp_file):
        """Analyze command writes to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(cli, ["analyze", temp_file, "-o", output_path])
            assert result.exit_code in [0, 1, 2]
            assert "written to" in result.output

            # File should exist and contain JSON
            content = Path(output_path).read_text()
            parsed = json.loads(content)
            assert "file_path" in parsed
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_analyze_nonexistent_file(self, runner):
        """Analyze nonexistent file fails gracefully."""
        result = runner.invoke(cli, ["analyze", "/nonexistent/file.jpg"])
        # Click should fail because file doesn't exist
        assert result.exit_code != 0


class TestBatchCommand:
    """Test batch command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def temp_files(self):
        """Create multiple temporary files."""
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(f"fake jpg content {i}".encode())
                paths.append(f.name)
        yield paths
        for p in paths:
            Path(p).unlink(missing_ok=True)

    def test_batch_help(self, runner):
        """Batch command shows help."""
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Analyze multiple files" in result.output

    def test_batch_files(self, runner, temp_files):
        """Batch command analyzes multiple files."""
        result = runner.invoke(cli, ["batch"] + temp_files)
        assert result.exit_code in [0, 1]
        assert "Analyzed 3 files" in result.output

    def test_batch_json_output(self, runner, temp_files):
        """Batch command produces JSON for each file."""
        result = runner.invoke(cli, ["batch", "--json"] + temp_files)
        assert result.exit_code in [0, 1]

        # Should have multiple JSON objects in output
        lines = [l for l in result.output.strip().split("\n") if l.startswith("{")]
        assert len(lines) >= 1  # At least some JSON

    def test_batch_output_directory(self, runner, temp_files):
        """Batch command writes to directory."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(cli, ["batch", "-o", output_dir] + temp_files)
            assert result.exit_code in [0, 1]

            # Check output files exist
            json_files = list(Path(output_dir).glob("*_wu.json"))
            assert len(json_files) == 3

    def test_batch_no_files(self, runner):
        """Batch with no files shows error."""
        result = runner.invoke(cli, ["batch"])
        assert result.exit_code != 0
        assert "No files specified" in result.output


class TestFormatsCommand:
    """Test formats command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_formats_lists_extensions(self, runner):
        """Formats command lists supported extensions."""
        result = runner.invoke(cli, ["formats"])
        assert result.exit_code == 0
        assert ".jpg" in result.output
        assert ".png" in result.output
        assert ".cr2" in result.output


class TestOutputFormatting:
    """Test output formatting helpers."""

    def test_format_assessment_inconsistent(self):
        """Inconsistent assessment formatted red."""
        result = format_assessment(OverallAssessment.INCONSISTENCIES_DETECTED)
        assert "INCONSISTENCIES" in result
        assert Colors.RED in result

    def test_format_assessment_anomalies(self):
        """Anomalies assessment formatted yellow."""
        result = format_assessment(OverallAssessment.ANOMALIES_DETECTED)
        assert "ANOMALIES" in result
        assert Colors.YELLOW in result

    def test_format_assessment_clean(self):
        """Clean assessment formatted green."""
        result = format_assessment(OverallAssessment.NO_ANOMALIES)
        assert "NO ANOMALIES" in result
        assert Colors.GREEN in result

    def test_format_assessment_insufficient(self):
        """Insufficient data formatted blue."""
        result = format_assessment(OverallAssessment.INSUFFICIENT_DATA)
        assert "INSUFFICIENT" in result
        assert Colors.BLUE in result


class TestColorize:
    """Test colorize helper."""

    def test_colorize_basic(self):
        """Basic colorization works."""
        result = colorize("test", Colors.RED)
        assert "test" in result
        assert Colors.RED in result
        assert Colors.RESET in result

    def test_colorize_bold(self):
        """Bold colorization works."""
        result = colorize("test", Colors.RED, bold=True)
        assert Colors.BOLD in result

    def test_colorize_not_bold(self):
        """Non-bold colorization."""
        result = colorize("test", Colors.RED, bold=False)
        # BOLD should not appear before color (could appear in RESET)
        assert result.startswith(Colors.RED)


class TestExitCodes:
    """Test CLI exit codes."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_exit_code_meanings(self):
        """Document expected exit codes."""
        # 0 = no anomalies
        # 1 = anomalies detected
        # 2 = inconsistencies detected
        # These are tested implicitly in analyze tests
        pass


class TestDimensionToggles:
    """Test dimension enable/disable options."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpg content")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_no_metadata_option(self, runner, temp_file):
        """--no-metadata disables metadata analysis."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json", "--no-metadata"])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        # Metadata should be null/None when disabled
        assert parsed["dimensions"]["metadata"] is None

    def test_no_c2pa_option(self, runner, temp_file):
        """--no-c2pa disables C2PA analysis."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json", "--no-c2pa"])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        # C2PA should be null/None when disabled
        assert parsed["dimensions"]["c2pa"] is None

    def test_no_visual_option(self, runner, temp_file):
        """--no-visual disables visual analysis."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json", "--no-visual"])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        # Visual should be null/None when disabled
        assert parsed["dimensions"]["visual"] is None

    def test_all_dimensions_disabled(self, runner, temp_file):
        """All dimensions can be disabled."""
        result = runner.invoke(cli, [
            "analyze", temp_file, "--json",
            "--no-metadata", "--no-c2pa", "--no-visual"
        ])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        assert parsed["dimensions"]["metadata"] is None
        assert parsed["dimensions"]["c2pa"] is None
        assert parsed["dimensions"]["visual"] is None
        assert parsed["overall_assessment"] == "insufficient_data"

    def test_default_all_enabled(self, runner, temp_file):
        """By default all dimensions are enabled."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json"])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        # All should be present (not None)
        assert parsed["dimensions"]["metadata"] is not None
        assert parsed["dimensions"]["c2pa"] is not None
        assert parsed["dimensions"]["visual"] is not None

    def test_help_shows_dimension_options(self, runner):
        """Help shows dimension toggle options."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert "--no-metadata" in result.output
        assert "--no-c2pa" in result.output
        assert "--no-visual" in result.output
        assert "--enf" in result.output

    def test_enf_option(self, runner, temp_file):
        """--enf enables ENF analysis."""
        result = runner.invoke(cli, ["analyze", temp_file, "--json", "--enf"])
        assert result.exit_code in [0, 1, 2]

        parsed = json.loads(result.output)
        # ENF should be present when enabled
        assert parsed["dimensions"]["enf"] is not None
