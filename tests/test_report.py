"""
Tests for PDF forensic report generator.

Tests court-ready report generation functionality.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from wu.report import ForensicReportGenerator, generate_report
from wu.state import (
    WuAnalysis,
    DimensionResult,
    DimensionState,
    Confidence,
    Evidence,
    OverallAssessment,
)


class TestReportGeneratorBasics:
    """Test basic report generator functionality."""

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        analysis = WuAnalysis(
            file_path="/test/evidence.jpg",
            file_hash="abc123def456" * 5 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding="Metadata consistent",
                explanation="No anomalies detected in EXIF data"
            )],
            methodology="EXIF analysis with device capability verification"
        )
        analysis.overall = OverallAssessment.NO_ANOMALIES
        analysis.findings_summary = ["No anomalies detected in: metadata"]
        return analysis

    def test_generator_creation(self):
        """Report generator can be created."""
        generator = ForensicReportGenerator()
        assert generator is not None

    def test_generate_basic_report(self, sample_analysis):
        """Basic report can be generated."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator()
            result = generator.generate(sample_analysis, output_path)

            assert result == output_path
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_with_examiner(self, sample_analysis):
        """Report with examiner info can be generated."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator()
            result = generator.generate(
                sample_analysis,
                output_path,
                examiner_name="John Smith",
                case_number="2024-001",
                notes="Preliminary analysis for discovery."
            )

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestReportWithInconsistencies:
    """Test reports with detected inconsistencies."""

    @pytest.fixture
    def inconsistent_analysis(self):
        """Create analysis with inconsistencies."""
        analysis = WuAnalysis(
            file_path="/test/suspicious.jpg",
            file_hash="def789" * 10 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[
                Evidence(
                    finding="Device claims iPhone 6",
                    explanation="Resolution is 8K (7680x4320)",
                    contradiction="iPhone 6 maximum resolution is 3264x2448 (8MP)",
                    citation="Apple iPhone 6 Technical Specifications"
                ),
                Evidence(
                    finding="AI generation signature detected",
                    explanation="Stable Diffusion parameters found in metadata"
                )
            ],
            methodology="EXIF analysis with device capability verification"
        )
        analysis.overall = OverallAssessment.INCONSISTENCIES_DETECTED
        analysis.findings_summary = [
            "[METADATA] Device claims iPhone 6: iPhone 6 cannot produce 8K",
            "[METADATA] AI generation signature detected"
        ]
        return analysis

    def test_generate_inconsistent_report(self, inconsistent_analysis):
        """Report with inconsistencies can be generated."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator()
            generator.generate(inconsistent_analysis, output_path)

            assert Path(output_path).exists()
            # File should be reasonably sized (has content)
            assert Path(output_path).stat().st_size > 1000
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestReportWithSuspicious:
    """Test reports with suspicious findings."""

    @pytest.fixture
    def suspicious_analysis(self):
        """Create analysis with suspicious findings."""
        analysis = WuAnalysis(
            file_path="/test/edited.jpg",
            file_hash="789abc" * 10 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.MEDIUM,
            evidence=[
                Evidence(
                    finding="Editing software detected: Adobe Photoshop",
                    explanation="File has been processed with editing software"
                ),
                Evidence(
                    finding="Metadata appears stripped",
                    explanation="File has minimal metadata, possibly intentionally removed"
                )
            ],
            methodology="EXIF analysis with device capability verification"
        )
        analysis.overall = OverallAssessment.ANOMALIES_DETECTED
        analysis.findings_summary = [
            "[METADATA] Suspicious: Editing software detected",
            "[METADATA] Suspicious: Metadata appears stripped"
        ]
        return analysis

    def test_generate_suspicious_report(self, suspicious_analysis):
        """Report with suspicious findings can be generated."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator()
            generator.generate(suspicious_analysis, output_path)

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestReportWithNoDimensions:
    """Test reports with no analyzed dimensions."""

    @pytest.fixture
    def empty_analysis(self):
        """Create analysis with no dimensions."""
        return WuAnalysis(
            file_path="/test/unknown.dat",
            file_hash="000000" * 10 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )

    def test_generate_empty_report(self, empty_analysis):
        """Report with no dimensions can be generated."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator()
            generator.generate(empty_analysis, output_path)

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestConvenienceFunction:
    """Test generate_report convenience function."""

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis."""
        analysis = WuAnalysis(
            file_path="/test/file.jpg",
            file_hash="abc123" * 10 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
        )
        analysis.overall = OverallAssessment.NO_ANOMALIES
        return analysis

    def test_generate_report_function(self, sample_analysis):
        """Convenience function works."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            result = generate_report(sample_analysis, output_path)
            assert result == output_path
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestReportFormatting:
    """Test report formatting helpers."""

    @pytest.fixture
    def generator(self):
        return ForensicReportGenerator()

    def test_format_assessment_no_anomalies(self, generator):
        """No anomalies formatted correctly."""
        result = generator._format_assessment(OverallAssessment.NO_ANOMALIES)
        assert "NO ANOMALIES" in result

    def test_format_assessment_anomalies(self, generator):
        """Anomalies formatted correctly."""
        result = generator._format_assessment(OverallAssessment.ANOMALIES_DETECTED)
        assert "ANOMALIES" in result
        assert "investigation" in result.lower()

    def test_format_assessment_inconsistencies(self, generator):
        """Inconsistencies formatted correctly."""
        result = generator._format_assessment(OverallAssessment.INCONSISTENCIES_DETECTED)
        assert "INCONSISTENCIES" in result
        assert "manipulation" in result.lower()

    def test_format_state_consistent(self, generator):
        """Consistent state formatted."""
        result = generator._format_state(DimensionState.CONSISTENT)
        assert result == "Consistent"

    def test_format_state_inconsistent(self, generator):
        """Inconsistent state formatted."""
        result = generator._format_state(DimensionState.INCONSISTENT)
        assert result == "INCONSISTENT"

    def test_format_state_tampered(self, generator):
        """Tampered state formatted."""
        result = generator._format_state(DimensionState.TAMPERED)
        assert result == "TAMPERED"


class TestReportStyles:
    """Test report style selection."""

    @pytest.fixture
    def generator(self):
        return ForensicReportGenerator()

    def test_finding_style_inconsistent(self, generator):
        """Inconsistent finding gets problem style."""
        style = generator._get_finding_style("Resolution impossible for device")
        assert style.name == "FindingProblem"

    def test_finding_style_suspicious(self, generator):
        """Suspicious finding gets suspicious style."""
        style = generator._get_finding_style("Editing software detected")
        assert style.name == "FindingSuspicious"

    def test_finding_style_clean(self, generator):
        """Clean finding gets clean style."""
        style = generator._get_finding_style("No issues found")
        assert style.name == "FindingClean"

    def test_style_for_state_inconsistent(self, generator):
        """Inconsistent state gets problem style."""
        style = generator._get_finding_style_for_state(DimensionState.INCONSISTENT)
        assert style.name == "FindingProblem"

    def test_style_for_state_suspicious(self, generator):
        """Suspicious state gets suspicious style."""
        style = generator._get_finding_style_for_state(DimensionState.SUSPICIOUS)
        assert style.name == "FindingSuspicious"

    def test_style_for_state_consistent(self, generator):
        """Consistent state gets clean style."""
        style = generator._get_finding_style_for_state(DimensionState.CONSISTENT)
        assert style.name == "FindingClean"


class TestA4PageSize:
    """Test A4 page size option."""

    def test_a4_report(self):
        """Report can be generated with A4 page size."""
        from reportlab.lib.pagesizes import A4

        analysis = WuAnalysis(
            file_path="/test/file.jpg",
            file_hash="abc123" * 10 + "abcd",
            analyzed_at=datetime(2024, 1, 15, 10, 30, 0),
            wu_version="0.1.0",
        )
        analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generator = ForensicReportGenerator(pagesize=A4)
            generator.generate(analysis, output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)
