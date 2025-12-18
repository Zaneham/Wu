"""
Tests for epistemic state model.

Tests the core data structures that represent forensic findings.
"""

import pytest
from datetime import datetime
from wu.state import (
    DimensionState,
    Confidence,
    Evidence,
    DimensionResult,
    OverallAssessment,
    WuAnalysis,
)


class TestDimensionState:
    """Test DimensionState enum."""

    def test_all_states_have_values(self):
        """All states have string values."""
        assert DimensionState.CONSISTENT.value == "consistent"
        assert DimensionState.INCONSISTENT.value == "inconsistent"
        assert DimensionState.SUSPICIOUS.value == "suspicious"
        assert DimensionState.UNCERTAIN.value == "uncertain"
        assert DimensionState.VERIFIED.value == "verified"
        assert DimensionState.TAMPERED.value == "tampered"
        assert DimensionState.MISSING.value == "missing"
        assert DimensionState.INVALID.value == "invalid"

    def test_state_count(self):
        """Verify expected number of states."""
        assert len(DimensionState) == 8


class TestConfidence:
    """Test Confidence enum."""

    def test_all_levels(self):
        """All confidence levels exist."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"
        assert Confidence.NA.value == "n/a"


class TestEvidence:
    """Test Evidence dataclass."""

    def test_minimal_evidence(self):
        """Evidence with only required fields."""
        ev = Evidence(finding="Test", explanation="Test explanation")
        assert ev.finding == "Test"
        assert ev.explanation == "Test explanation"
        assert ev.contradiction is None
        assert ev.citation is None
        assert ev.timestamp is None

    def test_full_evidence(self):
        """Evidence with all fields."""
        ev = Evidence(
            finding="Device mismatch",
            explanation="Resolution exceeds device capability",
            contradiction="iPhone 6 cannot produce 8K",
            citation="Apple iPhone 6 specifications",
            timestamp="2023-06-15T10:30:00"
        )
        assert ev.contradiction == "iPhone 6 cannot produce 8K"
        assert ev.citation == "Apple iPhone 6 specifications"

    def test_evidence_to_dict_minimal(self):
        """Minimal evidence serializes correctly."""
        ev = Evidence(finding="Test", explanation="Explanation")
        d = ev.to_dict()
        assert d == {"finding": "Test", "explanation": "Explanation"}

    def test_evidence_to_dict_full(self):
        """Full evidence serializes all fields."""
        ev = Evidence(
            finding="Test",
            explanation="Explanation",
            contradiction="Contradiction",
            citation="Citation",
            timestamp="2023-01-01"
        )
        d = ev.to_dict()
        assert d["finding"] == "Test"
        assert d["contradiction"] == "Contradiction"
        assert d["citation"] == "Citation"
        assert d["timestamp"] == "2023-01-01"


class TestDimensionResult:
    """Test DimensionResult dataclass."""

    def test_minimal_result(self):
        """Result with minimal fields."""
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.dimension == "metadata"
        assert result.evidence == []
        assert result.methodology is None

    def test_result_with_evidence(self):
        """Result with evidence list."""
        ev = Evidence(finding="Test", explanation="Test")
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[ev]
        )
        assert len(result.evidence) == 1

    def test_is_problematic_inconsistent(self):
        """INCONSISTENT is problematic."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.is_problematic is True

    def test_is_problematic_tampered(self):
        """TAMPERED is problematic."""
        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.TAMPERED,
            confidence=Confidence.HIGH
        )
        assert result.is_problematic is True

    def test_is_problematic_invalid(self):
        """INVALID is problematic."""
        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.INVALID,
            confidence=Confidence.HIGH
        )
        assert result.is_problematic is True

    def test_is_problematic_consistent_false(self):
        """CONSISTENT is not problematic."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.is_problematic is False

    def test_is_suspicious(self):
        """SUSPICIOUS is suspicious."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.MEDIUM
        )
        assert result.is_suspicious is True

    def test_is_suspicious_false(self):
        """Non-suspicious states are not suspicious."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.is_suspicious is False

    def test_is_clean_consistent(self):
        """CONSISTENT is clean."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.is_clean is True

    def test_is_clean_verified(self):
        """VERIFIED is clean."""
        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.VERIFIED,
            confidence=Confidence.HIGH
        )
        assert result.is_clean is True

    def test_is_clean_inconsistent_false(self):
        """INCONSISTENT is not clean."""
        result = DimensionResult(
            dimension="test",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH
        )
        assert result.is_clean is False

    def test_to_dict(self):
        """Result serializes to dict."""
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            methodology="EXIF analysis"
        )
        d = result.to_dict()
        assert d["dimension"] == "metadata"
        assert d["state"] == "consistent"
        assert d["confidence"] == "high"
        assert d["methodology"] == "EXIF analysis"
        assert d["evidence"] == []


class TestOverallAssessment:
    """Test OverallAssessment enum."""

    def test_all_assessments(self):
        """All assessments have values."""
        assert OverallAssessment.NO_ANOMALIES.value == "no_anomalies_detected"
        assert OverallAssessment.ANOMALIES_DETECTED.value == "anomalies_detected"
        assert OverallAssessment.INCONSISTENCIES_DETECTED.value == "inconsistencies_detected"
        assert OverallAssessment.INSUFFICIENT_DATA.value == "insufficient_data"


class TestWuAnalysis:
    """Test WuAnalysis dataclass."""

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis."""
        return WuAnalysis(
            file_path="/test/file.jpg",
            file_hash="abc123" * 10 + "abcd",
            analyzed_at=datetime(2023, 6, 15, 10, 30, 0),
            wu_version="0.1.0"
        )

    def test_basic_fields(self, sample_analysis):
        """Basic fields are set."""
        assert sample_analysis.file_path == "/test/file.jpg"
        assert sample_analysis.wu_version == "0.1.0"

    def test_default_dimension_none(self, sample_analysis):
        """Dimension results default to None."""
        assert sample_analysis.metadata is None
        assert sample_analysis.visual is None
        assert sample_analysis.audio is None
        assert sample_analysis.temporal is None
        assert sample_analysis.c2pa is None

    def test_default_overall(self, sample_analysis):
        """Overall defaults to INSUFFICIENT_DATA."""
        assert sample_analysis.overall == OverallAssessment.INSUFFICIENT_DATA

    def test_dimensions_property_empty(self, sample_analysis):
        """Dimensions property returns empty list when no results."""
        assert sample_analysis.dimensions == []

    def test_dimensions_property_with_results(self, sample_analysis):
        """Dimensions property returns list of results."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        sample_analysis.visual = DimensionResult(
            dimension="visual",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert len(sample_analysis.dimensions) == 2

    def test_has_inconsistencies_false(self, sample_analysis):
        """has_inconsistencies is False when all clean."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert sample_analysis.has_inconsistencies is False

    def test_has_inconsistencies_true(self, sample_analysis):
        """has_inconsistencies is True when any problematic."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH
        )
        assert sample_analysis.has_inconsistencies is True

    def test_has_anomalies_false(self, sample_analysis):
        """has_anomalies is False when none suspicious."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert sample_analysis.has_anomalies is False

    def test_has_anomalies_true(self, sample_analysis):
        """has_anomalies is True when any suspicious."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.MEDIUM
        )
        assert sample_analysis.has_anomalies is True

    def test_is_clean_empty(self, sample_analysis):
        """is_clean is False when no dimensions."""
        assert sample_analysis.is_clean is False

    def test_is_clean_all_clean(self, sample_analysis):
        """is_clean is True when all dimensions clean."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        assert sample_analysis.is_clean is True

    def test_is_clean_mixed(self, sample_analysis):
        """is_clean is False when any not clean."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        sample_analysis.visual = DimensionResult(
            dimension="visual",
            state=DimensionState.SUSPICIOUS,
            confidence=Confidence.MEDIUM
        )
        assert sample_analysis.is_clean is False

    def test_to_dict(self, sample_analysis):
        """Analysis serializes to dict."""
        d = sample_analysis.to_dict()
        assert d["file_path"] == "/test/file.jpg"
        assert d["wu_version"] == "0.1.0"
        assert d["overall_assessment"] == "insufficient_data"
        assert "dimensions" in d
        assert d["dimensions"]["metadata"] is None

    def test_to_dict_with_dimension(self, sample_analysis):
        """Analysis with dimension result serializes correctly."""
        sample_analysis.metadata = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH
        )
        d = sample_analysis.to_dict()
        assert d["dimensions"]["metadata"] is not None
        assert d["dimensions"]["metadata"]["state"] == "consistent"

    def test_to_json(self, sample_analysis):
        """Analysis serializes to JSON string."""
        json_str = sample_analysis.to_json()
        assert isinstance(json_str, str)
        assert "file_path" in json_str
        assert "/test/file.jpg" in json_str

    def test_findings_summary(self, sample_analysis):
        """Findings summary can be set."""
        sample_analysis.findings_summary = ["Finding 1", "Finding 2"]
        assert len(sample_analysis.findings_summary) == 2
        d = sample_analysis.to_dict()
        assert d["findings_summary"] == ["Finding 1", "Finding 2"]
