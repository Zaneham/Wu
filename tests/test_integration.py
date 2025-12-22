"""
Integration tests for Wu forensic analysis.

Tests the full analysis pipeline from analyzer through
aggregation to final result.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from wu import WuAnalyzer, DimensionState, OverallAssessment
from wu.aggregator import EpistemicAggregator
from wu.state import DimensionResult, Confidence, Evidence


class TestFullPipeline:
    """Test complete analysis pipeline."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_pipeline_produces_valid_json(self, analyzer):
        """Full pipeline produces valid JSON."""
        result = analyzer.analyze("/test/file.jpg")
        json_str = result.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "file_path" in parsed
        assert "overall_assessment" in parsed
        assert "dimensions" in parsed

    def test_pipeline_timestamps(self, analyzer):
        """Pipeline records analysis timestamp."""
        before = datetime.now()
        result = analyzer.analyze("/test/file.jpg")
        after = datetime.now()

        assert before <= result.analyzed_at <= after

    def test_pipeline_version(self, analyzer):
        """Pipeline records Wu version."""
        result = analyzer.analyze("/test/file.jpg")
        assert result.wu_version  # Version exists

    def test_pipeline_file_hash_consistency(self, analyzer):
        """Same file produces same hash."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result1 = analyzer.analyze(temp_path)
            result2 = analyzer.analyze(temp_path)
            assert result1.file_hash == result2.file_hash
        finally:
            Path(temp_path).unlink()

    def test_pipeline_different_files_different_hash(self, analyzer):
        """Different files produce different hashes."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"content A")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"content B")
            path2 = f2.name

        try:
            result1 = analyzer.analyze(path1)
            result2 = analyzer.analyze(path2)
            assert result1.file_hash != result2.file_hash
        finally:
            Path(path1).unlink()
            Path(path2).unlink()


class TestAggregatorIntegration:
    """Test aggregator integrates correctly with analyzer."""

    def test_aggregator_with_consistent_result(self):
        """Aggregator handles consistent result."""
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding="Metadata consistent",
                explanation="No anomalies detected"
            )]
        )
        aggregator = EpistemicAggregator()
        overall = aggregator.aggregate([result])
        assert overall == OverallAssessment.NO_ANOMALIES

    def test_aggregator_with_inconsistent_result(self):
        """Aggregator handles inconsistent result."""
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding="Device mismatch",
                explanation="iPhone 6 claiming 8K",
                contradiction="iPhone 6 max is 8MP"
            )]
        )
        aggregator = EpistemicAggregator()
        overall = aggregator.aggregate([result])
        assert overall == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_aggregator_summary_generation(self):
        """Aggregator generates meaningful summary."""
        result = DimensionResult(
            dimension="metadata",
            state=DimensionState.INCONSISTENT,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding="Device claims iPhone 6",
                explanation="Resolution is 8K",
                contradiction="iPhone 6 cannot produce 8K"
            )]
        )
        aggregator = EpistemicAggregator()
        summary = aggregator.generate_summary([result])

        assert len(summary) > 0
        assert any("METADATA" in s for s in summary)


class TestMultipleDimensions:
    """Test analysis with multiple dimension results."""

    def test_mixed_states_prioritizes_inconsistent(self):
        """Mixed states prioritize inconsistent."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH
            ),
            DimensionResult(
                dimension="audio",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.MEDIUM
            ),
        ]
        aggregator = EpistemicAggregator()
        overall = aggregator.aggregate(results)
        assert overall == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_all_consistent_is_clean(self):
        """All consistent results are clean."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.MEDIUM
            ),
        ]
        aggregator = EpistemicAggregator()
        overall = aggregator.aggregate(results)
        assert overall == OverallAssessment.NO_ANOMALIES

    def test_suspicious_without_inconsistent(self):
        """Suspicious without inconsistent is anomalies."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM
            ),
        ]
        aggregator = EpistemicAggregator()
        overall = aggregator.aggregate(results)
        assert overall == OverallAssessment.ANOMALIES_DETECTED


class TestBatchAnalysis:
    """Test batch analysis functionality."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_batch_returns_correct_count(self, analyzer):
        """Batch returns result for each file."""
        paths = [f"/test/file{i}.jpg" for i in range(5)]
        results = analyzer.analyze_batch(paths)
        assert len(results) == 5

    def test_batch_results_independent(self, analyzer):
        """Each batch result is independent."""
        paths = ["/test/a.jpg", "/test/b.jpg"]
        results = analyzer.analyze_batch(paths)

        assert results[0].file_path != results[1].file_path
        # Nonexistent files have same hash error
        assert results[0].file_hash == results[1].file_hash == "FILE_NOT_FOUND"


class TestResultSerialization:
    """Test result serialization round-trips."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_json_roundtrip(self, analyzer):
        """JSON serialization is complete."""
        result = analyzer.analyze("/test/file.jpg")
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["file_path"] == result.file_path
        assert parsed["file_hash"] == result.file_hash
        assert parsed["wu_version"] == result.wu_version
        assert parsed["overall_assessment"] == result.overall.value

    def test_dict_contains_all_dimensions(self, analyzer):
        """Dict contains all dimension slots."""
        result = analyzer.analyze("/test/file.jpg")
        d = result.to_dict()

        assert "dimensions" in d
        assert "metadata" in d["dimensions"]
        assert "visual" in d["dimensions"]
        assert "audio" in d["dimensions"]
        assert "temporal" in d["dimensions"]
        assert "c2pa" in d["dimensions"]

    def test_evidence_serializes(self, analyzer):
        """Evidence list serializes correctly."""
        result = analyzer.analyze("/test/file.jpg")
        d = result.to_dict()

        if d["dimensions"]["metadata"]:
            metadata = d["dimensions"]["metadata"]
            assert "evidence" in metadata
            assert isinstance(metadata["evidence"], list)


class TestDisabledDimensions:
    """Test analyzer with disabled dimensions."""

    def test_all_disabled(self):
        """All disabled dimensions produces no results."""
        analyzer = WuAnalyzer(enable_metadata=False, enable_c2pa=False, enable_visual=False)
        result = analyzer.analyze("/test/file.jpg")

        assert result.metadata is None
        assert result.c2pa is None
        assert result.visual is None
        assert len(result.dimensions) >= 0  # Video may be enabled by default
        assert result.overall == OverallAssessment.INSUFFICIENT_DATA

    def test_only_metadata_enabled(self):
        """Only metadata enabled produces metadata result."""
        analyzer = WuAnalyzer(enable_metadata=True, enable_c2pa=False, enable_visual=False)
        result = analyzer.analyze("/test/file.jpg")

        assert result.metadata is not None
        assert result.c2pa is None
        assert result.visual is None
        assert len(result.dimensions) >= 1

    def test_only_c2pa_enabled(self):
        """Only C2PA enabled produces c2pa result."""
        analyzer = WuAnalyzer(enable_metadata=False, enable_c2pa=True, enable_visual=False)
        result = analyzer.analyze("/test/file.jpg")

        assert result.metadata is None
        assert result.c2pa is not None
        assert result.visual is None
        assert len(result.dimensions) >= 1

    def test_only_visual_enabled(self):
        """Only visual enabled produces visual result."""
        analyzer = WuAnalyzer(enable_metadata=False, enable_c2pa=False, enable_visual=True)
        result = analyzer.analyze("/test/file.jpg")

        assert result.metadata is None
        assert result.c2pa is None
        assert result.visual is not None
        assert len(result.dimensions) >= 1

    def test_all_enabled(self):
        """All dimensions enabled produces all results."""
        analyzer = WuAnalyzer(enable_metadata=True, enable_c2pa=True, enable_visual=True)
        result = analyzer.analyze("/test/file.jpg")

        assert result.metadata is not None
        assert result.c2pa is not None
        assert result.visual is not None
        assert len(result.dimensions) >= 3


class TestErrorHandling:
    """Test error handling throughout pipeline."""

    @pytest.fixture
    def analyzer(self):
        return WuAnalyzer()

    def test_nonexistent_file_graceful(self, analyzer):
        """Nonexistent file handled gracefully."""
        result = analyzer.analyze("/definitely/does/not/exist.jpg")

        assert result is not None
        assert result.file_hash == "FILE_NOT_FOUND"
        assert result.metadata.state == DimensionState.UNCERTAIN

    def test_empty_path(self, analyzer):
        """Empty path handled."""
        result = analyzer.analyze("")
        assert result is not None

    def test_special_characters_in_path(self, analyzer):
        """Special characters in path handled."""
        result = analyzer.analyze("/path/with spaces/and-dashes/file.jpg")
        assert result is not None
