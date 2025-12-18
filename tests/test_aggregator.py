"""
Tests for epistemic aggregator.

Verifies the aggregation logic that combines dimension results
into an overall assessment with proper epistemic reasoning.
"""

import pytest
from wu.aggregator import (
    EpistemicAggregator,
    CorroboratingEvidence,
    CORROBORATION_CATEGORIES,
    get_relevant_categories,
)
from wu.state import (
    DimensionResult,
    DimensionState,
    Confidence,
    Evidence,
    OverallAssessment,
)


class TestAggregatorBasic:
    """Test basic aggregation logic."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_empty_results_insufficient(self, aggregator):
        """Empty results return INSUFFICIENT_DATA."""
        result = aggregator.aggregate([])
        assert result == OverallAssessment.INSUFFICIENT_DATA

    def test_single_consistent_no_anomalies(self, aggregator):
        """Single consistent result returns NO_ANOMALIES."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            )
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.NO_ANOMALIES

    def test_single_inconsistent_detected(self, aggregator):
        """Single inconsistent result returns INCONSISTENCIES_DETECTED."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
            )
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_single_suspicious_anomalies(self, aggregator):
        """Single suspicious result returns ANOMALIES_DETECTED."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            )
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.ANOMALIES_DETECTED

    def test_all_uncertain_insufficient(self, aggregator):
        """All uncertain results return INSUFFICIENT_DATA."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INSUFFICIENT_DATA


class TestAggregatorPriority:
    """Test aggregation priority (inconsistent > suspicious > clean)."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_inconsistent_overrides_consistent(self, aggregator):
        """One inconsistent overrides multiple consistent."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="audio",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_suspicious_overrides_consistent(self, aggregator):
        """One suspicious overrides multiple consistent."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.ANOMALIES_DETECTED

    def test_inconsistent_overrides_suspicious(self, aggregator):
        """Inconsistent overrides suspicious."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INCONSISTENCIES_DETECTED


class TestAggregatorC2PAStates:
    """Test aggregation with C2PA-specific states."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_tampered_is_problematic(self, aggregator):
        """TAMPERED state triggers INCONSISTENCIES_DETECTED."""
        results = [
            DimensionResult(
                dimension="c2pa",
                state=DimensionState.TAMPERED,
                confidence=Confidence.HIGH,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_invalid_is_problematic(self, aggregator):
        """INVALID credentials triggers INCONSISTENCIES_DETECTED."""
        results = [
            DimensionResult(
                dimension="c2pa",
                state=DimensionState.INVALID,
                confidence=Confidence.HIGH,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.INCONSISTENCIES_DETECTED

    def test_verified_is_clean(self, aggregator):
        """VERIFIED credentials is clean."""
        results = [
            DimensionResult(
                dimension="c2pa",
                state=DimensionState.VERIFIED,
                confidence=Confidence.HIGH,
            ),
        ]
        result = aggregator.aggregate(results)
        assert result == OverallAssessment.NO_ANOMALIES


class TestSummaryGeneration:
    """Test human-readable summary generation."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_summary_includes_inconsistencies(self, aggregator):
        """Summary includes inconsistency findings."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Device claims iPhone 6",
                        explanation="Resolution is 4K",
                        contradiction="iPhone 6 cannot produce 4K"
                    )
                ]
            ),
        ]
        summary = aggregator.generate_summary(results)
        assert len(summary) > 0
        assert any("METADATA" in s for s in summary)
        assert any("iPhone 6" in s for s in summary)

    def test_summary_includes_suspicious(self, aggregator):
        """Summary includes suspicious findings."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
                evidence=[
                    Evidence(
                        finding="Editing software detected",
                        explanation="Adobe Photoshop found"
                    )
                ]
            ),
        ]
        summary = aggregator.generate_summary(results)
        assert any("Suspicious" in s for s in summary)

    def test_summary_notes_clean_dimensions(self, aggregator):
        """Summary notes clean dimensions."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        summary = aggregator.generate_summary(results)
        assert any("No anomalies" in s for s in summary)


class TestConfidenceCalculation:
    """Test overall confidence calculation."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_high_confidence_inconsistency(self, aggregator):
        """High confidence inconsistency yields HIGH overall."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        confidence = aggregator.calculate_overall_confidence(results)
        assert confidence == Confidence.HIGH

    def test_empty_results_na(self, aggregator):
        """Empty results yield N/A confidence."""
        confidence = aggregator.calculate_overall_confidence([])
        assert confidence == Confidence.NA

    def test_all_uncertain_na(self, aggregator):
        """All uncertain yields N/A confidence."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
            ),
        ]
        confidence = aggregator.calculate_overall_confidence(results)
        assert confidence == Confidence.NA

    def test_multiple_high_clean_is_high(self, aggregator):
        """Multiple high-confidence clean results yields HIGH."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        confidence = aggregator.calculate_overall_confidence(results)
        assert confidence == Confidence.HIGH


class TestCorroborationCategories:
    """Test corroboration category definitions."""

    def test_categories_defined(self):
        """Corroboration categories should be defined."""
        assert len(CORROBORATION_CATEGORIES) > 0

    def test_categories_have_required_fields(self):
        """Each category should have description, dimensions, and narrative."""
        for name, cat in CORROBORATION_CATEGORIES.items():
            assert "description" in cat
            assert "dimensions" in cat
            assert "narrative" in cat
            assert len(cat["dimensions"]) >= 2

    def test_get_relevant_categories(self):
        """get_relevant_categories should return correct categories."""
        metadata_cats = get_relevant_categories("metadata")
        assert len(metadata_cats) > 0
        assert "post_capture_editing" in metadata_cats

    def test_get_relevant_categories_unknown_dimension(self):
        """Unknown dimension returns empty list."""
        cats = get_relevant_categories("unknown_dimension")
        assert cats == []


class TestCorroboratingEvidence:
    """Test CorroboratingEvidence dataclass."""

    def test_corroborating_evidence_creation(self):
        """CorroboratingEvidence should be creatable."""
        corr = CorroboratingEvidence(
            category="post_capture_editing",
            category_description="Evidence of editing after initial capture",
            supporting_dimensions=["metadata", "thumbnail"],
            narrative="Test narrative",
            strength="moderate",
        )
        assert corr.category == "post_capture_editing"
        assert len(corr.supporting_dimensions) == 2
        assert corr.strength == "moderate"


class TestFindCorroboratingEvidence:
    """Test find_corroborating_evidence method."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_no_corroboration_with_clean_results(self, aggregator):
        """Clean results should produce no corroboration."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)
        assert len(corroborations) == 0

    def test_no_corroboration_with_single_problem(self, aggregator):
        """Single problematic dimension should not produce corroboration."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)
        assert len(corroborations) == 0

    def test_corroboration_with_two_problems_same_category(self, aggregator):
        """Two problems in same category should produce corroboration."""
        # metadata and thumbnail are both in post_capture_editing
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Software tag detected",
                        explanation="Adobe Photoshop found",
                    )
                ],
            ),
            DimensionResult(
                dimension="thumbnail",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Thumbnail mismatch",
                        explanation="EXIF thumbnail differs from main image",
                    )
                ],
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)
        assert len(corroborations) >= 1

        # Check that one of them is post_capture_editing
        categories = [c.category for c in corroborations]
        assert "post_capture_editing" in categories

    def test_corroboration_strength_moderate_for_two(self, aggregator):
        """Two dimensions should produce moderate strength."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="thumbnail",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)
        if corroborations:
            # Find the post_capture_editing corroboration
            for corr in corroborations:
                if corr.category == "post_capture_editing":
                    assert corr.strength == "moderate"

    def test_corroboration_strength_strong_for_three(self, aggregator):
        """Three or more dimensions should produce strong strength."""
        # metadata, thumbnail, and quantization are in post_capture_editing
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="thumbnail",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="quantization",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="visual",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)

        # Find one with 3+ dimensions
        strong_corrs = [c for c in corroborations if c.strength == "strong"]
        assert len(strong_corrs) >= 1

    def test_empty_results_no_corroboration(self, aggregator):
        """Empty results should produce no corroboration."""
        corroborations = aggregator.find_corroborating_evidence([])
        assert len(corroborations) == 0


class TestCorroborationSummary:
    """Test generate_corroboration_summary method."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_no_summary_for_clean_results(self, aggregator):
        """Clean results should produce None summary."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.CONSISTENT,
                confidence=Confidence.HIGH,
            ),
        ]
        summary = aggregator.generate_corroboration_summary(results)
        assert summary is None

    def test_summary_includes_narrative(self, aggregator):
        """Summary should include narrative text."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="thumbnail",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
        ]
        summary = aggregator.generate_corroboration_summary(results)
        if summary:
            assert "convergent" in summary.lower() or "independent" in summary.lower()

    def test_summary_mentions_dimensions(self, aggregator):
        """Summary should mention the supporting dimensions."""
        results = [
            DimensionResult(
                dimension="metadata",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
            DimensionResult(
                dimension="thumbnail",
                state=DimensionState.SUSPICIOUS,
                confidence=Confidence.MEDIUM,
            ),
        ]
        summary = aggregator.generate_corroboration_summary(results)
        if summary:
            assert "metadata" in summary or "thumbnail" in summary

    def test_summary_empty_results(self, aggregator):
        """Empty results should produce None summary."""
        summary = aggregator.generate_corroboration_summary([])
        assert summary is None


class TestGeometricCorroboration:
    """Test corroboration for geometric inconsistencies."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_shadows_and_lighting_corroborate(self, aggregator):
        """Shadows and lighting issues should corroborate as geometric."""
        results = [
            DimensionResult(
                dimension="shadows",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Multiple shadow directions",
                        explanation="Shadows point in different directions",
                    )
                ],
            ),
            DimensionResult(
                dimension="lighting",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Inconsistent light sources",
                        explanation="Different faces lit from different angles",
                    )
                ],
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)

        categories = [c.category for c in corroborations]
        assert "geometric_inconsistency" in categories


class TestSplicingCorroboration:
    """Test corroboration for splicing detection."""

    @pytest.fixture
    def aggregator(self):
        return EpistemicAggregator()

    def test_blockgrid_and_copymove_corroborate(self, aggregator):
        """Block grid and copy-move issues should corroborate as splicing."""
        results = [
            DimensionResult(
                dimension="blockgrid",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Grid misalignment detected",
                        explanation="JPEG blocks do not align",
                    )
                ],
            ),
            DimensionResult(
                dimension="copymove",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Cloned regions found",
                        explanation="Duplicate content detected",
                    )
                ],
            ),
        ]
        corroborations = aggregator.find_corroborating_evidence(results)

        categories = [c.category for c in corroborations]
        assert "splicing_detected" in categories
