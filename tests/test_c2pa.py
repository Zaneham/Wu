"""
Tests for C2PA content credentials analyzer.

Tests the cryptographic verification of content credentials.
"""

import pytest
from wu.dimensions.c2pa import C2PAAnalyzer
from wu.state import DimensionState, Confidence


class TestC2PAAnalyzerBasics:
    """Test basic C2PA analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        return C2PAAnalyzer()

    def test_analyzer_creation(self, analyzer):
        """C2PA analyzer can be created."""
        assert analyzer is not None

    def test_available_property(self, analyzer):
        """Available property indicates if c2pa library is installed."""
        # Either True or False depending on installation
        assert isinstance(analyzer.available, bool)

    def test_analyze_nonexistent_file(self, analyzer):
        """Nonexistent file returns UNCERTAIN."""
        result = analyzer.analyze("/nonexistent/file.jpg")
        assert result.dimension == "c2pa"
        assert result.state == DimensionState.UNCERTAIN
        assert result.confidence == Confidence.NA


class TestC2PAWithoutLibrary:
    """Test behavior when c2pa library is not available."""

    def test_unavailable_returns_uncertain(self):
        """When c2pa not available, returns UNCERTAIN with explanation."""
        analyzer = C2PAAnalyzer()

        if not analyzer.available:
            result = analyzer.analyze("/any/file.jpg")
            assert result.state == DimensionState.UNCERTAIN
            assert "not available" in result.evidence[0].finding.lower()


class TestC2PAStateValues:
    """Test that C2PA uses correct dimension states."""

    def test_missing_state_value(self):
        """MISSING state has correct value."""
        assert DimensionState.MISSING.value == "missing"

    def test_verified_state_value(self):
        """VERIFIED state has correct value."""
        assert DimensionState.VERIFIED.value == "verified"

    def test_tampered_state_value(self):
        """TAMPERED state has correct value."""
        assert DimensionState.TAMPERED.value == "tampered"

    def test_invalid_state_value(self):
        """INVALID state has correct value."""
        assert DimensionState.INVALID.value == "invalid"


class TestC2PAResultStructure:
    """Test C2PA result structure."""

    @pytest.fixture
    def analyzer(self):
        return C2PAAnalyzer()

    def test_result_has_dimension(self, analyzer):
        """Result has c2pa dimension name."""
        result = analyzer.analyze("/test/file.jpg")
        assert result.dimension == "c2pa"

    def test_result_has_methodology(self, analyzer):
        """Result includes methodology."""
        result = analyzer.analyze("/test/file.jpg")
        # Methodology may or may not be set depending on result type
        # but should be set for most cases
        assert result.methodology is None or "C2PA" in result.methodology

    def test_result_has_evidence(self, analyzer):
        """Result has evidence list."""
        result = analyzer.analyze("/test/file.jpg")
        assert isinstance(result.evidence, list)
        assert len(result.evidence) > 0


class TestNoCredentialsResult:
    """Test the no-credentials result."""

    @pytest.fixture
    def analyzer(self):
        return C2PAAnalyzer()

    def test_no_credentials_is_missing(self, analyzer):
        """Files without C2PA return MISSING state."""
        result = analyzer._no_credentials_result()
        assert result.state == DimensionState.MISSING

    def test_no_credentials_high_confidence(self, analyzer):
        """Missing credentials has HIGH confidence (we're sure they're missing)."""
        result = analyzer._no_credentials_result()
        assert result.confidence == Confidence.HIGH

    def test_no_credentials_explains_commonality(self, analyzer):
        """Missing credentials explains this is common."""
        result = analyzer._no_credentials_result()
        explanation = result.evidence[0].explanation.lower()
        assert "common" in explanation or "not inherently suspicious" in explanation


class TestC2PAIntegration:
    """Integration tests for C2PA with main analyzer."""

    def test_analyzer_has_c2pa(self):
        """Main analyzer includes C2PA by default."""
        from wu import WuAnalyzer
        analyzer = WuAnalyzer()
        assert analyzer._c2pa_analyzer is not None

    def test_analyzer_c2pa_disabled(self):
        """C2PA can be disabled."""
        from wu import WuAnalyzer
        analyzer = WuAnalyzer(enable_c2pa=False)
        assert analyzer._c2pa_analyzer is None

    def test_analysis_includes_c2pa_result(self):
        """Analysis includes c2pa dimension result."""
        from wu import WuAnalyzer
        analyzer = WuAnalyzer()
        result = analyzer.analyze("/test/file.jpg")
        assert result.c2pa is not None
        assert result.c2pa.dimension == "c2pa"

    def test_dimensions_includes_c2pa(self):
        """Dimensions list includes c2pa."""
        from wu import WuAnalyzer
        analyzer = WuAnalyzer()
        result = analyzer.analyze("/test/file.jpg")
        dim_names = [d.dimension for d in result.dimensions]
        assert "c2pa" in dim_names


class TestC2PAAggregation:
    """Test C2PA results aggregate correctly."""

    def test_missing_credentials_not_problematic(self):
        """MISSING credentials don't trigger problems."""
        from wu.state import DimensionResult
        from wu.aggregator import EpistemicAggregator

        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.MISSING,
            confidence=Confidence.HIGH,
        )
        aggregator = EpistemicAggregator()
        assert result.is_problematic is False
        assert result.is_suspicious is False

    def test_tampered_is_problematic(self):
        """TAMPERED credentials are problematic."""
        from wu.state import DimensionResult

        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.TAMPERED,
            confidence=Confidence.HIGH,
        )
        assert result.is_problematic is True

    def test_invalid_is_problematic(self):
        """INVALID credentials are problematic."""
        from wu.state import DimensionResult

        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.INVALID,
            confidence=Confidence.HIGH,
        )
        assert result.is_problematic is True

    def test_verified_is_clean(self):
        """VERIFIED credentials are clean."""
        from wu.state import DimensionResult

        result = DimensionResult(
            dimension="c2pa",
            state=DimensionState.VERIFIED,
            confidence=Confidence.HIGH,
        )
        assert result.is_clean is True
