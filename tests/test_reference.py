"""
Tests for reference test vector verification system.

These tests verify that the verification system itself works correctly,
ensuring that legitimate installations pass and modified code would fail.
"""

import pytest
import tempfile
from pathlib import Path

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

from wu.reference import (
    ReferenceVerifier,
    ReferenceVector,
    VerificationResult,
    verify_installation,
    generate_solid_blue_jpeg,
    generate_gradient_jpeg,
    generate_checkerboard_jpeg,
    generate_solid_red_png,
    generate_geometry_test_jpeg,
    IMAGE_GENERATORS,
    REFERENCE_VECTORS,
)


pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PIL,
    reason="numpy and PIL required for reference tests"
)


class TestImageGenerators:
    """Tests for deterministic image generators."""

    def test_solid_blue_generates_file(self, tmp_path):
        path = str(tmp_path / "blue.jpg")
        result = generate_solid_blue_jpeg(path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_gradient_generates_file(self, tmp_path):
        path = str(tmp_path / "gradient.jpg")
        result = generate_gradient_jpeg(path)
        assert Path(result).exists()

    def test_checkerboard_generates_file(self, tmp_path):
        path = str(tmp_path / "checker.jpg")
        result = generate_checkerboard_jpeg(path)
        assert Path(result).exists()

    def test_solid_red_png_generates_file(self, tmp_path):
        path = str(tmp_path / "red.png")
        result = generate_solid_red_png(path)
        assert Path(result).exists()

    def test_geometry_test_generates_file(self, tmp_path):
        path = str(tmp_path / "geometry.jpg")
        result = generate_geometry_test_jpeg(path)
        assert Path(result).exists()

    def test_solid_blue_is_deterministic(self, tmp_path):
        """Same parameters should produce identical files."""
        path1 = str(tmp_path / "blue1.jpg")
        path2 = str(tmp_path / "blue2.jpg")

        generate_solid_blue_jpeg(path1, quality=85)
        generate_solid_blue_jpeg(path2, quality=85)

        # Files should be identical
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            assert f1.read() == f2.read()

    def test_gradient_is_deterministic(self, tmp_path):
        """Same parameters should produce identical files."""
        path1 = str(tmp_path / "grad1.jpg")
        path2 = str(tmp_path / "grad2.jpg")

        generate_gradient_jpeg(path1, quality=85)
        generate_gradient_jpeg(path2, quality=85)

        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            assert f1.read() == f2.read()

    def test_all_generators_registered(self):
        """All generator functions should be in registry."""
        assert "solid_blue_jpeg" in IMAGE_GENERATORS
        assert "gradient_jpeg" in IMAGE_GENERATORS
        assert "checkerboard_jpeg" in IMAGE_GENERATORS
        assert "solid_red_png" in IMAGE_GENERATORS
        assert "geometry_test_jpeg" in IMAGE_GENERATORS


class TestReferenceVectors:
    """Tests for reference vector definitions."""

    def test_vectors_defined(self):
        """Reference vectors should be defined."""
        assert len(REFERENCE_VECTORS) > 0

    def test_vectors_have_required_fields(self):
        """Each vector should have all required fields."""
        for vector in REFERENCE_VECTORS:
            assert vector.name
            assert vector.description
            assert vector.image_generator in IMAGE_GENERATORS
            assert len(vector.dimensions) > 0
            assert len(vector.expected_states) > 0

    def test_vector_dimensions_have_expected_states(self):
        """Each dimension in a vector should have an expected state."""
        for vector in REFERENCE_VECTORS:
            for dim in vector.dimensions:
                assert dim in vector.expected_states, \
                    f"Vector {vector.name}: dimension {dim} missing expected state"


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_passed_result(self):
        result = VerificationResult(
            vector_name="test",
            passed=True,
            details={"hash": "abc123"},
            errors=[]
        )
        assert result.passed is True
        assert result.vector_name == "test"

    def test_failed_result(self):
        result = VerificationResult(
            vector_name="test",
            passed=False,
            details={},
            errors=["Something went wrong"]
        )
        assert result.passed is False
        assert len(result.errors) == 1


class TestReferenceVerifier:
    """Tests for ReferenceVerifier class."""

    def test_verifier_creation(self):
        verifier = ReferenceVerifier()
        assert verifier is not None

    def test_verifier_creation_verbose(self):
        verifier = ReferenceVerifier(verbose=True)
        assert verifier.verbose is True

    def test_verify_all_runs(self):
        """verify_all should complete without errors."""
        verifier = ReferenceVerifier()
        passed, report = verifier.verify_all()

        assert isinstance(passed, bool)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_verify_all_report_structure(self):
        """Report should have proper structure."""
        verifier = ReferenceVerifier()
        passed, report = verifier.verify_all()

        assert "VERIFICATION" in report
        assert "STATUS" in report

    def test_verify_all_verbose_report(self):
        """Verbose report should include details."""
        verifier = ReferenceVerifier(verbose=True)
        passed, report = verifier.verify_all()

        assert "DETAILED RESULTS" in report

    def test_generate_checksums(self):
        """generate_checksums should return dict of hashes."""
        verifier = ReferenceVerifier()
        checksums = verifier.generate_checksums()

        assert isinstance(checksums, dict)
        assert len(checksums) > 0

        for name, checksum in checksums.items():
            assert len(checksum) == 64  # SHA256 hex


class TestVerifyInstallation:
    """Tests for verify_installation convenience function."""

    def test_verify_installation_runs(self):
        """Function should complete without errors."""
        passed, report = verify_installation()

        assert isinstance(passed, bool)
        assert isinstance(report, str)

    def test_verify_installation_verbose(self):
        """Verbose mode should include more detail."""
        passed, report = verify_installation(verbose=True)

        assert "DETAILED RESULTS" in report


class TestIntegration:
    """Integration tests for verification system."""

    def test_unmodified_installation_passes(self):
        """
        Unmodified Wu installation should pass verification.

        This is the key test: if this fails, either the reference vectors
        need updating or something is wrong with the installation.
        """
        passed, report = verify_installation()

        # This should pass for a correct installation
        assert passed, f"Verification failed:\n{report}"

    def test_verification_detects_all_vectors(self):
        """All reference vectors should be verified."""
        verifier = ReferenceVerifier(verbose=True)
        passed, report = verifier.verify_all()

        # Check that each vector is mentioned in report
        for vector in REFERENCE_VECTORS:
            assert vector.name in report


class TestVerificationReproducibility:
    """Tests for reproducibility of verification."""

    def test_multiple_runs_same_result(self):
        """Multiple verification runs should produce same result."""
        results = []
        for _ in range(3):
            passed, _ = verify_installation()
            results.append(passed)

        # All runs should have same outcome
        assert all(r == results[0] for r in results)

    def test_hash_reproducibility(self):
        """Generated image hashes should be reproducible."""
        verifier = ReferenceVerifier()

        checksums1 = verifier.generate_checksums()
        checksums2 = verifier.generate_checksums()

        assert checksums1 == checksums2
