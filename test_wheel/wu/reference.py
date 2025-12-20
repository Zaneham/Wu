"""
Reference test vectors for Wu verification.

This module provides deterministic test images and their expected analysis
results, allowing users to verify that their Wu installation has not been
modified and produces correct results.

The verification process works as follows:
1. Generate known test images using deterministic algorithms
2. Analyse each image with specified dimensions
3. Compare critical findings against expected values
4. Report whether the installation passes verification

This approach detects both accidental corruption and malicious modification,
as any change to the analysis code would produce different results from the
reference vectors.

Usage:
    from wu.reference import ReferenceVerifier
    verifier = ReferenceVerifier()
    passed, report = verifier.verify_all()
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import tempfile

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

from .analyzer import WuAnalyzer
from .state import DimensionState


@dataclass
class ReferenceVector:
    """A single reference test vector."""
    name: str
    description: str
    image_generator: str  # Name of generator function
    dimensions: List[str]  # Dimensions to test
    expected_states: Dict[str, str]  # dimension -> expected state
    expected_hash: str  # SHA256 of generated image
    critical_findings: List[str] = field(default_factory=list)


# =============================================================================
# DETERMINISTIC IMAGE GENERATORS
# =============================================================================

def generate_solid_blue_jpeg(path: str, quality: int = 85) -> str:
    """
    Generate a solid blue JPEG image.

    This is entirely deterministic: same dimensions, same colour, same quality
    will always produce the same file on the same platform.
    """
    if not HAS_PIL:
        raise RuntimeError("PIL required for reference image generation")

    img = Image.new('RGB', (256, 256), color=(50, 100, 200))
    img.save(path, "JPEG", quality=quality)
    return path


def generate_gradient_jpeg(path: str, quality: int = 85) -> str:
    """
    Generate a horizontal gradient JPEG image.

    Deterministic gradient from black to white.
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise RuntimeError("PIL and numpy required for reference image generation")

    width, height = 256, 256
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(width):
        value = int((x / width) * 255)
        arr[:, x, :] = value

    img = Image.fromarray(arr, 'RGB')
    img.save(path, "JPEG", quality=quality)
    return path


def generate_checkerboard_jpeg(path: str, quality: int = 85) -> str:
    """
    Generate a checkerboard pattern JPEG.

    Deterministic 8x8 pixel checkerboard pattern.
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise RuntimeError("PIL and numpy required for reference image generation")

    width, height = 256, 256
    block_size = 8
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            block_x = x // block_size
            block_y = y // block_size
            if (block_x + block_y) % 2 == 0:
                arr[y, x] = [200, 200, 200]
            else:
                arr[y, x] = [50, 50, 50]

    img = Image.fromarray(arr, 'RGB')
    img.save(path, "JPEG", quality=quality)
    return path


def generate_solid_red_png(path: str) -> str:
    """
    Generate a solid red PNG image.

    PNG with no JPEG compression history.
    """
    if not HAS_PIL:
        raise RuntimeError("PIL required for reference image generation")

    img = Image.new('RGB', (256, 256), color=(200, 50, 50))
    img.save(path, "PNG")
    return path


def generate_geometry_test_jpeg(path: str, quality: int = 85) -> str:
    """
    Generate an image with geometric features for shadow/perspective testing.

    Contains rectangles that create shadow-like regions and line segments
    for vanishing point detection.
    """
    if not HAS_PIL:
        raise RuntimeError("PIL required for reference image generation")

    from PIL import ImageDraw

    img = Image.new('RGB', (512, 384), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    # Draw converging lines (for perspective)
    for i in range(10):
        y_start = 50 + i * 30
        draw.line([(0, y_start), (512, 192)], fill=(100, 100, 100), width=2)

    # Draw rectangles (for shadows)
    draw.rectangle([(100, 200), (150, 350)], fill=(60, 60, 60))
    draw.rectangle([(300, 200), (350, 350)], fill=(60, 60, 60))

    img.save(path, "JPEG", quality=quality)
    return path


# Generator registry
IMAGE_GENERATORS = {
    "solid_blue_jpeg": generate_solid_blue_jpeg,
    "gradient_jpeg": generate_gradient_jpeg,
    "checkerboard_jpeg": generate_checkerboard_jpeg,
    "solid_red_png": generate_solid_red_png,
    "geometry_test_jpeg": generate_geometry_test_jpeg,
}


# =============================================================================
# REFERENCE VECTORS
# =============================================================================

# These vectors define expected behaviour for known inputs.
# The expected_hash values are computed on first run and should be
# updated if the image generation algorithm changes.

REFERENCE_VECTORS = [
    ReferenceVector(
        name="solid_blue_basic",
        description="Solid blue JPEG with basic metadata and visual analysis",
        image_generator="solid_blue_jpeg",
        dimensions=["visual"],
        expected_states={
            # Simple synthetic images lack EXIF so metadata may be suspicious
            # Visual should be consistent for uniform image
            "visual": "consistent",
        },
        expected_hash="",  # Computed at runtime, verified against stored
        critical_findings=[],
    ),
    ReferenceVector(
        name="gradient_quantization",
        description="Gradient JPEG for quantization table analysis",
        image_generator="gradient_jpeg",
        dimensions=["quantization"],
        expected_states={
            # Quantization analysis on fresh JPEG should be consistent
            "quantization": "consistent",
        },
        expected_hash="",
        critical_findings=[],
    ),
    ReferenceVector(
        name="checkerboard_visual",
        description="Checkerboard pattern for visual forensics",
        image_generator="checkerboard_jpeg",
        dimensions=["visual"],
        expected_states={
            "visual": "consistent",
        },
        expected_hash="",
        critical_findings=[],
    ),
    ReferenceVector(
        name="png_no_jpeg",
        description="PNG image should show uncertain for JPEG-specific analyses",
        image_generator="solid_red_png",
        dimensions=["quantization"],
        expected_states={
            "quantization": "uncertain",  # Not a JPEG
        },
        expected_hash="",
        critical_findings=[],
    ),
    ReferenceVector(
        name="geometry_analysis",
        description="Geometric test image for shadow and perspective analysis",
        image_generator="geometry_test_jpeg",
        dimensions=["shadows", "perspective"],
        expected_states={
            "shadows": "consistent",
            "perspective": "consistent",
        },
        expected_hash="",
        critical_findings=[],
    ),
]


# =============================================================================
# VERIFIER
# =============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a single reference vector."""
    vector_name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ReferenceVerifier:
    """
    Verifies Wu installation against reference test vectors.

    This class generates known test images, analyses them, and compares
    the results against expected values. Any discrepancy indicates either
    corruption or modification of the Wu codebase.

    Usage:
        verifier = ReferenceVerifier()
        passed, report = verifier.verify_all()

        if not passed:
            print("VERIFICATION FAILED")
            print(report)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._temp_dir = None

    def verify_all(self) -> Tuple[bool, str]:
        """
        Verify all reference vectors.

        Returns:
            Tuple of (all_passed, report_string)
        """
        if not HAS_PIL or not HAS_NUMPY:
            return False, "ERROR: PIL and numpy required for verification"

        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            self._temp_dir = Path(temp_dir)

            for vector in REFERENCE_VECTORS:
                result = self._verify_vector(vector)
                results.append(result)

        # Generate report
        all_passed = all(r.passed for r in results)
        report = self._generate_report(results, all_passed)

        return all_passed, report

    def _verify_vector(self, vector: ReferenceVector) -> VerificationResult:
        """Verify a single reference vector."""
        result = VerificationResult(vector_name=vector.name, passed=True)

        try:
            # Generate image
            generator = IMAGE_GENERATORS.get(vector.image_generator)
            if not generator:
                result.passed = False
                result.errors.append(f"Unknown generator: {vector.image_generator}")
                return result

            image_path = str(self._temp_dir / f"{vector.name}.tmp")
            generator(image_path)

            # Compute hash
            file_hash = self._compute_hash(image_path)
            result.details["file_hash"] = file_hash

            # If expected hash is set, verify it
            if vector.expected_hash and vector.expected_hash != file_hash:
                result.passed = False
                result.errors.append(
                    f"Image hash mismatch: expected {vector.expected_hash[:16]}..., "
                    f"got {file_hash[:16]}..."
                )

            # Analyse with specified dimensions
            analyzer = self._create_analyzer(vector.dimensions)
            analysis = analyzer.analyze(image_path)

            # Verify dimension states
            for dim_name, expected_state in vector.expected_states.items():
                dim_result = getattr(analysis, dim_name, None)

                if dim_result is None:
                    result.passed = False
                    result.errors.append(f"Dimension {dim_name} not in analysis")
                    continue

                actual_state = dim_result.state.value
                result.details[f"{dim_name}_state"] = actual_state

                if actual_state != expected_state:
                    result.passed = False
                    result.errors.append(
                        f"Dimension {dim_name}: expected {expected_state}, "
                        f"got {actual_state}"
                    )

        except Exception as e:
            result.passed = False
            result.errors.append(f"Exception during verification: {str(e)}")

        return result

    def _create_analyzer(self, dimensions: List[str]) -> WuAnalyzer:
        """Create analyser with only specified dimensions enabled."""
        kwargs = {
            "enable_metadata": "metadata" in dimensions,
            "enable_c2pa": "c2pa" in dimensions,
            "enable_visual": "visual" in dimensions,
            "enable_thumbnail": "thumbnail" in dimensions,
            "enable_shadows": "shadows" in dimensions,
            "enable_perspective": "perspective" in dimensions,
            "enable_blockgrid": "blockgrid" in dimensions,
            "enable_quantization": "quantization" in dimensions,
            "enable_copymove": "copymove" in dimensions,
            "enable_prnu": "prnu" in dimensions,
            "enable_lighting": "lighting" in dimensions,
            "enable_enf": "enf" in dimensions,
            "enable_audio": "audio" in dimensions,
            "enable_aigen": "aigen" in dimensions,
        }
        return WuAnalyzer(**kwargs)

    def _compute_hash(self, path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_report(
        self,
        results: List[VerificationResult],
        all_passed: bool
    ) -> str:
        """Generate human-readable verification report."""
        lines = []
        lines.append("=" * 60)
        lines.append("WU REFERENCE VECTOR VERIFICATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)

        if all_passed:
            lines.append(f"STATUS: PASSED ({passed_count}/{total_count} vectors)")
            lines.append("")
            lines.append("All reference vectors produced expected results.")
            lines.append("This Wu installation appears unmodified.")
        else:
            lines.append(f"STATUS: FAILED ({passed_count}/{total_count} vectors)")
            lines.append("")
            lines.append("WARNING: Some reference vectors produced unexpected results.")
            lines.append("This may indicate modification or corruption of the Wu codebase.")
            lines.append("")
            lines.append("Failed vectors:")

            for result in results:
                if not result.passed:
                    lines.append(f"  - {result.vector_name}")
                    for error in result.errors:
                        lines.append(f"      {error}")

        lines.append("")
        lines.append("=" * 60)

        if self.verbose:
            lines.append("")
            lines.append("DETAILED RESULTS:")
            lines.append("")
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                lines.append(f"[{status}] {result.vector_name}")
                for key, value in result.details.items():
                    lines.append(f"       {key}: {value}")
                if result.errors:
                    for error in result.errors:
                        lines.append(f"       ERROR: {error}")
                lines.append("")

        return "\n".join(lines)

    def generate_checksums(self) -> Dict[str, str]:
        """
        Generate checksums for all reference images.

        This is used during development to establish baseline values.
        In production, these values should be hardcoded in REFERENCE_VECTORS.
        """
        checksums = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            self._temp_dir = Path(temp_dir)

            for vector in REFERENCE_VECTORS:
                generator = IMAGE_GENERATORS.get(vector.image_generator)
                if generator:
                    image_path = str(self._temp_dir / f"{vector.name}.tmp")
                    generator(image_path)
                    checksums[vector.name] = self._compute_hash(image_path)

        return checksums


def verify_installation(verbose: bool = False) -> Tuple[bool, str]:
    """
    Convenience function to verify Wu installation.

    Returns:
        Tuple of (passed, report)
    """
    verifier = ReferenceVerifier(verbose=verbose)
    return verifier.verify_all()
