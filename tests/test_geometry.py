"""
Tests for projective geometry forensic analysis.

Tests shadow direction and vanishing point consistency detection.
"""

import pytest
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from wu.dimensions.geometry import (
    Line2D,
    VanishingPoint,
    ShadowDirection,
    GeometryResult,
    line_intersection,
    angle_difference,
    cluster_lines_by_angle,
    estimate_vanishing_point,
    ShadowAnalyzer,
    PerspectiveAnalyzer,
    GeometricAnalyzer,
)
from wu.state import DimensionState, Confidence


pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PIL,
    reason="numpy and PIL required for geometry tests"
)


class TestLine2D:
    """Tests for Line2D dataclass."""

    def test_creation(self):
        line = Line2D(0, 0, 100, 0)
        assert line.x1 == 0
        assert line.y1 == 0
        assert line.x2 == 100
        assert line.y2 == 0

    def test_horizontal_angle(self):
        line = Line2D(0, 0, 100, 0)
        assert abs(line.angle) < 0.01
        assert abs(line.angle_degrees) < 1

    def test_vertical_angle(self):
        line = Line2D(0, 0, 0, 100)
        assert abs(line.angle - math.pi/2) < 0.01
        assert abs(line.angle_degrees - 90) < 1

    def test_diagonal_angle(self):
        line = Line2D(0, 0, 100, 100)
        assert abs(line.angle - math.pi/4) < 0.01
        assert abs(line.angle_degrees - 45) < 1

    def test_length(self):
        line = Line2D(0, 0, 3, 4)
        assert abs(line.length - 5) < 0.01

    def test_midpoint(self):
        line = Line2D(0, 0, 100, 100)
        mx, my = line.midpoint
        assert abs(mx - 50) < 0.01
        assert abs(my - 50) < 0.01

    def test_to_homogeneous(self):
        line = Line2D(0, 0, 100, 0)
        homo = line.to_homogeneous()
        assert homo.shape == (3,)
        # Line y=0 should have form (0, 1, 0) after normalization
        # ax + by + c = 0 -> y = 0 when a=0, b=1, c=0


class TestVanishingPoint:
    """Tests for VanishingPoint dataclass."""

    def test_creation(self):
        vp = VanishingPoint(
            x=100, y=200,
            confidence=0.9,
            supporting_lines=5,
            direction="horizontal"
        )
        assert vp.x == 100
        assert vp.y == 200
        assert vp.confidence == 0.9
        assert vp.supporting_lines == 5

    def test_is_at_infinity(self):
        vp_normal = VanishingPoint(100, 200, 0.9, 5, "horizontal")
        assert not vp_normal.is_at_infinity

        vp_inf = VanishingPoint(50000, 200, 0.9, 5, "horizontal")
        assert vp_inf.is_at_infinity


class TestShadowDirection:
    """Tests for ShadowDirection dataclass."""

    def test_creation(self):
        sd = ShadowDirection(
            angle=45.0,
            confidence=0.8,
            region=(100, 200, 50, 60),
            supporting_edges=10
        )
        assert sd.angle == 45.0
        assert sd.confidence == 0.8
        assert sd.region == (100, 200, 50, 60)


class TestGeometryResult:
    """Tests for GeometryResult dataclass."""

    def test_empty_creation(self):
        result = GeometryResult()
        assert result.vanishing_points == []
        assert result.shadow_directions == []
        assert result.vp_consistent is True
        assert result.shadows_consistent is True

    def test_with_data(self):
        vp = VanishingPoint(100, 200, 0.9, 5, "horizontal")
        sd = ShadowDirection(45.0, 0.8, (0, 0, 10, 10), 5)

        result = GeometryResult(
            vanishing_points=[vp],
            shadow_directions=[sd],
            vp_consistent=False,
            vp_inconsistency_reason="Test reason"
        )
        assert len(result.vanishing_points) == 1
        assert len(result.shadow_directions) == 1
        assert result.vp_consistent is False


class TestLineIntersection:
    """Tests for line_intersection function."""

    def test_perpendicular_lines(self):
        # Horizontal line y=50: 0x + 1y - 50 = 0
        line1 = np.array([0, 1, -50])
        # Vertical line x=100: 1x + 0y - 100 = 0
        line2 = np.array([1, 0, -100])

        result = line_intersection(line1, line2)
        assert result is not None
        x, y = result
        assert abs(x - 100) < 0.01
        assert abs(y - 50) < 0.01

    def test_parallel_lines(self):
        # Two horizontal lines
        line1 = np.array([0, 1, -50])
        line2 = np.array([0, 1, -100])

        result = line_intersection(line1, line2)
        assert result is None

    def test_diagonal_lines(self):
        # y = x (goes through origin with slope 1)
        line1 = np.array([1, -1, 0])
        # y = -x + 100 (slope -1, intercept 100)
        line2 = np.array([1, 1, -100])

        result = line_intersection(line1, line2)
        assert result is not None
        x, y = result
        assert abs(x - 50) < 0.01
        assert abs(y - 50) < 0.01


class TestAngleDifference:
    """Tests for angle_difference function."""

    def test_same_angle(self):
        assert angle_difference(45, 45) < 0.01

    def test_small_difference(self):
        assert abs(angle_difference(45, 50) - 5) < 0.01

    def test_opposite_angles(self):
        # 0 and 180 should be considered same direction
        assert angle_difference(0, 180) < 0.01

    def test_near_opposite(self):
        assert abs(angle_difference(10, 170) - 20) < 0.01

    def test_wraparound(self):
        # -45 and 135 are perpendicular (90 degrees apart)
        assert abs(angle_difference(-45, 45) - 90) < 0.01


class TestClusterLinesByAngle:
    """Tests for cluster_lines_by_angle function."""

    def test_empty_input(self):
        assert cluster_lines_by_angle([]) == []

    def test_single_line(self):
        lines = [Line2D(0, 0, 100, 0)]
        clusters = cluster_lines_by_angle(lines)
        assert clusters == []  # Need at least 2 lines per cluster

    def test_similar_angles(self):
        # Three horizontal lines
        lines = [
            Line2D(0, 0, 100, 0),
            Line2D(0, 10, 100, 10),
            Line2D(0, 20, 100, 22),  # Slight angle
        ]
        clusters = cluster_lines_by_angle(lines, angle_threshold=10.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_angles(self):
        # Horizontal and vertical lines
        lines = [
            Line2D(0, 0, 100, 0),    # Horizontal
            Line2D(0, 10, 100, 10),  # Horizontal
            Line2D(50, 0, 50, 100),  # Vertical
            Line2D(60, 0, 60, 100),  # Vertical
        ]
        clusters = cluster_lines_by_angle(lines, angle_threshold=10.0)
        assert len(clusters) == 2


class TestEstimateVanishingPoint:
    """Tests for estimate_vanishing_point function."""

    def test_insufficient_lines(self):
        lines = [Line2D(0, 0, 100, 0)]
        vp = estimate_vanishing_point(lines)
        assert vp is None

    def test_converging_lines(self):
        # Lines that should converge to a point
        # All lines through (100, 100)
        lines = [
            Line2D(0, 0, 100, 100),
            Line2D(0, 200, 100, 100),
            Line2D(200, 200, 100, 100),
        ]
        vp = estimate_vanishing_point(lines)
        assert vp is not None
        assert abs(vp.x - 100) < 5
        assert abs(vp.y - 100) < 5


class TestShadowAnalyzerBasic:
    """Basic tests for ShadowAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = ShadowAnalyzer()
        assert analyzer is not None
        assert analyzer.SHADOW_THRESHOLD == 0.3

    def test_analyze_missing_file(self):
        analyzer = ShadowAnalyzer()
        result = analyzer.analyze("/nonexistent/photo.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "shadows"

    def test_analyze_invalid_file(self, tmp_path):
        invalid_file = tmp_path / "not_image.txt"
        invalid_file.write_text("This is not an image")

        analyzer = ShadowAnalyzer()
        result = analyzer.analyze(str(invalid_file))
        assert result.state == DimensionState.UNCERTAIN


class TestPerspectiveAnalyzerBasic:
    """Basic tests for PerspectiveAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = PerspectiveAnalyzer()
        assert analyzer is not None
        assert analyzer.MIN_LINES_FOR_VP == 4

    def test_analyze_missing_file(self):
        analyzer = PerspectiveAnalyzer()
        result = analyzer.analyze("/nonexistent/photo.jpg")
        assert result.state == DimensionState.UNCERTAIN
        assert result.dimension == "perspective"

    def test_analyze_invalid_file(self, tmp_path):
        invalid_file = tmp_path / "not_image.txt"
        invalid_file.write_text("This is not an image")

        analyzer = PerspectiveAnalyzer()
        result = analyzer.analyze(str(invalid_file))
        assert result.state == DimensionState.UNCERTAIN


@pytest.fixture
def simple_image(tmp_path):
    """Create a simple test image."""
    img = Image.new('RGB', (400, 300), color='white')
    file_path = tmp_path / "simple.jpg"
    img.save(str(file_path), "JPEG")
    return str(file_path)


@pytest.fixture
def image_with_lines(tmp_path):
    """Create an image with distinct line segments."""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some horizontal lines
    for y in range(50, 250, 50):
        draw.line([(50, y), (350, y)], fill='black', width=2)

    # Draw some vertical lines
    for x in range(50, 350, 50):
        draw.line([(x, 50), (x, 250)], fill='black', width=2)

    file_path = tmp_path / "lines.jpg"
    img.save(str(file_path), "JPEG")
    return str(file_path)


@pytest.fixture
def image_with_shadows(tmp_path):
    """Create an image with shadow-like regions."""
    img = Image.new('RGB', (400, 300), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Draw a "shadow" (dark region)
    draw.polygon([(100, 200), (150, 200), (200, 280), (50, 280)], fill=(30, 30, 30))

    # Draw an "object" that casts shadow
    draw.rectangle([(100, 100), (150, 200)], fill=(100, 100, 200))

    file_path = tmp_path / "shadow.jpg"
    img.save(str(file_path), "JPEG")
    return str(file_path)


class TestShadowAnalysis:
    """Tests for shadow analysis on images."""

    def test_simple_image(self, simple_image):
        analyzer = ShadowAnalyzer()
        result = analyzer.analyze(simple_image)

        assert result.dimension == "shadows"
        assert result.state is not None
        assert result.confidence is not None

    def test_image_with_shadows(self, image_with_shadows):
        analyzer = ShadowAnalyzer()
        result = analyzer.analyze(image_with_shadows)

        assert result.dimension == "shadows"
        # May or may not detect shadows depending on edge detection
        assert len(result.evidence) > 0


class TestPerspectiveAnalysis:
    """Tests for perspective analysis on images."""

    def test_simple_image(self, simple_image):
        analyzer = PerspectiveAnalyzer()
        result = analyzer.analyze(simple_image)

        assert result.dimension == "perspective"
        assert result.state is not None

    def test_image_with_lines(self, image_with_lines):
        analyzer = PerspectiveAnalyzer()
        result = analyzer.analyze(image_with_lines)

        assert result.dimension == "perspective"
        assert len(result.evidence) > 0


class TestGeometricAnalyzer:
    """Tests for combined GeometricAnalyzer."""

    def test_combined_analyzer_creation(self):
        analyzer = GeometricAnalyzer()
        assert analyzer.shadow_analyzer is not None
        assert analyzer.perspective_analyzer is not None

    def test_analyze_shadows(self, simple_image):
        analyzer = GeometricAnalyzer()
        result = analyzer.analyze_shadows(simple_image)
        assert result.dimension == "shadows"

    def test_analyze_perspective(self, simple_image):
        analyzer = GeometricAnalyzer()
        result = analyzer.analyze_perspective(simple_image)
        assert result.dimension == "perspective"


class TestIntegration:
    """Integration tests with main WuAnalyzer."""

    def test_wu_analyzer_with_shadows(self, simple_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_shadows=True
        )
        result = analyzer.analyze(simple_image)

        assert result.shadows is not None
        assert result.shadows.dimension == "shadows"

    def test_wu_analyzer_with_perspective(self, simple_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_perspective=True
        )
        result = analyzer.analyze(simple_image)

        assert result.perspective is not None
        assert result.perspective.dimension == "perspective"

    def test_wu_analyzer_both_disabled_by_default(self, simple_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer()
        result = analyzer.analyze(simple_image)

        # Both should be disabled by default
        assert result.shadows is None
        assert result.perspective is None

    def test_wu_analyzer_full_geometry(self, simple_image):
        from wu.analyzer import WuAnalyzer

        analyzer = WuAnalyzer(
            enable_metadata=False,
            enable_c2pa=False,
            enable_visual=False,
            enable_shadows=True,
            enable_perspective=True
        )
        result = analyzer.analyze(simple_image)

        assert result.shadows is not None
        assert result.perspective is not None
        assert len(result.dimensions) == 2


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_image(self, tmp_path):
        """Very small images should be handled."""
        img = Image.new('RGB', (32, 32), color='gray')
        file_path = tmp_path / "tiny.jpg"
        img.save(str(file_path), "JPEG")

        shadow_analyzer = ShadowAnalyzer()
        result = shadow_analyzer.analyze(str(file_path))
        assert result.dimension == "shadows"

        perspective_analyzer = PerspectiveAnalyzer()
        result = perspective_analyzer.analyze(str(file_path))
        assert result.dimension == "perspective"

    def test_grayscale_image(self, tmp_path):
        """Grayscale images should be handled."""
        img = Image.new('L', (200, 200), color=128)
        file_path = tmp_path / "gray.jpg"
        img.save(str(file_path), "JPEG")

        shadow_analyzer = ShadowAnalyzer()
        result = shadow_analyzer.analyze(str(file_path))
        assert result.dimension == "shadows"

    def test_png_image(self, tmp_path):
        """PNG images should work."""
        img = Image.new('RGB', (200, 200), color='blue')
        file_path = tmp_path / "test.png"
        img.save(str(file_path), "PNG")

        analyzer = ShadowAnalyzer()
        result = analyzer.analyze(str(file_path))
        assert result.dimension == "shadows"


class TestNoDependencies:
    """Test graceful handling when dependencies are missing."""

    def test_handles_missing_deps_gracefully(self, monkeypatch, tmp_path):
        """Should return UNCERTAIN when deps missing."""
        img = Image.new('RGB', (200, 200), color='blue')
        file_path = tmp_path / "test.jpg"
        img.save(str(file_path), "JPEG")

        import wu.dimensions.geometry as geom_module
        monkeypatch.setattr(geom_module, "HAS_PIL", False)

        shadow_analyzer = ShadowAnalyzer()
        result = shadow_analyzer.analyze(str(file_path))
        assert result.state == DimensionState.UNCERTAIN

        perspective_analyzer = PerspectiveAnalyzer()
        result = perspective_analyzer.analyze(str(file_path))
        assert result.state == DimensionState.UNCERTAIN
