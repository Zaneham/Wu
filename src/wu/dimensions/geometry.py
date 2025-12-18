"""
Projective geometry forensic analysis.

Unified module for shadow and perspective analysis - they share the same
mathematical foundation: projective geometry and line intersection.

Shadow Analysis:
    Shadows are projections from a light source through objects onto surfaces.
    In a real scene, all shadows must be geometrically consistent with a
    single light source position.

Perspective Analysis:
    Parallel lines in 3D converge to vanishing points in 2D images.
    In a real scene, sets of parallel lines (building edges, road markings)
    must converge consistently.

Court relevance:
    "The shadow directions indicate light sources at incompatible positions.
    The vanishing points for parallel structures don't converge, proving
    the image is a composite of multiple photographs."

PERFORMANCE NOTES:
    # OPTIMIZE: C - Line detection and Hough transform
    # OPTIMIZE: ASM - SIMD for gradient computation

References:
    Kee, E., O'Brien, J.F., & Farid, H. (2013). Exposing Photo Manipulation
        with Inconsistent Shadows. ACM Transactions on Graphics.
    Johnson, M.K. & Farid, H. (2007). Exposing Digital Forgeries Through
        Specular Highlights on the Eye. Information Hiding.
    Criminisi, A. & Zisserman, A. (2000). Shape from Texture: Homogeneity
        Revisited. BMVC.
"""

import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
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

try:
    from scipy import ndimage
    from scipy.signal import convolve2d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Line2D:
    """A 2D line segment."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def angle(self) -> float:
        """Angle in radians (-pi/2 to pi/2)."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.atan2(dy, dx)

    @property
    def angle_degrees(self) -> float:
        """Angle in degrees (-90 to 90)."""
        return math.degrees(self.angle)

    @property
    def length(self) -> float:
        """Length of line segment."""
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint of line segment."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_homogeneous(self) -> np.ndarray:
        """Convert to homogeneous line representation (a, b, c) where ax + by + c = 0."""
        # Line through two points
        a = self.y1 - self.y2
        b = self.x2 - self.x1
        c = self.x1 * self.y2 - self.x2 * self.y1
        # Normalize
        norm = math.sqrt(a*a + b*b)
        if norm > 0:
            return np.array([a/norm, b/norm, c/norm])
        return np.array([a, b, c])


@dataclass
class VanishingPoint:
    """A detected vanishing point."""
    x: float
    y: float
    confidence: float  # 0-1
    supporting_lines: int  # Number of lines converging here
    direction: str  # "horizontal", "vertical", or angle description

    @property
    def is_at_infinity(self) -> bool:
        """True if vanishing point is very far (parallel lines in image)."""
        return abs(self.x) > 10000 or abs(self.y) > 10000


@dataclass
class ShadowDirection:
    """A detected shadow direction."""
    angle: float  # Direction angle in degrees
    confidence: float  # 0-1
    region: Tuple[int, int, int, int]  # (x, y, w, h) where detected
    supporting_edges: int  # Number of shadow edges


@dataclass
class GeometryResult:
    """Result of geometric consistency analysis."""
    # Vanishing points
    vanishing_points: List[VanishingPoint] = field(default_factory=list)
    vp_consistent: bool = True
    vp_inconsistency_reason: Optional[str] = None

    # Shadow analysis
    shadow_directions: List[ShadowDirection] = field(default_factory=list)
    shadows_consistent: bool = True
    shadow_inconsistency_reason: Optional[str] = None

    # Overall
    processing_time_ms: float = 0.0


# =============================================================================
# CORE GEOMETRY FUNCTIONS
# =============================================================================

def line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two lines in homogeneous coordinates.

    Lines are represented as (a, b, c) where ax + by + c = 0.
    Returns (x, y) or None if parallel.
    """
    # Cross product gives intersection point in homogeneous coordinates
    p = np.cross(line1, line2)

    if abs(p[2]) < 1e-10:
        return None  # Parallel lines

    return (p[0] / p[2], p[1] / p[2])


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute smallest angle difference in degrees.

    Handles wraparound at 180 degrees.
    """
    diff = abs(angle1 - angle2) % 180
    return min(diff, 180 - diff)


def cluster_lines_by_angle(
    lines: List[Line2D],
    angle_threshold: float = 10.0
) -> List[List[Line2D]]:
    """
    Cluster lines by their angles.

    Returns list of clusters, each containing lines with similar angles.
    """
    if not lines:
        return []

    # Sort by angle
    sorted_lines = sorted(lines, key=lambda l: l.angle_degrees)

    clusters = []
    current_cluster = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        if angle_difference(line.angle_degrees, current_cluster[-1].angle_degrees) < angle_threshold:
            current_cluster.append(line)
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [line]

    if len(current_cluster) >= 2:
        clusters.append(current_cluster)

    return clusters


def estimate_vanishing_point(lines: List[Line2D]) -> Optional[VanishingPoint]:
    """
    Estimate vanishing point from a set of lines.

    Uses least-squares intersection of all line pairs.
    """
    if len(lines) < 2:
        return None

    # Convert all lines to homogeneous form
    homo_lines = [line.to_homogeneous() for line in lines]

    # Find all pairwise intersections
    intersections = []
    for i in range(len(homo_lines)):
        for j in range(i + 1, len(homo_lines)):
            pt = line_intersection(homo_lines[i], homo_lines[j])
            if pt is not None:
                intersections.append(pt)

    if not intersections:
        return None

    # Robust estimation: use median
    xs = [p[0] for p in intersections]
    ys = [p[1] for p in intersections]

    # Filter outliers (intersections too far from median)
    med_x, med_y = np.median(xs), np.median(ys)
    distances = [math.sqrt((x - med_x)**2 + (y - med_y)**2) for x, y in intersections]
    threshold = np.percentile(distances, 75) * 2

    filtered = [(x, y) for (x, y), d in zip(intersections, distances) if d < threshold]

    if not filtered:
        filtered = intersections

    # Final estimate: mean of filtered intersections
    vp_x = np.mean([p[0] for p in filtered])
    vp_y = np.mean([p[1] for p in filtered])

    # Confidence based on consistency
    if len(filtered) > 1:
        spread = np.std([p[0] for p in filtered]) + np.std([p[1] for p in filtered])
        confidence = 1.0 / (1.0 + spread / 100)
    else:
        confidence = 0.5

    # Determine direction
    mean_angle = np.mean([line.angle_degrees for line in lines])
    if abs(mean_angle) < 20:
        direction = "horizontal"
    elif abs(abs(mean_angle) - 90) < 20:
        direction = "vertical"
    else:
        direction = f"{mean_angle:.0f}°"

    return VanishingPoint(
        x=vp_x,
        y=vp_y,
        confidence=confidence,
        supporting_lines=len(lines),
        direction=direction
    )


# =============================================================================
# SHADOW ANALYZER
# =============================================================================

class ShadowAnalyzer:
    """
    Analyzes shadow consistency in images.

    In a real scene with a single light source, all shadows must:
    1. Point in consistent directions
    2. Have lengths proportional to object heights
    3. Be geometrically consistent with the light source position

    Detection method:
    1. Find dark regions (potential shadows)
    2. Detect edges at shadow boundaries
    3. Estimate shadow directions from edge orientations
    4. Check consistency across the image
    """

    SHADOW_THRESHOLD = 0.3  # Brightness threshold for shadow detection
    MIN_SHADOW_AREA = 100  # Minimum pixels for shadow region
    DIRECTION_TOLERANCE = 20.0  # Degrees tolerance for consistent shadows

    def analyze(self, file_path: str) -> DimensionResult:
        """Analyze shadow consistency in image."""
        if not HAS_NUMPY or not HAS_PIL:
            return self._uncertain_result("Dependencies not available")

        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = np.array(img)
        except Exception as e:
            return self._uncertain_result(f"Cannot load image: {e}")

        result = self._analyze_shadows(image)
        return self._build_result(result)

    def _analyze_shadows(self, image: np.ndarray) -> GeometryResult:
        """Perform shadow analysis."""
        import time
        start = time.time()

        height, width = image.shape[:2]

        # Convert to grayscale
        gray = np.mean(image, axis=2).astype(np.float64) / 255.0

        # Find shadow regions (dark areas)
        shadow_mask = gray < self.SHADOW_THRESHOLD

        # Detect edges in shadow regions
        edges = self._detect_edges(gray)

        # Find lines at shadow boundaries
        shadow_edges = edges * shadow_mask.astype(np.float64)

        # Detect line segments
        lines = self._detect_lines(shadow_edges, width, height)

        # Estimate shadow directions
        shadow_directions = self._estimate_shadow_directions(lines, width, height)

        # Check consistency
        consistent, reason = self._check_shadow_consistency(shadow_directions)

        elapsed = (time.time() - start) * 1000

        return GeometryResult(
            shadow_directions=shadow_directions,
            shadows_consistent=consistent,
            shadow_inconsistency_reason=reason,
            processing_time_ms=elapsed
        )

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel operator."""
        if HAS_CV2:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            return edges.astype(np.float64) / 255.0

        # Fallback: Sobel magnitude
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        gx = convolve2d(gray, sobel_x, mode='same', boundary='symm')
        gy = convolve2d(gray, sobel_y, mode='same', boundary='symm')

        magnitude = np.sqrt(gx**2 + gy**2)
        threshold = np.percentile(magnitude, 90)

        return (magnitude > threshold).astype(np.float64)

    def _detect_lines(
        self,
        edges: np.ndarray,
        width: int,
        height: int
    ) -> List[Line2D]:
        """Detect line segments in edge image."""
        lines = []

        if HAS_CV2:
            edge_uint8 = (edges * 255).astype(np.uint8)
            detected = cv2.HoughLinesP(
                edge_uint8,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=min(width, height) // 20,
                maxLineGap=10
            )

            if detected is not None:
                for line in detected:
                    x1, y1, x2, y2 = line[0]
                    lines.append(Line2D(x1, y1, x2, y2))

        return lines

    def _estimate_shadow_directions(
        self,
        lines: List[Line2D],
        width: int,
        height: int
    ) -> List[ShadowDirection]:
        """Estimate shadow directions from detected lines."""
        if not lines:
            return []

        # Cluster lines by angle
        clusters = cluster_lines_by_angle(lines, angle_threshold=15.0)

        directions = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Estimate direction from cluster
            angles = [line.angle_degrees for line in cluster]
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            # Confidence based on consistency
            confidence = 1.0 / (1.0 + std_angle / 10.0)

            # Determine region
            midpoints = [line.midpoint for line in cluster]
            min_x = min(p[0] for p in midpoints)
            max_x = max(p[0] for p in midpoints)
            min_y = min(p[1] for p in midpoints)
            max_y = max(p[1] for p in midpoints)

            directions.append(ShadowDirection(
                angle=mean_angle,
                confidence=confidence,
                region=(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)),
                supporting_edges=len(cluster)
            ))

        return directions

    def _check_shadow_consistency(
        self,
        directions: List[ShadowDirection]
    ) -> Tuple[bool, Optional[str]]:
        """Check if shadow directions are consistent."""
        if len(directions) < 2:
            return True, None  # Can't check with fewer than 2 directions

        # Get high-confidence directions
        confident = [d for d in directions if d.confidence > 0.5]
        if len(confident) < 2:
            return True, None

        # Check pairwise angle differences
        angles = [d.angle for d in confident]

        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = angle_difference(angles[i], angles[j])
                if diff > self.DIRECTION_TOLERANCE:
                    return False, (
                        f"Shadow directions differ by {diff:.0f}° "
                        f"(regions at {confident[i].region[:2]} and {confident[j].region[:2]})"
                    )

        return True, None

    def _uncertain_result(self, reason: str) -> DimensionResult:
        """Return uncertain result."""
        return DimensionResult(
            dimension="shadows",
            state=DimensionState.UNCERTAIN,
            confidence=Confidence.NA,
            evidence=[Evidence(finding="Analysis not possible", explanation=reason)]
        )

    def _build_result(self, result: GeometryResult) -> DimensionResult:
        """Build DimensionResult from analysis."""
        evidence = []

        if result.shadow_directions:
            n_dirs = len(result.shadow_directions)
            evidence.append(Evidence(
                finding=f"Detected {n_dirs} shadow direction(s)",
                explanation="Shadow boundaries analyzed for directional consistency"
            ))

            for i, sd in enumerate(result.shadow_directions[:3]):
                evidence.append(Evidence(
                    finding=f"Shadow direction {i+1}: {sd.angle:.0f}° (confidence: {sd.confidence:.0%})",
                    explanation=f"Region: {sd.region}, supporting edges: {sd.supporting_edges}"
                ))

        if not result.shadows_consistent:
            evidence.append(Evidence(
                finding="SHADOW INCONSISTENCY DETECTED",
                explanation=result.shadow_inconsistency_reason or "Shadow directions are inconsistent",
                citation="Kee et al. (2013) - Shadow consistency analysis"
            ))

            return DimensionResult(
                dimension="shadows",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=evidence,
                methodology="Shadow direction consistency analysis"
            )

        evidence.append(Evidence(
            finding="Shadow directions consistent",
            explanation="No geometric inconsistencies detected in shadow directions"
        ))

        return DimensionResult(
            dimension="shadows",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="Shadow direction consistency analysis"
        )


# =============================================================================
# PERSPECTIVE ANALYZER
# =============================================================================

class PerspectiveAnalyzer:
    """
    Analyzes perspective consistency via vanishing point detection.

    In a real photograph:
    1. Parallel lines in 3D converge to vanishing points in 2D
    2. All horizontal parallels converge to points on the horizon line
    3. Vertical lines either converge (tilted camera) or are parallel

    Manipulation detection:
    - Different regions having incompatible vanishing points
    - Vanishing points that should align (horizon) but don't
    - Geometric impossibilities in perspective projection
    """

    MIN_LINES_FOR_VP = 4  # Minimum lines to estimate vanishing point
    VP_CONSISTENCY_THRESHOLD = 50  # Pixels tolerance for horizon consistency

    def analyze(self, file_path: str) -> DimensionResult:
        """Analyze perspective consistency in image."""
        if not HAS_NUMPY or not HAS_PIL:
            return self._uncertain_result("Dependencies not available")

        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = np.array(img)
        except Exception as e:
            return self._uncertain_result(f"Cannot load image: {e}")

        result = self._analyze_perspective(image)
        return self._build_result(result)

    def _analyze_perspective(self, image: np.ndarray) -> GeometryResult:
        """Perform perspective analysis."""
        import time
        start = time.time()

        height, width = image.shape[:2]

        # Convert to grayscale
        gray = np.mean(image, axis=2).astype(np.float64) / 255.0

        # Detect edges
        edges = self._detect_edges(gray)

        # Detect line segments
        lines = self._detect_lines(edges, width, height)

        # Cluster lines and find vanishing points
        vanishing_points = self._find_vanishing_points(lines, width, height)

        # Check consistency
        consistent, reason = self._check_vp_consistency(vanishing_points, width, height)

        elapsed = (time.time() - start) * 1000

        return GeometryResult(
            vanishing_points=vanishing_points,
            vp_consistent=consistent,
            vp_inconsistency_reason=reason,
            processing_time_ms=elapsed
        )

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges using Canny or Sobel."""
        if HAS_CV2:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            return edges.astype(np.float64) / 255.0

        # Fallback
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        gx = convolve2d(gray, sobel_x, mode='same', boundary='symm')
        gy = convolve2d(gray, sobel_y, mode='same', boundary='symm')

        magnitude = np.sqrt(gx**2 + gy**2)
        threshold = np.percentile(magnitude, 90)

        return (magnitude > threshold).astype(np.float64)

    def _detect_lines(
        self,
        edges: np.ndarray,
        width: int,
        height: int
    ) -> List[Line2D]:
        """Detect line segments."""
        lines = []

        if HAS_CV2:
            edge_uint8 = (edges * 255).astype(np.uint8)
            detected = cv2.HoughLinesP(
                edge_uint8,
                rho=1,
                theta=np.pi/180,
                threshold=80,
                minLineLength=min(width, height) // 10,
                maxLineGap=10
            )

            if detected is not None:
                for line in detected:
                    x1, y1, x2, y2 = line[0]
                    lines.append(Line2D(x1, y1, x2, y2))

        return lines

    def _find_vanishing_points(
        self,
        lines: List[Line2D],
        width: int,
        height: int
    ) -> List[VanishingPoint]:
        """Find vanishing points from line clusters."""
        if len(lines) < self.MIN_LINES_FOR_VP:
            return []

        # Cluster lines by angle
        clusters = cluster_lines_by_angle(lines, angle_threshold=15.0)

        vanishing_points = []
        for cluster in clusters:
            if len(cluster) < self.MIN_LINES_FOR_VP:
                continue

            vp = estimate_vanishing_point(cluster)
            if vp and vp.confidence > 0.3:
                vanishing_points.append(vp)

        return vanishing_points

    def _check_vp_consistency(
        self,
        vanishing_points: List[VanishingPoint],
        width: int,
        height: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if vanishing points are geometrically consistent."""
        if len(vanishing_points) < 2:
            return True, None

        # Separate horizontal and vertical VPs
        horizontal_vps = [vp for vp in vanishing_points if vp.direction == "horizontal"]
        vertical_vps = [vp for vp in vanishing_points if vp.direction == "vertical"]

        # Check horizon consistency: horizontal VPs should have similar y-coordinates
        if len(horizontal_vps) >= 2:
            y_coords = [vp.y for vp in horizontal_vps if not vp.is_at_infinity]
            if len(y_coords) >= 2:
                y_range = max(y_coords) - min(y_coords)
                if y_range > self.VP_CONSISTENCY_THRESHOLD:
                    return False, (
                        f"Horizontal vanishing points don't align on horizon "
                        f"(y-coordinate range: {y_range:.0f}px)"
                    )

        # Check for impossible geometry:
        # If we have 3+ strong VPs, they should be geometrically possible
        confident_vps = [vp for vp in vanishing_points if vp.confidence > 0.6]
        if len(confident_vps) >= 3:
            # In real images, VPs form specific patterns
            # This is a simplified check - full verification needs camera calibration
            pass

        return True, None

    def _uncertain_result(self, reason: str) -> DimensionResult:
        """Return uncertain result."""
        return DimensionResult(
            dimension="perspective",
            state=DimensionState.UNCERTAIN,
            confidence=Confidence.NA,
            evidence=[Evidence(finding="Analysis not possible", explanation=reason)]
        )

    def _build_result(self, result: GeometryResult) -> DimensionResult:
        """Build DimensionResult from analysis."""
        evidence = []

        if result.vanishing_points:
            n_vps = len(result.vanishing_points)
            evidence.append(Evidence(
                finding=f"Detected {n_vps} vanishing point(s)",
                explanation="Line convergence analyzed for perspective consistency"
            ))

            for i, vp in enumerate(result.vanishing_points[:3]):
                if vp.is_at_infinity:
                    loc = "at infinity (parallel lines)"
                else:
                    loc = f"({vp.x:.0f}, {vp.y:.0f})"
                evidence.append(Evidence(
                    finding=f"VP {i+1} ({vp.direction}): {loc}",
                    explanation=f"Confidence: {vp.confidence:.0%}, supporting lines: {vp.supporting_lines}"
                ))

        if not result.vp_consistent:
            evidence.append(Evidence(
                finding="PERSPECTIVE INCONSISTENCY DETECTED",
                explanation=result.vp_inconsistency_reason or "Vanishing points are inconsistent",
                citation="Criminisi & Zisserman (2000) - Perspective geometry analysis"
            ))

            return DimensionResult(
                dimension="perspective",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=evidence,
                methodology="Vanishing point consistency analysis"
            )

        evidence.append(Evidence(
            finding="Perspective geometry consistent",
            explanation="No geometric inconsistencies detected in vanishing points"
        ))

        return DimensionResult(
            dimension="perspective",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="Vanishing point consistency analysis"
        )


# =============================================================================
# COMBINED GEOMETRIC ANALYZER
# =============================================================================

class GeometricAnalyzer:
    """
    Combined shadow and perspective analysis.

    Analyzes geometric consistency across multiple constraints:
    - Shadow directions
    - Vanishing points
    - (Future: reflections, light source estimation)

    The power is in combination: if shadows indicate light from the left,
    but reflections show light from the right, that's impossible geometry.
    """

    def __init__(self):
        self.shadow_analyzer = ShadowAnalyzer()
        self.perspective_analyzer = PerspectiveAnalyzer()

    def analyze_shadows(self, file_path: str) -> DimensionResult:
        """Analyze shadow consistency."""
        return self.shadow_analyzer.analyze(file_path)

    def analyze_perspective(self, file_path: str) -> DimensionResult:
        """Analyze perspective consistency."""
        return self.perspective_analyzer.analyze(file_path)
