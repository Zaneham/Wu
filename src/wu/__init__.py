"""
Wu - Epistemic Media Forensics Toolkit.

Detects manipulated media with structured uncertainty output
suitable for court admissibility (Daubert standard).

Named after Chien-Shiung Wu (1912-1997), who disproved parity
conservation and found asymmetries everyone assumed didn't exist.

References:
    Wu, C.S. et al. (1957). Experimental Test of Parity Conservation
        in Beta Decay. Physical Review, 105(4), 1413-1415.
"""

from importlib.metadata import version as _get_version, PackageNotFoundError

try:
    __version__ = _get_version("wu-forensics")
except PackageNotFoundError:
    # Package not installed (running from source)
    __version__ = "0.0.0-dev"

from .state import (
    DimensionState,
    DimensionResult,
    WuAnalysis,
    OverallAssessment,
    Confidence,
    Evidence,
    CorrelationWarning,
    AuthenticityAssessment,
    AuthenticityResult,
)
from .analyzer import WuAnalyzer
from .aggregator import EpistemicAggregator, AuthenticityAggregator
from .correlator import DimensionCorrelator
from .report import ForensicReportGenerator, generate_report
from .dimensions import (
    MetadataAnalyzer,
    C2PAAnalyzer,
    GPSAnalyzer,
    GPSCoordinate,
    VisualAnalyzer,
    ENFAnalyzer,
    ENFSignal,
    GridRegion,
    CopyMoveAnalyzer,
    CloneRegion,
    PRNUAnalyzer,
    PRNUFingerprint,
    BlockGridAnalyzer,
    BlockGridOffset,
    LightingAnalyzer,
    LightVector,
)

__all__ = [
    "__version__",
    # Main interface
    "WuAnalyzer",
    "EpistemicAggregator",
    "AuthenticityAggregator",
    "ForensicReportGenerator",
    "generate_report",
    # State model
    "DimensionState",
    "DimensionResult",
    "WuAnalysis",
    "OverallAssessment",
    "Confidence",
    "Evidence",
    "CorrelationWarning",
    "AuthenticityAssessment",
    "AuthenticityResult",
    "DimensionCorrelator",
    # Dimension analysers
    "MetadataAnalyzer",
    "C2PAAnalyzer",
    "GPSAnalyzer",
    "GPSCoordinate",
    "VisualAnalyzer",
    "ENFAnalyzer",
    "ENFSignal",
    "GridRegion",
    "CopyMoveAnalyzer",
    "CloneRegion",
    "PRNUAnalyzer",
    "PRNUFingerprint",
    "BlockGridAnalyzer",
    "BlockGridOffset",
    "LightingAnalyzer",
    "LightVector",
]
