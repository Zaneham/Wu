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

from .state import (
    DimensionState,
    DimensionResult,
    WuAnalysis,
    OverallAssessment,
    Confidence,
    Evidence,
)
from .analyzer import WuAnalyzer
from .aggregator import EpistemicAggregator
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

__version__ = "1.1.0"
__all__ = [
    # Main interface
    "WuAnalyzer",
    "EpistemicAggregator",
    "ForensicReportGenerator",
    "generate_report",
    # State model
    "DimensionState",
    "DimensionResult",
    "WuAnalysis",
    "OverallAssessment",
    "Confidence",
    "Evidence",
    # Dimension analyzers
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
