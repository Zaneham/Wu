"""
Forensic analysis dimensions.

Each dimension analyzes a specific aspect of the media file
and returns a DimensionResult with epistemic state.
"""

from .metadata import MetadataAnalyzer
from .c2pa import C2PAAnalyzer
from .gps import GPSAnalyzer, GPSCoordinate
from .visual import VisualAnalyzer
from .enf import ENFAnalyzer, ENFSignal, GridRegion
from .copymove import CopyMoveAnalyzer, CloneRegion
from .prnu import PRNUAnalyzer, PRNUFingerprint
from .blockgrid import BlockGridAnalyzer, BlockGridOffset
from .lighting import LightingAnalyzer, LightVector
from .audio import AudioAnalyzer, SpectralDiscontinuity, AudioENFResult
from .thumbnail import ThumbnailAnalyzer, ThumbnailComparison
from .geometry import (
    ShadowAnalyzer,
    PerspectiveAnalyzer,
    GeometricAnalyzer,
    Line2D,
    VanishingPoint,
    ShadowDirection,
    GeometryResult,
)
from .quantization import (
    QuantizationAnalyzer,
    QuantizationTable,
    QuantizationResult,
    DoubleCompressionResult,
    SoftwareMatch,
)
from .aigen import (
    AIGenerationAnalyzer,
    AIGenAnalysis,
    FrequencyAnalysis,
    CheckerboardAnalysis,
    NoiseAnalysis,
    ColorAnalysis,
)

from .lipsync import (
    LipSyncAnalyzer,
    PhonemeClass,
    Viseme,
    SyncResult,
)

__all__ = [
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
    "AudioAnalyzer",
    "SpectralDiscontinuity",
    "AudioENFResult",
    "ThumbnailAnalyzer",
    "ThumbnailComparison",
    "ShadowAnalyzer",
    "PerspectiveAnalyzer",
    "GeometricAnalyzer",
    "Line2D",
    "VanishingPoint",
    "ShadowDirection",
    "GeometryResult",
    "QuantizationAnalyzer",
    "QuantizationTable",
    "QuantizationResult",
    "DoubleCompressionResult",
    "SoftwareMatch",
    "AIGenerationAnalyzer",
    "AIGenAnalysis",
    "FrequencyAnalysis",
    "CheckerboardAnalysis",
    "NoiseAnalysis",
    "ColorAnalysis",
    "LipSyncAnalyzer",
    "PhonemeClass",
    "Viseme",
    "SyncResult",
]
