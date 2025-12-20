"""
Epistemic state model for forensic analysis dimensions.

Each dimension (metadata, visual, audio, etc.) produces a DimensionResult
with a state indicating consistency/inconsistency and supporting evidence.

Unlike binary fake/real classifiers, Wu provides structured uncertainty
that courts can evaluate dimension by dimension.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime


class DimensionState(Enum):
    """
    Epistemic states for a forensic dimension.

    Designed for legal clarity - each state has clear meaning
    that can be explained to a jury.
    """
    CONSISTENT = "consistent"       # No anomalies detected
    INCONSISTENT = "inconsistent"   # Clear contradictions found
    SUSPICIOUS = "suspicious"       # Anomalies that warrant investigation
    UNCERTAIN = "uncertain"         # Insufficient data for analysis

    # Special states for C2PA
    VERIFIED = "verified"           # Valid content credentials
    TAMPERED = "tampered"           # Credentials present but file modified
    MISSING = "missing"             # No credentials (not necessarily fake)
    INVALID = "invalid"             # Credentials present but invalid


class Confidence(Enum):
    """Confidence level in the finding."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NA = "n/a"  # Not applicable (e.g., UNCERTAIN state)


@dataclass
class Evidence:
    """A single piece of evidence supporting a finding."""
    finding: str
    explanation: str
    contradiction: Optional[str] = None
    citation: Optional[str] = None
    timestamp: Optional[str] = None  # For temporal evidence

    def to_dict(self) -> Dict[str, Any]:
        d = {"finding": self.finding, "explanation": self.explanation}
        if self.contradiction:
            d["contradiction"] = self.contradiction
        if self.citation:
            d["citation"] = self.citation
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


@dataclass
class DimensionResult:
    """
    Result of analyzing a single forensic dimension.

    Each dimension (metadata, visual, audio, etc.) produces one of these.
    The epistemic aggregator combines them into an overall assessment.
    """
    dimension: str
    state: DimensionState
    confidence: Confidence
    evidence: List[Evidence] = field(default_factory=list)
    methodology: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None  # Additional structured data for debugging

    @property
    def is_problematic(self) -> bool:
        """True if this dimension found issues."""
        return self.state in (
            DimensionState.INCONSISTENT,
            DimensionState.TAMPERED,
            DimensionState.INVALID
        )

    @property
    def is_suspicious(self) -> bool:
        """True if this dimension warrants further investigation."""
        return self.state == DimensionState.SUSPICIOUS

    @property
    def is_clean(self) -> bool:
        """True if no issues detected."""
        return self.state in (
            DimensionState.CONSISTENT,
            DimensionState.VERIFIED
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "dimension": self.dimension,
            "state": self.state.value,
            "confidence": self.confidence.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "methodology": self.methodology,
        }
        if self.raw_data:
            result["raw_data"] = self.raw_data
        return result


class OverallAssessment(Enum):
    """Overall assessment after aggregating all dimensions."""
    NO_ANOMALIES = "no_anomalies_detected"
    ANOMALIES_DETECTED = "anomalies_detected"
    INCONSISTENCIES_DETECTED = "inconsistencies_detected"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class WuAnalysis:
    """
    Complete analysis result from Wu.

    Contains per-dimension results plus aggregated assessment.
    Can generate court-ready reports.
    """
    file_path: str
    file_hash: str  # SHA256 of analyzed file
    analyzed_at: datetime
    wu_version: str

    # Dimension results
    metadata: Optional[DimensionResult] = None
    visual: Optional[DimensionResult] = None
    audio: Optional[DimensionResult] = None
    temporal: Optional[DimensionResult] = None
    c2pa: Optional[DimensionResult] = None
    enf: Optional[DimensionResult] = None  # Electric Network Frequency
    copymove: Optional[DimensionResult] = None  # Copy-move (clone) detection
    prnu: Optional[DimensionResult] = None  # Sensor fingerprint analysis
    blockgrid: Optional[DimensionResult] = None  # JPEG block grid alignment
    lighting: Optional[DimensionResult] = None  # Lighting direction consistency
    thumbnail: Optional[DimensionResult] = None  # EXIF thumbnail mismatch
    shadows: Optional[DimensionResult] = None  # Shadow direction consistency
    perspective: Optional[DimensionResult] = None  # Vanishing point consistency
    quantization: Optional[DimensionResult] = None  # JPEG quantization table forensics
    aigen: Optional[DimensionResult] = None  # AI generation indicators

    # Aggregated
    overall: OverallAssessment = OverallAssessment.INSUFFICIENT_DATA
    findings_summary: List[str] = field(default_factory=list)
    corroboration_summary: Optional[str] = None  # Narrative of convergent findings

    @property
    def dimensions(self) -> List[DimensionResult]:
        """All analyzed dimensions."""
        return [d for d in [
            self.metadata, self.visual, self.audio,
            self.temporal, self.c2pa, self.enf, self.copymove, self.prnu,
            self.blockgrid, self.lighting, self.thumbnail, self.shadows,
            self.perspective, self.quantization, self.aigen
        ] if d is not None]

    @property
    def has_inconsistencies(self) -> bool:
        """True if any dimension found inconsistencies."""
        return any(d.is_problematic for d in self.dimensions)

    @property
    def has_anomalies(self) -> bool:
        """True if any dimension is suspicious."""
        return any(d.is_suspicious for d in self.dimensions)

    @property
    def is_clean(self) -> bool:
        """True if all analyzed dimensions are clean."""
        dims = self.dimensions
        return len(dims) > 0 and all(d.is_clean for d in dims)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "analyzed_at": self.analyzed_at.isoformat(),
            "wu_version": self.wu_version,
            "overall_assessment": self.overall.value,
            "findings_summary": self.findings_summary,
            "corroboration_summary": self.corroboration_summary,
            "dimensions": {
                "metadata": self.metadata.to_dict() if self.metadata else None,
                "visual": self.visual.to_dict() if self.visual else None,
                "audio": self.audio.to_dict() if self.audio else None,
                "temporal": self.temporal.to_dict() if self.temporal else None,
                "c2pa": self.c2pa.to_dict() if self.c2pa else None,
                "enf": self.enf.to_dict() if self.enf else None,
                "copymove": self.copymove.to_dict() if self.copymove else None,
                "prnu": self.prnu.to_dict() if self.prnu else None,
                "blockgrid": self.blockgrid.to_dict() if self.blockgrid else None,
                "lighting": self.lighting.to_dict() if self.lighting else None,
                "thumbnail": self.thumbnail.to_dict() if self.thumbnail else None,
                "shadows": self.shadows.to_dict() if self.shadows else None,
                "perspective": self.perspective.to_dict() if self.perspective else None,
                "quantization": self.quantization.to_dict() if self.quantization else None,
                "aigen": self.aigen.to_dict() if self.aigen else None,
            }
        }

    def to_json(self) -> str:
        import json

        class WuEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                if hasattr(obj, "item"):
                    return obj.item()
                if hasattr(obj, "value") and hasattr(obj, "__enum__"):
                    return obj.value
                return str(obj)

        return json.dumps(self.to_dict(), indent=2, cls=WuEncoder)

