"""
Epistemic aggregator for forensic dimension results.

Combines results from multiple analysis dimensions into an overall
assessment with proper epistemic reasoning. Unlike simple voting,
this aggregator respects the asymmetry between proving manipulation
and proving authenticity.

Key principle: A single INCONSISTENT finding (e.g., "iPhone 6 claiming 4K")
is sufficient for overall concern, while CONSISTENT findings merely indicate
absence of detected anomalies, not proof of authenticity.

This module also provides cross-dimension corroboration, identifying when
multiple independent analyses point toward the same conclusion. Such
convergent evidence is forensically significant because each dimension
examines different technical aspects of the file; independent agreement
strengthens the overall inference substantially.

References:
    Farid, H. (2016). Photo Forensics. MIT Press.
    Scientific Working Group on Imaging Technology (SWGIT) guidelines.
"""

from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from .state import (
    DimensionResult,
    DimensionState,
    Confidence,
    OverallAssessment,
    WuAnalysis,
)


# =============================================================================
# CORROBORATION CATEGORIES
# =============================================================================

# Categories of forensic conclusions that dimensions can support.
# Each category represents a type of manipulation or inconsistency
# that multiple independent dimensions might detect.

CORROBORATION_CATEGORIES = {
    "post_capture_editing": {
        "description": "Evidence of editing after initial capture",
        "dimensions": ["metadata", "thumbnail", "quantization", "visual"],
        "narrative": (
            "Multiple independent analyses suggest that this file was modified "
            "after its initial capture. {details}"
        ),
    },
    "compression_history": {
        "description": "Evidence of multiple compression passes",
        "dimensions": ["quantization", "blockgrid", "visual"],
        "narrative": (
            "The compression characteristics indicate that this image has been "
            "saved multiple times, which is consistent with editing and re-saving. {details}"
        ),
    },
    "geometric_inconsistency": {
        "description": "Physically impossible geometry in the scene",
        "dimensions": ["shadows", "perspective", "lighting"],
        "narrative": (
            "The geometric properties of the scene are internally inconsistent, "
            "suggesting that elements from different sources have been combined. {details}"
        ),
    },
    "splicing_detected": {
        "description": "Evidence of content from multiple sources",
        "dimensions": ["blockgrid", "copymove", "prnu", "lighting"],
        "narrative": (
            "Technical analysis indicates that portions of this image originated "
            "from different source files or were duplicated within the image. {details}"
        ),
    },
    "metadata_content_mismatch": {
        "description": "Metadata does not match visible content",
        "dimensions": ["metadata", "thumbnail", "c2pa"],
        "narrative": (
            "The file metadata is inconsistent with the actual content, suggesting "
            "either metadata manipulation or content substitution. {details}"
        ),
    },
    "device_attribution_conflict": {
        "description": "Conflicting evidence about capture device",
        "dimensions": ["metadata", "prnu", "quantization"],
        "narrative": (
            "The evidence regarding which device captured this image is contradictory, "
            "which may indicate metadata falsification or image compositing. {details}"
        ),
    },
}


@dataclass
class CorroboratingEvidence:
    """A group of dimensions that independently support the same conclusion."""
    category: str
    category_description: str
    supporting_dimensions: List[str]
    narrative: str
    strength: str  # "strong" (3+ dimensions), "moderate" (2 dimensions)


# =============================================================================
# DIMENSION TO CATEGORY MAPPING
# =============================================================================

def get_relevant_categories(dimension: str) -> List[str]:
    """Get categories that a dimension can contribute evidence toward."""
    categories = []
    for cat_name, cat_info in CORROBORATION_CATEGORIES.items():
        if dimension in cat_info["dimensions"]:
            categories.append(cat_name)
    return categories


class EpistemicAggregator:
    """
    Aggregates dimension results into overall assessment.

    Follows epistemic asymmetry principle:
    - Inconsistency in ANY dimension is significant
    - Consistency in ALL dimensions merely means "no issues found"
    - Absence of evidence is not evidence of absence
    """

    def aggregate(self, results: List[DimensionResult]) -> OverallAssessment:
        """
        Combine dimension results into overall assessment.

        Logic:
        1. Any INCONSISTENT/TAMPERED/INVALID -> INCONSISTENCIES_DETECTED
        2. Any SUSPICIOUS -> ANOMALIES_DETECTED
        3. All CONSISTENT/VERIFIED -> NO_ANOMALIES (not "authentic"!)
        4. Otherwise -> INSUFFICIENT_DATA
        """
        if not results:
            return OverallAssessment.INSUFFICIENT_DATA

        has_inconsistency = any(r.is_problematic for r in results)
        has_anomaly = any(r.is_suspicious for r in results)
        all_clean = all(r.is_clean for r in results)
        all_uncertain = all(r.state == DimensionState.UNCERTAIN for r in results)

        if has_inconsistency:
            return OverallAssessment.INCONSISTENCIES_DETECTED
        elif has_anomaly:
            return OverallAssessment.ANOMALIES_DETECTED
        elif all_uncertain:
            return OverallAssessment.INSUFFICIENT_DATA
        elif all_clean:
            return OverallAssessment.NO_ANOMALIES
        else:
            # Mixed uncertain and clean
            return OverallAssessment.NO_ANOMALIES

    def generate_summary(self, results: List[DimensionResult]) -> List[str]:
        """
        Generate human-readable findings summary.

        Prioritizes:
        1. Inconsistencies (definite problems)
        2. Suspicious findings (warrant investigation)
        3. Clean findings (no issues detected)
        """
        summary = []

        # Report inconsistencies first
        for r in results:
            if r.is_problematic:
                for evidence in r.evidence:
                    if evidence.contradiction:
                        summary.append(
                            f"[{r.dimension.upper()}] {evidence.finding}: "
                            f"{evidence.contradiction}"
                        )
                    else:
                        summary.append(
                            f"[{r.dimension.upper()}] {evidence.finding}: "
                            f"{evidence.explanation}"
                        )

        # Then suspicious findings
        for r in results:
            if r.is_suspicious:
                for evidence in r.evidence:
                    summary.append(
                        f"[{r.dimension.upper()}] Suspicious: {evidence.finding}"
                    )

        # Note clean dimensions
        clean_dims = [r.dimension for r in results if r.is_clean]
        if clean_dims:
            summary.append(
                f"No anomalies detected in: {', '.join(clean_dims)}"
            )

        # Note uncertain dimensions
        uncertain_dims = [
            r.dimension for r in results
            if r.state == DimensionState.UNCERTAIN
        ]
        if uncertain_dims:
            summary.append(
                f"Insufficient data for: {', '.join(uncertain_dims)}"
            )

        return summary

    def calculate_overall_confidence(
        self,
        results: List[DimensionResult]
    ) -> Confidence:
        """
        Calculate confidence in overall assessment.

        High confidence requires:
        - Multiple dimensions analyzed
        - At least one HIGH confidence finding
        - No contradictory states between dimensions
        """
        if not results:
            return Confidence.NA

        # Filter out uncertain results
        definite_results = [
            r for r in results
            if r.state != DimensionState.UNCERTAIN
        ]

        if not definite_results:
            return Confidence.NA

        # Check for HIGH confidence findings
        has_high = any(r.confidence == Confidence.HIGH for r in definite_results)

        # Any inconsistency with HIGH confidence is definitive
        for r in definite_results:
            if r.is_problematic and r.confidence == Confidence.HIGH:
                return Confidence.HIGH

        # Multiple clean dimensions with high confidence
        clean_high = [
            r for r in definite_results
            if r.is_clean and r.confidence == Confidence.HIGH
        ]
        if len(clean_high) >= 2:
            return Confidence.HIGH

        if has_high:
            return Confidence.MEDIUM

        return Confidence.LOW

    def find_corroborating_evidence(
        self,
        results: List[DimensionResult]
    ) -> List[CorroboratingEvidence]:
        """
        Identify cases where multiple independent dimensions support the same conclusion.

        Corroborating evidence is forensically significant because each dimension
        examines different technical aspects of the file. When multiple unrelated
        analyses independently point toward the same conclusion, the combined
        evidence is substantially stronger than any single finding would be alone.

        This method does not assign numerical scores or probabilities; instead,
        it identifies convergent findings and generates narrative descriptions
        suitable for inclusion in forensic reports.

        Returns:
            List of CorroboratingEvidence objects describing convergent findings
        """
        if not results:
            return []

        # Find dimensions with problematic or suspicious findings
        concerning_dims = {
            r.dimension: r for r in results
            if r.is_problematic or r.is_suspicious
        }

        if len(concerning_dims) < 2:
            return []

        corroborations = []

        # Check each category for convergent evidence
        for cat_name, cat_info in CORROBORATION_CATEGORIES.items():
            # Find dimensions in this category that have concerning findings
            supporting = []
            details_parts = []

            for dim_name in cat_info["dimensions"]:
                if dim_name in concerning_dims:
                    dim_result = concerning_dims[dim_name]
                    supporting.append(dim_name)

                    # Collect key findings for narrative
                    for ev in dim_result.evidence[:2]:  # First two findings
                        if ev.finding and "consistent" not in ev.finding.lower():
                            details_parts.append(
                                f"{dim_name} analysis: {ev.finding}"
                            )

            # Need at least 2 dimensions for corroboration
            if len(supporting) >= 2:
                # Determine strength
                strength = "strong" if len(supporting) >= 3 else "moderate"

                # Build details string
                if details_parts:
                    details = "Specifically, " + "; ".join(details_parts[:3]) + "."
                else:
                    details = ""

                # Generate narrative
                narrative = cat_info["narrative"].format(details=details)

                corroborations.append(CorroboratingEvidence(
                    category=cat_name,
                    category_description=cat_info["description"],
                    supporting_dimensions=supporting,
                    narrative=narrative,
                    strength=strength,
                ))

        # Sort by strength (strong first) then by number of dimensions
        corroborations.sort(
            key=lambda c: (0 if c.strength == "strong" else 1, -len(c.supporting_dimensions))
        )

        return corroborations

    def generate_corroboration_summary(
        self,
        results: List[DimensionResult]
    ) -> Optional[str]:
        """
        Generate a narrative summary of corroborating evidence.

        This summary is suitable for inclusion in forensic reports and
        explains in plain language when multiple analyses converge on
        the same conclusion.

        Returns:
            Narrative string, or None if no corroboration found
        """
        corroborations = self.find_corroborating_evidence(results)

        if not corroborations:
            return None

        parts = []

        # Introductory statement
        if len(corroborations) == 1:
            parts.append(
                "The following convergent evidence was identified, where multiple "
                "independent analytical methods point toward the same conclusion:"
            )
        else:
            parts.append(
                f"The analysis identified {len(corroborations)} areas where multiple "
                "independent analytical methods converge on the same conclusion. "
                "Such convergent evidence is forensically significant because each "
                "method examines different technical properties of the file."
            )

        # Detail each corroboration
        for i, corr in enumerate(corroborations, 1):
            dim_list = ", ".join(corr.supporting_dimensions)
            strength_text = (
                "strongly supported" if corr.strength == "strong"
                else "supported"
            )

            parts.append("")
            parts.append(
                f"Finding {i}: {corr.category_description} "
                f"({strength_text} by {len(corr.supporting_dimensions)} dimensions: {dim_list})"
            )
            parts.append(corr.narrative)

        return "\n".join(parts)
