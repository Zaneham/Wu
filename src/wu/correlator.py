"""
Dimension correlation engine for cross-referencing forensic findings.

Analyses relationships between dimension results to detect contradictions
that individual dimensions might miss. For example, metadata claiming
Canon but PRNU matching iPhone is a critical conflict.

This layer runs after individual dimension analysis and before report
generation, adding correlation warnings to the final WuAnalysis.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import WuAnalysis

from .state import CorrelationWarning, DimensionState


# Severity ordering for sorting warnings (most severe first)
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Countries/regions using 60Hz power grid (others use 50Hz)
REGIONS_60HZ = {
    "US", "USA", "United States", "America",
    "CA", "Canada",
    "MX", "Mexico",
    "BR", "Brazil",
    "JP", "Japan",  # Mixed, but includes 60Hz regions
    "TW", "Taiwan",
    "KR", "South Korea", "Korea",
    "PH", "Philippines",
    "SA", "Saudi Arabia",
}


class DimensionCorrelator:
    """
    Cross-references dimension findings to detect contradictions.

    Each check method examines specific dimension pairs for conflicts.
    Missing dimensions are gracefully handled (no warning generated).
    """

    def correlate(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Run all correlation checks and return sorted warnings.

        Args:
            analysis: Complete WuAnalysis with dimension results

        Returns:
            List of CorrelationWarning sorted by severity (critical first)
        """
        warnings = []
        warnings.extend(self._check_device_attribution(analysis))
        warnings.extend(self._check_c2pa_conflicts(analysis))
        warnings.extend(self._check_thumbnail_mismatch(analysis))
        warnings.extend(self._check_temporal_consistency(analysis))
        warnings.extend(self._check_compression_history(analysis))
        warnings.extend(self._check_geometric_consistency(analysis))
        warnings.extend(self._check_enf_gps(analysis))
        warnings.extend(self._check_aigen_metadata(analysis))
        warnings.extend(self._check_lipsync_enf(analysis))

        return sorted(warnings, key=lambda w: SEVERITY_ORDER.get(w.severity, 99))

    def _check_device_attribution(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if metadata device claims conflict with PRNU fingerprint.

        CRITICAL if metadata claims Device A but PRNU matches Device B.
        """
        warnings = []

        if not analysis.metadata or not analysis.prnu:
            return warnings

        metadata_raw = analysis.metadata.raw_data or {}
        prnu_raw = analysis.prnu.raw_data or {}

        metadata_make = metadata_raw.get("make", "").strip()
        metadata_model = metadata_raw.get("model", "").strip()
        metadata_device = f"{metadata_make} {metadata_model}".strip()

        if not metadata_device:
            return warnings

        prnu_matches = prnu_raw.get("matches", [])
        for match in prnu_matches:
            if not match.get("matched", False):
                continue

            prnu_camera = match.get("camera_id", "")
            correlation = match.get("correlation", 0)

            if correlation >= 0.7 and prnu_camera:
                meta_norm = metadata_device.lower().replace("corporation", "").strip()
                prnu_norm = prnu_camera.lower().replace("corporation", "").strip()

                if not self._devices_compatible(meta_norm, prnu_norm):
                    warnings.append(CorrelationWarning(
                        severity="critical",
                        category="device_mismatch",
                        dimensions=["metadata", "prnu"],
                        finding=f"Metadata claims '{metadata_device}' but PRNU fingerprint matches '{prnu_camera}' (correlation: {correlation:.2f})",
                        details={
                            "metadata_device": metadata_device,
                            "prnu_device": prnu_camera,
                            "prnu_correlation": correlation,
                        }
                    ))

        return warnings

    def _devices_compatible(self, device_a: str, device_b: str) -> bool:
        """Check if two device strings could refer to the same device."""
        if device_a == device_b:
            return True

        if device_a in device_b or device_b in device_a:
            return True

        make_a = device_a.split()[0] if device_a.split() else ""
        make_b = device_b.split()[0] if device_b.split() else ""
        if make_a and make_b and make_a == make_b:
            return True

        return False

    def _check_c2pa_conflicts(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if C2PA claims authenticity but other dimensions detect manipulation.

        CRITICAL if C2PA says VERIFIED but other dimensions show INCONSISTENT.
        """
        warnings = []

        if not analysis.c2pa:
            return warnings

        if analysis.c2pa.state != DimensionState.VERIFIED:
            return warnings

        manipulation_dims = []
        for dim in analysis.dimensions:
            if dim.dimension == "c2pa":
                continue
            if dim.state == DimensionState.INCONSISTENT:
                manipulation_dims.append(dim.dimension)

        if manipulation_dims:
            warnings.append(CorrelationWarning(
                severity="critical",
                category="c2pa_conflict",
                dimensions=["c2pa"] + manipulation_dims,
                finding=f"C2PA credentials verified as authentic, but {', '.join(manipulation_dims)} detected manipulation",
                details={
                    "c2pa_state": "verified",
                    "conflicting_dimensions": manipulation_dims,
                }
            ))

        return warnings

    def _check_thumbnail_mismatch(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if thumbnail differs significantly but no editing software in metadata.

        HIGH if thumbnail SSIM < 0.9 but metadata shows no editing software.
        """
        warnings = []

        if not analysis.thumbnail or not analysis.metadata:
            return warnings

        thumb_raw = analysis.thumbnail.raw_data or {}
        meta_raw = analysis.metadata.raw_data or {}

        significant_diff = thumb_raw.get("significant_difference", False)
        ssim = thumb_raw.get("similarity", 1.0)

        if not significant_diff:
            return warnings

        software = meta_raw.get("software", "")
        editing_tools = ["photoshop", "gimp", "lightroom", "capture one", "affinity"]
        has_editing_software = any(tool in software.lower() for tool in editing_tools)

        if not has_editing_software:
            warnings.append(CorrelationWarning(
                severity="high",
                category="thumbnail_mismatch",
                dimensions=["thumbnail", "metadata"],
                finding=f"Thumbnail differs from main image (SSIM: {ssim:.2f}) but no editing software detected in metadata",
                details={
                    "ssim": ssim,
                    "metadata_software": software or "none",
                }
            ))

        return warnings

    def _check_temporal_consistency(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check for temporal impossibilities in metadata timestamps.

        HIGH if DateTimeOriginal > DateTimeDigitized (image created after capture).
        """
        warnings = []

        if not analysis.metadata:
            return warnings

        meta_raw = analysis.metadata.raw_data or {}

        dt_original = meta_raw.get("datetime_original")
        dt_digitized = meta_raw.get("datetime_digitized")
        dt_modified = meta_raw.get("datetime_modified")

        if dt_original and dt_digitized:
            try:
                from datetime import datetime
                # Handle various timestamp formats
                for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        orig = datetime.strptime(str(dt_original), fmt)
                        digi = datetime.strptime(str(dt_digitized), fmt)
                        break
                    except ValueError:
                        continue
                else:
                    orig = digi = None

                if orig and digi and orig > digi:
                    warnings.append(CorrelationWarning(
                        severity="high",
                        category="temporal_impossibility",
                        dimensions=["metadata"],
                        finding=f"Image created ({dt_original}) after it was digitised ({dt_digitized})",
                        details={
                            "datetime_original": str(dt_original),
                            "datetime_digitized": str(dt_digitized),
                        }
                    ))
            except Exception:
                pass  # Timestamp parsing failed, skip check

        return warnings

    def _check_compression_history(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if compression claims conflict with detected artifacts.

        MEDIUM if high quality claimed but double compression detected.
        """
        warnings = []

        if not analysis.quantization or not analysis.blockgrid:
            return warnings

        quant_raw = analysis.quantization.raw_data or {}
        block_raw = analysis.blockgrid.raw_data or {}

        quality = quant_raw.get("estimated_quality", 0)
        double_comp = block_raw.get("double_compression_detected", False)

        if quality >= 90 and double_comp:
            warnings.append(CorrelationWarning(
                severity="medium",
                category="compression_conflict",
                dimensions=["quantization", "blockgrid"],
                finding=f"Claims high quality (Q{quality}) but double compression artifacts detected",
                details={
                    "estimated_quality": quality,
                    "double_compression": True,
                }
            ))

        return warnings

    def _check_geometric_consistency(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if lighting and shadow directions are physically consistent.

        MEDIUM if light vector conflicts with shadow direction.
        """
        warnings = []

        if not analysis.lighting or not analysis.shadows:
            return warnings

        light_raw = analysis.lighting.raw_data or {}
        shadow_raw = analysis.shadows.raw_data or {}

        light_elevation = light_raw.get("elevation_degrees")
        shadow_elevation = shadow_raw.get("inferred_elevation_degrees")

        if light_elevation is not None and shadow_elevation is not None:
            diff = abs(light_elevation - shadow_elevation)
            if diff > 30:
                warnings.append(CorrelationWarning(
                    severity="medium",
                    category="geometric_impossibility",
                    dimensions=["lighting", "shadows"],
                    finding=f"Lighting suggests {light_elevation:.0f}° elevation but shadows suggest {shadow_elevation:.0f}° (difference: {diff:.0f}°)",
                    details={
                        "lighting_elevation": light_elevation,
                        "shadow_elevation": shadow_elevation,
                        "difference": diff,
                    }
                ))

        return warnings

    def _check_enf_gps(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if ENF grid frequency matches GPS location's power grid.

        MEDIUM if GPS says Europe (50Hz) but ENF detects 60Hz.
        """
        warnings = []

        if not analysis.enf or not analysis.metadata:
            return warnings

        enf_raw = analysis.enf.raw_data or {}
        meta_raw = analysis.metadata.raw_data or {}

        detected_freq = enf_raw.get("detected_frequency")
        gps_country = meta_raw.get("gps_country", "")

        if not detected_freq or not gps_country:
            return warnings

        expected_60hz = any(region in gps_country for region in REGIONS_60HZ)
        detected_60hz = abs(detected_freq - 60) < abs(detected_freq - 50)

        if expected_60hz != detected_60hz:
            expected = "60Hz" if expected_60hz else "50Hz"
            detected = "60Hz" if detected_60hz else "50Hz"
            warnings.append(CorrelationWarning(
                severity="medium",
                category="enf_gps_mismatch",
                dimensions=["enf", "metadata"],
                finding=f"GPS indicates {gps_country} ({expected} grid) but ENF detected {detected_freq:.1f}Hz ({detected})",
                details={
                    "gps_country": gps_country,
                    "expected_frequency": expected,
                    "detected_frequency": detected_freq,
                }
            ))

        return warnings

    def _check_aigen_metadata(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if AI generation detected but metadata claims photographic capture.

        MEDIUM if AIGen shows high synthetic probability but metadata claims camera.
        """
        warnings = []

        if not analysis.aigen or not analysis.metadata:
            return warnings

        aigen_raw = analysis.aigen.raw_data or {}
        meta_raw = analysis.metadata.raw_data or {}

        synthetic_prob = aigen_raw.get("synthetic_probability", 0)
        metadata_make = meta_raw.get("make", "")
        metadata_model = meta_raw.get("model", "")

        if synthetic_prob >= 0.7 and (metadata_make or metadata_model):
            device = f"{metadata_make} {metadata_model}".strip()
            warnings.append(CorrelationWarning(
                severity="medium",
                category="aigen_metadata_conflict",
                dimensions=["aigen", "metadata"],
                finding=f"AI generation detected ({synthetic_prob*100:.0f}% probability) but metadata claims capture by {device}",
                details={
                    "synthetic_probability": synthetic_prob,
                    "claimed_device": device,
                }
            ))

        return warnings

    def _check_lipsync_enf(self, analysis: "WuAnalysis") -> List[CorrelationWarning]:
        """
        Check if lip-sync desync detected but ENF is continuous.

        HIGH if audio doesn't match lips but ENF shows no discontinuity,
        suggesting audio was replaced with recording from same grid region.
        """
        warnings = []

        if not analysis.lipsync or not analysis.enf:
            return warnings

        lipsync_raw = analysis.lipsync.raw_data or {}
        enf_raw = analysis.enf.raw_data or {}

        offset_ms = abs(lipsync_raw.get("offset_ms", 0))
        lipsync_suspicious = (
            analysis.lipsync.state in (DimensionState.INCONSISTENT, DimensionState.SUSPICIOUS)
            or offset_ms > 100
        )

        enf_continuous = analysis.enf.state in (DimensionState.CONSISTENT, DimensionState.VERIFIED)
        enf_discontinuity = enf_raw.get("discontinuity_detected", False)

        if lipsync_suspicious and enf_continuous and not enf_discontinuity:
            warnings.append(CorrelationWarning(
                severity="high",
                category="lipsync_enf_conflict",
                dimensions=["lipsync", "enf"],
                finding=f"Audio-visual desync detected ({offset_ms:.0f}ms offset) but ENF signal is continuous - audio may have been replaced with recording from same grid region",
                details={
                    "lipsync_offset_ms": offset_ms,
                    "enf_continuous": True,
                    "implication": "Audio replaced but from same power grid region (same session or location)",
                }
            ))

        return warnings
