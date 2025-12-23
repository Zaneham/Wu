"""
Metadata forensic analysis.

Checks for inconsistencies in file metadata that indicate manipulation:
- Device claimed vs device capabilities (e.g., iPhone 6 claiming 4K)
- GPS consistency with claimed location
- Timestamp plausibility
- Software signatures indicating editing
- EXIF data integrity

No ML required - just logic and a device capability database.

References:
    JEITA CP-3451C (Exif 2.32 specification)
    Kee, E. & Farid, H. (2011). A Perceptual Metric for Photo Retouching.
        ACM Transactions on Graphics, 30(6).
"""

import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

try:
    import exifread
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence
from .devices import DEVICE_CAPABILITIES, get_device_max_resolution
from .gps import GPSAnalyzer


class MetadataAnalyzer:
    """
    Analyzes file metadata for forensic inconsistencies.

    Phase 0 implementation: catches obvious fakes like
    "iPhone 6 with 4K video" without any ML.
    """

    # Known editing software signatures
    EDITING_SOFTWARE = [
        "adobe photoshop",
        "adobe premiere",
        "final cut",
        "davinci resolve",
        "after effects",
        "lightroom",
        "gimp",
        "paint.net",
        "pixelmator",
        "affinity photo",
        "capture one",
        "luminar",
        "topaz",
        "ffmpeg",
        "handbrake",
        "avidemux",
    ]

    # AI generation signatures
    AI_SIGNATURES = [
        "dall-e",
        "midjourney",
        "stable diffusion",
        "comfyui",
        "automatic1111",
        "sora",
        "runway",
        "pika",
        "kling",
        "flux",
        "firefly",
    ]

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze file metadata for inconsistencies.

        Returns a DimensionResult with state indicating
        consistency/inconsistency and supporting evidence.
        """
        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="metadata",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        # Extract metadata
        metadata = self._extract_metadata(file_path)
        if not metadata:
            return DimensionResult(
                dimension="metadata",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="No metadata found",
                    explanation="File contains no extractable EXIF data"
                )],
                methodology="EXIF extraction via exifread/PIL"
            )

        # Run checks
        evidence = []
        issues_found = False
        suspicious = False

        # Check 1: Device vs resolution consistency
        device_result = self._check_device_resolution(metadata)
        if device_result:
            evidence.append(device_result)
            if "impossible" in device_result.explanation.lower():
                issues_found = True
            else:
                suspicious = True

        # Check 2: Editing software detection
        software_result = self._check_software_signatures(metadata)
        if software_result:
            evidence.append(software_result)
            suspicious = True

        # Check 3: AI generation signatures
        ai_result = self._check_ai_signatures(metadata)
        if ai_result:
            evidence.append(ai_result)
            issues_found = True

        # Check 4: Timestamp consistency
        time_result = self._check_timestamps(metadata)
        if time_result:
            evidence.append(time_result)
            if "inconsistent" in time_result.explanation.lower():
                issues_found = True
            else:
                suspicious = True

        # Check 5: Stripped metadata (noted but not suspicious alone)
        # Many legitimate sources strip metadata (social media, privacy tools, screenshots)
        if self._is_metadata_stripped(metadata):
            evidence.append(Evidence(
                finding="Metadata appears stripped",
                explanation="File has minimal metadata. Common for screenshots, social media, or privacy-conscious sharing."
            ))
            # stripped metadata alone is not suspicious

        # Check 6: GPS consistency
        gps_results = self._check_gps(metadata)
        for gps_evidence in gps_results:
            evidence.append(gps_evidence)
            if "impossible" in gps_evidence.explanation.lower():
                issues_found = True
            elif "null island" in gps_evidence.finding.lower():
                issues_found = True
            else:
                suspicious = True

        # Determine state
        if issues_found:
            state = DimensionState.INCONSISTENT
            confidence = Confidence.HIGH
        elif suspicious:
            state = DimensionState.SUSPICIOUS
            confidence = Confidence.MEDIUM
        elif evidence:
            state = DimensionState.CONSISTENT
            confidence = Confidence.HIGH
        else:
            state = DimensionState.CONSISTENT
            confidence = Confidence.MEDIUM
            evidence.append(Evidence(
                finding="Metadata appears consistent",
                explanation="No inconsistencies detected in available metadata"
            ))

        return DimensionResult(
            dimension="metadata",
            state=state,
            confidence=confidence,
            evidence=evidence,
            methodology="EXIF analysis with device capability verification"
        )

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF and other metadata from file."""
        metadata = {}

        # Try exifread first (more complete for images)
        if HAS_EXIF:
            try:
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    for tag, value in tags.items():
                        # Clean up tag names
                        clean_tag = tag.replace("EXIF ", "").replace("Image ", "")
                        metadata[clean_tag] = str(value)
            except Exception:
                pass

        # Also try PIL for additional data
        if HAS_PIL:
            try:
                with Image.open(file_path) as img:
                    metadata["_image_width"] = img.width
                    metadata["_image_height"] = img.height
                    metadata["_image_format"] = img.format

                    exif_data = img.getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore')
                                except:
                                    value = str(value)
                            metadata[str(tag)] = str(value)
            except Exception:
                pass

        return metadata

    def _check_device_resolution(self, metadata: Dict[str, Any]) -> Optional[Evidence]:
        """Check if claimed device can produce the file's resolution."""
        # Get device info
        make = metadata.get("Make", "").strip()
        model = metadata.get("Model", "").strip()

        if not make and not model:
            return None

        device_name = f"{make} {model}".strip()

        # Get actual resolution
        width = metadata.get("_image_width") or metadata.get("ExifImageWidth")
        height = metadata.get("_image_height") or metadata.get("ExifImageHeight")

        if not width or not height:
            return None

        try:
            width = int(str(width).split()[0])
            height = int(str(height).split()[0])
        except (ValueError, IndexError):
            return None

        # Check against device capabilities
        max_res = get_device_max_resolution(make, model)
        if max_res is None:
            return None

        max_width, max_height = max_res
        actual_pixels = width * height
        max_pixels = max_width * max_height

        if actual_pixels > max_pixels * 1.1:  # 10% tolerance for rounding
            return Evidence(
                finding=f"Device claims {device_name}",
                contradiction=f"Resolution is {width}x{height} ({self._resolution_name(width, height)})",
                explanation=f"This is impossible - {device_name} maximum is {max_width}x{max_height}",
                citation=f"{make} {model} technical specifications"
            )

        return None

    def _check_software_signatures(self, metadata: Dict[str, Any]) -> Optional[Evidence]:
        """Check for editing software signatures."""
        software_fields = [
            metadata.get("Software", ""),
            metadata.get("ProcessingSoftware", ""),
            metadata.get("CreatorTool", ""),
            metadata.get("HistorySoftwareAgent", ""),
        ]

        software_str = " ".join(software_fields).lower()

        for sig in self.EDITING_SOFTWARE:
            if sig in software_str:
                return Evidence(
                    finding=f"Editing software detected: {sig.title()}",
                    explanation="File has been processed with editing software"
                )

        return None

    def _check_ai_signatures(self, metadata: Dict[str, Any]) -> Optional[Evidence]:
        """Check for AI generation signatures."""
        # Only check text-based metadata fields, not binary blobs
        # Binary fields like ImageSourceData (37724) can contain false positives
        text_fields = [
            "Software", "ProcessingSoftware", "CreatorTool", "Creator",
            "Artist", "Copyright", "ImageDescription", "UserComment",
            "XPComment", "XPAuthor", "XPKeywords", "XPSubject", "XPTitle",
            "Make", "Model", "HostComputer", "DocumentName",
            "XMLPacket", "parameters", "Comment", "Description",
        ]

        # Collect text from relevant fields only (skip large binary blobs)
        text_parts = []
        for key, value in metadata.items():
            val_str = str(value)
            # Skip binary blobs (large values or those with many non-printable chars)
            if len(val_str) > 10000:
                continue
            # Check if it's a known text field or a small value
            if key in text_fields or len(val_str) < 500:
                # Skip values with many non-printable characters (binary data)
                printable_ratio = sum(1 for c in val_str if c.isprintable() or c.isspace()) / max(len(val_str), 1)
                if printable_ratio > 0.9:
                    text_parts.append(val_str.lower())

        all_values = " ".join(text_parts)

        for sig in self.AI_SIGNATURES:
            if sig in all_values:
                return Evidence(
                    finding=f"AI generation signature detected: {sig.upper()}",
                    explanation="Metadata indicates file was generated by AI",
                    contradiction="File presented as authentic capture"
                )

        # Check for ComfyUI/A1111 specific patterns
        if "parameters" in metadata:
            params = metadata["parameters"].lower()
            if "steps:" in params or "sampler:" in params or "cfg scale:" in params:
                return Evidence(
                    finding="Stable Diffusion generation parameters found",
                    explanation="File contains AI image generation parameters in metadata"
                )

        return None

    def _check_timestamps(self, metadata: Dict[str, Any]) -> Optional[Evidence]:
        """Check timestamp consistency."""
        timestamps = {}

        # Collect all timestamps
        time_fields = [
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
            "CreateDate",
            "ModifyDate",
        ]

        for field in time_fields:
            if field in metadata:
                try:
                    ts = self._parse_exif_datetime(metadata[field])
                    if ts:
                        timestamps[field] = ts
                except:
                    pass

        if not timestamps:
            return None

        # Check for future dates first (works with any number of timestamps)
        now = datetime.now()
        for field, ts in timestamps.items():
            if ts > now:
                return Evidence(
                    finding="Future timestamp detected",
                    explanation=f"{field} is set to a future date: {ts}",
                    contradiction="Timestamps cannot be in the future"
                )

        # Need at least 2 timestamps for consistency checks
        if len(timestamps) < 2:
            return None

        # Check for impossible ordering
        original = timestamps.get("DateTimeOriginal")
        digitized = timestamps.get("DateTimeDigitized")
        modified = timestamps.get("DateTime") or timestamps.get("ModifyDate")

        if original and modified and modified < original:
            return Evidence(
                finding="Timestamp inconsistency",
                explanation="Modification date is before original capture date - inconsistent",
                contradiction=f"Original: {original}, Modified: {modified}"
            )

        return None

    def _is_metadata_stripped(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata appears intentionally stripped."""
        # Very few fields suggests stripping
        exif_fields = [k for k in metadata.keys() if not k.startswith("_")]
        return len(exif_fields) < 5

    def _check_gps(self, metadata: Dict[str, Any]) -> List[Evidence]:
        """Check GPS coordinates for inconsistencies."""
        gps_analyzer = GPSAnalyzer()
        return gps_analyzer.analyze_gps(metadata)

    def _parse_exif_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse EXIF datetime format."""
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(dt_str).strip(), fmt)
            except ValueError:
                continue
        return None

    def _resolution_name(self, width: int, height: int) -> str:
        """Get common name for resolution."""
        pixels = width * height
        if pixels >= 33000000:  # ~8K
            return "8K"
        elif pixels >= 8000000:  # ~4K
            return "4K UHD"
        elif pixels >= 2000000:  # ~1080p
            return "1080p"
        elif pixels >= 900000:  # ~720p
            return "720p"
        else:
            return f"{width}x{height}"
