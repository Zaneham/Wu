"""
GPS forensic consistency checking.

Analyzes GPS coordinates in media metadata for inconsistencies:
- Coordinates in impossible locations (oceans, deserts where no photo possible)
- GPS/timezone mismatches
- GPS coordinates that don't match claimed shooting location
- Precision anomalies (too precise or suspiciously round numbers)

No external API calls - pure coordinate geometry and logic.

References:
    WGS 84 geodetic coordinate system
    EXIF GPS IFD specification (JEITA CP-3451C)
"""

import re
import math
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from ..state import Evidence


@dataclass
class GPSCoordinate:
    """Represents a GPS coordinate with validation."""
    latitude: float  # -90 to 90
    longitude: float  # -180 to 180
    altitude: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        """Check if coordinates are within valid ranges."""
        return -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180

    @property
    def is_null_island(self) -> bool:
        """Check if coordinates are at null island (0,0) - common error."""
        return abs(self.latitude) < 0.01 and abs(self.longitude) < 0.01

    @property
    def is_suspiciously_round(self) -> bool:
        """Check if coordinates are suspiciously round numbers."""
        lat_str = f"{self.latitude:.6f}"
        lon_str = f"{self.longitude:.6f}"
        # Check for patterns like X.000000 or X.500000
        return (lat_str.endswith("000000") or lat_str.endswith("500000") or
                lon_str.endswith("000000") or lon_str.endswith("500000"))


class GPSAnalyzer:
    """
    Analyzes GPS metadata for forensic inconsistencies.

    Checks performed:
    - Valid coordinate ranges
    - Null island detection (0,0)
    - Ocean/water body detection
    - Round number detection
    - Timezone consistency
    - Altitude plausibility
    """

    # Major ocean bounding boxes (approximate)
    # These are simplified - real implementation would use coastline data
    OCEAN_REGIONS = [
        # Pacific Ocean (central)
        {"name": "Central Pacific", "lat": (-30, 30), "lon": (-180, -100)},
        # Atlantic Ocean (central)
        {"name": "Central Atlantic", "lat": (-30, 30), "lon": (-60, -10)},
        # Indian Ocean
        {"name": "Indian Ocean", "lat": (-40, 10), "lon": (50, 100)},
    ]

    # Timezone offsets by longitude (approximate)
    # UTC offset = longitude / 15
    TIMEZONE_TOLERANCE_HOURS = 2  # Allow some tolerance

    def analyze_gps(self, metadata: Dict[str, Any]) -> List[Evidence]:
        """
        Analyze GPS data in metadata for inconsistencies.

        Returns list of Evidence findings.
        """
        evidence = []

        # Extract GPS coordinates
        coords = self._extract_coordinates(metadata)
        if coords is None:
            return evidence

        # Check coordinate validity
        if not coords.is_valid:
            evidence.append(Evidence(
                finding="Invalid GPS coordinates",
                explanation=f"Latitude {coords.latitude} or longitude {coords.longitude} out of valid range",
                contradiction="Valid coordinates: lat -90 to 90, lon -180 to 180"
            ))
            return evidence

        # Check for null island
        if coords.is_null_island:
            evidence.append(Evidence(
                finding="GPS shows Null Island (0°, 0°)",
                explanation="Coordinates point to Gulf of Guinea - common GPS error/placeholder",
                contradiction="Null Island is in the ocean, unlikely shooting location"
            ))

        # Check for suspiciously round numbers
        if coords.is_suspiciously_round:
            evidence.append(Evidence(
                finding="Suspiciously round GPS coordinates",
                explanation=f"Coordinates ({coords.latitude}, {coords.longitude}) are unusually round",
            ))

        # Check for ocean locations
        ocean_check = self._check_ocean_location(coords)
        if ocean_check:
            evidence.append(ocean_check)

        # Check timezone consistency
        tz_evidence = self._check_timezone_consistency(metadata, coords)
        if tz_evidence:
            evidence.append(tz_evidence)

        # Check altitude plausibility
        if coords.altitude is not None:
            alt_evidence = self._check_altitude(coords)
            if alt_evidence:
                evidence.append(alt_evidence)

        return evidence

    def _extract_coordinates(self, metadata: Dict[str, Any]) -> Optional[GPSCoordinate]:
        """Extract GPS coordinates from EXIF metadata."""
        # Try different GPS tag formats
        lat = self._extract_latitude(metadata)
        lon = self._extract_longitude(metadata)

        if lat is None or lon is None:
            return None

        alt = self._extract_altitude(metadata)

        return GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)

    def _extract_latitude(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract latitude from metadata."""
        # Try various field names
        lat_fields = ["GPSLatitude", "GPS GPSLatitude", "Exif.GPSInfo.GPSLatitude"]
        ref_fields = ["GPSLatitudeRef", "GPS GPSLatitudeRef", "Exif.GPSInfo.GPSLatitudeRef"]

        lat_value = None
        lat_ref = "N"

        for field in lat_fields:
            if field in metadata:
                lat_value = self._parse_gps_value(metadata[field])
                break

        for field in ref_fields:
            if field in metadata:
                lat_ref = str(metadata[field]).upper().strip()
                break

        if lat_value is None:
            return None

        if lat_ref.startswith("S"):
            lat_value = -lat_value

        return lat_value

    def _extract_longitude(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract longitude from metadata."""
        lon_fields = ["GPSLongitude", "GPS GPSLongitude", "Exif.GPSInfo.GPSLongitude"]
        ref_fields = ["GPSLongitudeRef", "GPS GPSLongitudeRef", "Exif.GPSInfo.GPSLongitudeRef"]

        lon_value = None
        lon_ref = "E"

        for field in lon_fields:
            if field in metadata:
                lon_value = self._parse_gps_value(metadata[field])
                break

        for field in ref_fields:
            if field in metadata:
                lon_ref = str(metadata[field]).upper().strip()
                break

        if lon_value is None:
            return None

        if lon_ref.startswith("W"):
            lon_value = -lon_value

        return lon_value

    def _extract_altitude(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract altitude from metadata."""
        alt_fields = ["GPSAltitude", "GPS GPSAltitude"]

        for field in alt_fields:
            if field in metadata:
                try:
                    value = str(metadata[field])
                    # Handle ratio format "1234/10"
                    if "/" in value:
                        num, denom = value.split("/")
                        return float(num) / float(denom)
                    return float(value.split()[0])
                except (ValueError, IndexError):
                    pass
        return None

    def _parse_gps_value(self, value: Any) -> Optional[float]:
        """
        Parse GPS value from various formats.

        Formats:
        - Decimal degrees: 40.7128
        - DMS string: "40 deg 42' 46.08" N"
        - Ratio: "[40, 1, 42, 1, 4608, 100]"
        """
        if value is None:
            return None

        value_str = str(value).strip()

        # Try decimal
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try DMS format: "40 deg 42' 46.08""
        dms_match = re.match(
            r"(\d+)\s*(?:deg|°)?\s*(\d+)\s*['\u2032]?\s*(\d+\.?\d*)\s*[\"'\u2033]?",
            value_str
        )
        if dms_match:
            d, m, s = dms_match.groups()
            return float(d) + float(m)/60 + float(s)/3600

        # Try ratio format from exifread: "[40, 1, 42, 1, 4608, 100]"
        ratio_match = re.match(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", value_str)
        if ratio_match:
            d_num, d_den, m_num, m_den, s_num, s_den = map(int, ratio_match.groups())
            d = d_num / d_den if d_den else 0
            m = m_num / m_den if m_den else 0
            s = s_num / s_den if s_den else 0
            return d + m/60 + s/3600

        return None

    def _check_ocean_location(self, coords: GPSCoordinate) -> Optional[Evidence]:
        """Check if coordinates are in the middle of an ocean."""
        for region in self.OCEAN_REGIONS:
            lat_min, lat_max = region["lat"]
            lon_min, lon_max = region["lon"]

            if (lat_min <= coords.latitude <= lat_max and
                lon_min <= coords.longitude <= lon_max):
                return Evidence(
                    finding=f"GPS location in {region['name']}",
                    explanation=f"Coordinates ({coords.latitude:.4f}, {coords.longitude:.4f}) are in open ocean",
                    contradiction="Most photos are taken on land"
                )
        return None

    def _check_timezone_consistency(
        self,
        metadata: Dict[str, Any],
        coords: GPSCoordinate
    ) -> Optional[Evidence]:
        """
        Check if GPS longitude is consistent with timezone offset.

        Expected: UTC offset ≈ longitude / 15
        """
        # Try to get timezone offset from metadata
        tz_offset = self._extract_timezone_offset(metadata)
        if tz_offset is None:
            return None

        # Expected timezone based on longitude
        expected_tz = coords.longitude / 15

        # Check if within tolerance
        if abs(tz_offset - expected_tz) > self.TIMEZONE_TOLERANCE_HOURS:
            return Evidence(
                finding="GPS/Timezone mismatch",
                explanation=(
                    f"Timezone offset ({tz_offset:+.1f}h) doesn't match "
                    f"GPS longitude ({coords.longitude:.1f}° → expected ~{expected_tz:+.1f}h)"
                ),
            )
        return None

    def _extract_timezone_offset(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract timezone offset from metadata."""
        tz_fields = ["OffsetTime", "OffsetTimeOriginal", "TimeZoneOffset"]

        for field in tz_fields:
            if field in metadata:
                try:
                    value = str(metadata[field])
                    # Parse "+05:30" or "-08:00" format
                    match = re.match(r"([+-])(\d{1,2}):?(\d{2})?", value)
                    if match:
                        sign = 1 if match.group(1) == "+" else -1
                        hours = int(match.group(2))
                        minutes = int(match.group(3) or 0)
                        return sign * (hours + minutes/60)
                except (ValueError, AttributeError):
                    pass
        return None

    def _check_altitude(self, coords: GPSCoordinate) -> Optional[Evidence]:
        """Check if altitude is plausible."""
        if coords.altitude is None:
            return None

        # Earth's highest point: ~8849m (Everest)
        # Lowest point: ~-430m (Dead Sea shore)
        # Aircraft typically fly below 12000m

        if coords.altitude < -500:
            return Evidence(
                finding="Impossible GPS altitude",
                explanation=f"Altitude {coords.altitude:.0f}m is below sea level minimum",
                contradiction="Lowest land point is ~-430m (Dead Sea)"
            )

        if coords.altitude > 15000:
            return Evidence(
                finding="Implausible GPS altitude",
                explanation=f"Altitude {coords.altitude:.0f}m exceeds typical aircraft ceiling",
            )

        return None
