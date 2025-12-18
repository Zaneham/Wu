"""
Tests for GPS forensic consistency analyzer.

Tests coordinate validation, null island detection, ocean detection,
and timezone consistency checking.
"""

import pytest
from wu.dimensions.gps import GPSAnalyzer, GPSCoordinate


class TestGPSCoordinate:
    """Test GPSCoordinate dataclass."""

    def test_valid_coordinates(self):
        """Valid coordinates are recognized."""
        coord = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
        assert coord.is_valid is True

    def test_invalid_latitude(self):
        """Invalid latitude is detected."""
        coord = GPSCoordinate(latitude=91.0, longitude=0.0)
        assert coord.is_valid is False

    def test_invalid_longitude(self):
        """Invalid longitude is detected."""
        coord = GPSCoordinate(latitude=0.0, longitude=181.0)
        assert coord.is_valid is False

    def test_null_island(self):
        """Null island (0,0) is detected."""
        coord = GPSCoordinate(latitude=0.0, longitude=0.0)
        assert coord.is_null_island is True

    def test_near_null_island(self):
        """Near null island is detected."""
        coord = GPSCoordinate(latitude=0.005, longitude=-0.005)
        assert coord.is_null_island is True

    def test_not_null_island(self):
        """Normal location is not null island."""
        coord = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
        assert coord.is_null_island is False

    def test_round_numbers_detected(self):
        """Suspiciously round numbers are detected."""
        coord = GPSCoordinate(latitude=40.0, longitude=-74.0)
        assert coord.is_suspiciously_round is True

    def test_half_numbers_detected(self):
        """Half-degree values are detected."""
        coord = GPSCoordinate(latitude=40.5, longitude=-74.5)
        assert coord.is_suspiciously_round is True

    def test_normal_precision_not_round(self):
        """Normal precision is not flagged."""
        coord = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
        assert coord.is_suspiciously_round is False


class TestGPSAnalyzer:
    """Test GPSAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_no_gps_data(self, analyzer):
        """No GPS data returns empty evidence."""
        metadata = {"Make": "Apple", "Model": "iPhone"}
        evidence = analyzer.analyze_gps(metadata)
        assert len(evidence) == 0

    def test_null_island_detected(self, analyzer):
        """Null island is flagged."""
        metadata = {
            "GPSLatitude": "0.0",
            "GPSLongitude": "0.0",
        }
        evidence = analyzer.analyze_gps(metadata)
        assert len(evidence) >= 1
        assert any("Null Island" in e.finding for e in evidence)

    def test_invalid_coordinates_flagged(self, analyzer):
        """Invalid coordinates are flagged."""
        metadata = {
            "GPSLatitude": "95.0",  # Invalid
            "GPSLatitudeRef": "N",
            "GPSLongitude": "0.0",
            "GPSLongitudeRef": "E",
        }
        evidence = analyzer.analyze_gps(metadata)
        assert len(evidence) >= 1
        assert any("Invalid" in e.finding for e in evidence)


class TestGPSParsing:
    """Test GPS value parsing."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_parse_decimal(self, analyzer):
        """Decimal degrees are parsed."""
        result = analyzer._parse_gps_value("40.7128")
        assert abs(result - 40.7128) < 0.0001

    def test_parse_dms_string(self, analyzer):
        """DMS format is parsed."""
        result = analyzer._parse_gps_value("40 deg 42' 46.08\"")
        assert result is not None
        assert abs(result - 40.7128) < 0.001

    def test_parse_ratio_format(self, analyzer):
        """Exifread ratio format is parsed."""
        result = analyzer._parse_gps_value("[40, 1, 42, 1, 4608, 100]")
        assert result is not None
        assert abs(result - 40.7128) < 0.001

    def test_parse_invalid(self, analyzer):
        """Invalid format returns None."""
        result = analyzer._parse_gps_value("not a coordinate")
        assert result is None


class TestOceanDetection:
    """Test ocean location detection."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_pacific_ocean_detected(self, analyzer):
        """Central Pacific location is flagged."""
        coord = GPSCoordinate(latitude=0.0, longitude=-150.0)
        evidence = analyzer._check_ocean_location(coord)
        assert evidence is not None
        assert "Pacific" in evidence.finding

    def test_atlantic_ocean_detected(self, analyzer):
        """Central Atlantic location is flagged."""
        coord = GPSCoordinate(latitude=0.0, longitude=-30.0)
        evidence = analyzer._check_ocean_location(coord)
        assert evidence is not None
        assert "Atlantic" in evidence.finding

    def test_land_location_not_flagged(self, analyzer):
        """Land location is not flagged."""
        # New York City
        coord = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
        evidence = analyzer._check_ocean_location(coord)
        assert evidence is None


class TestAltitudeCheck:
    """Test altitude plausibility checking."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_valid_altitude(self, analyzer):
        """Valid altitude is not flagged."""
        coord = GPSCoordinate(latitude=40.0, longitude=-74.0, altitude=100.0)
        evidence = analyzer._check_altitude(coord)
        assert evidence is None

    def test_impossible_low_altitude(self, analyzer):
        """Impossible low altitude is flagged."""
        coord = GPSCoordinate(latitude=40.0, longitude=-74.0, altitude=-600.0)
        evidence = analyzer._check_altitude(coord)
        assert evidence is not None
        assert "Impossible" in evidence.finding

    def test_implausible_high_altitude(self, analyzer):
        """Implausible high altitude is flagged."""
        coord = GPSCoordinate(latitude=40.0, longitude=-74.0, altitude=20000.0)
        evidence = analyzer._check_altitude(coord)
        assert evidence is not None
        assert "Implausible" in evidence.finding

    def test_mount_everest_ok(self, analyzer):
        """Mount Everest altitude is acceptable."""
        coord = GPSCoordinate(latitude=27.9881, longitude=86.9250, altitude=8849.0)
        evidence = analyzer._check_altitude(coord)
        assert evidence is None

    def test_no_altitude(self, analyzer):
        """No altitude returns no evidence."""
        coord = GPSCoordinate(latitude=40.0, longitude=-74.0, altitude=None)
        evidence = analyzer._check_altitude(coord)
        assert evidence is None


class TestLatLonExtraction:
    """Test latitude/longitude extraction from metadata."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_extract_latitude_north(self, analyzer):
        """North latitude is positive."""
        metadata = {
            "GPSLatitude": "40.7128",
            "GPSLatitudeRef": "N"
        }
        lat = analyzer._extract_latitude(metadata)
        assert lat is not None
        assert lat > 0

    def test_extract_latitude_south(self, analyzer):
        """South latitude is negative."""
        metadata = {
            "GPSLatitude": "40.7128",
            "GPSLatitudeRef": "S"
        }
        lat = analyzer._extract_latitude(metadata)
        assert lat is not None
        assert lat < 0

    def test_extract_longitude_west(self, analyzer):
        """West longitude is negative."""
        metadata = {
            "GPSLongitude": "74.006",
            "GPSLongitudeRef": "W"
        }
        lon = analyzer._extract_longitude(metadata)
        assert lon is not None
        assert lon < 0

    def test_extract_longitude_east(self, analyzer):
        """East longitude is positive."""
        metadata = {
            "GPSLongitude": "74.006",
            "GPSLongitudeRef": "E"
        }
        lon = analyzer._extract_longitude(metadata)
        assert lon is not None
        assert lon > 0


class TestTimezoneConsistency:
    """Test timezone/GPS consistency checking."""

    @pytest.fixture
    def analyzer(self):
        return GPSAnalyzer()

    def test_consistent_timezone(self, analyzer):
        """Consistent timezone is not flagged."""
        metadata = {
            "GPSLongitude": "-74.0",  # NYC
            "GPSLongitudeRef": "W",
            "OffsetTime": "-05:00"  # EST
        }
        coord = GPSCoordinate(latitude=40.7, longitude=-74.0)
        evidence = analyzer._check_timezone_consistency(metadata, coord)
        assert evidence is None

    def test_inconsistent_timezone(self, analyzer):
        """Inconsistent timezone is flagged."""
        metadata = {
            "GPSLongitude": "-74.0",  # NYC
            "GPSLongitudeRef": "W",
            "OffsetTime": "+09:00"  # Tokyo timezone
        }
        coord = GPSCoordinate(latitude=40.7, longitude=-74.0)
        evidence = analyzer._check_timezone_consistency(metadata, coord)
        assert evidence is not None
        assert "mismatch" in evidence.finding.lower()


class TestIntegrationWithMetadata:
    """Test GPS analysis integration with metadata analyzer."""

    def test_metadata_includes_gps_check(self):
        """Metadata analyzer includes GPS checking."""
        from wu.dimensions.metadata import MetadataAnalyzer

        analyzer = MetadataAnalyzer()
        # Verify method exists
        assert hasattr(analyzer, '_check_gps')

    def test_gps_evidence_in_results(self):
        """GPS evidence appears in metadata results."""
        from wu.dimensions.metadata import MetadataAnalyzer

        analyzer = MetadataAnalyzer()

        # Create metadata with null island GPS
        # This would require mocking _extract_metadata
        # For now, just verify the method chain exists
        assert callable(analyzer._check_gps)
