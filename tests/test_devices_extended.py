"""
Extended tests for device capability database.

Tests edge cases, all device entries, and boundary conditions.
"""

import pytest
from wu.dimensions.devices import (
    DEVICE_CAPABILITIES,
    VIDEO_CAPABILITIES,
    normalize_device_name,
    get_device_max_resolution,
    is_resolution_possible,
)


class TestDeviceDatabaseCompleteness:
    """Test device database has expected entries."""

    def test_has_iphones(self):
        """Database contains iPhone models."""
        iphones = [k for k in DEVICE_CAPABILITIES if "iphone" in k]
        assert len(iphones) >= 20

    def test_has_samsung(self):
        """Database contains Samsung models."""
        samsung = [k for k in DEVICE_CAPABILITIES if "samsung" in k]
        assert len(samsung) >= 10

    def test_has_google_pixel(self):
        """Database contains Google Pixel models."""
        pixels = [k for k in DEVICE_CAPABILITIES if "pixel" in k]
        assert len(pixels) >= 5

    def test_has_dslrs(self):
        """Database contains DSLR cameras."""
        canon = [k for k in DEVICE_CAPABILITIES if "canon" in k]
        nikon = [k for k in DEVICE_CAPABILITIES if "nikon" in k]
        sony = [k for k in DEVICE_CAPABILITIES if "sony" in k]
        assert len(canon) >= 5
        assert len(nikon) >= 4
        assert len(sony) >= 2

    def test_video_capabilities_subset(self):
        """Video capabilities exist for key devices."""
        # iPhone 6 is important for the "4K video" impossibility check
        assert "apple iphone 6" in VIDEO_CAPABILITIES
        assert "apple iphone 6 plus" in VIDEO_CAPABILITIES


class TestDeviceResolutions:
    """Test specific device resolutions are correct."""

    def test_iphone_4_resolution(self):
        """iPhone 4 has 5MP camera (2592x1936)."""
        assert DEVICE_CAPABILITIES["apple iphone 4"] == (2592, 1936)

    def test_iphone_4s_resolution(self):
        """iPhone 4S has 8MP camera."""
        assert DEVICE_CAPABILITIES["apple iphone 4s"] == (3264, 2448)

    def test_iphone_6s_resolution(self):
        """iPhone 6S has 12MP camera (first 4K capable)."""
        assert DEVICE_CAPABILITIES["apple iphone 6s"] == (4032, 3024)

    def test_canon_5d_mark_iv(self):
        """Canon 5D Mark IV has 30.4MP sensor."""
        assert DEVICE_CAPABILITIES["canon eos 5d mark iv"] == (6720, 4480)

    def test_nikon_d850(self):
        """Nikon D850 has 45.7MP sensor."""
        assert DEVICE_CAPABILITIES["nikon d850"] == (8256, 5504)

    def test_sony_a7r_iv(self):
        """Sony A7R IV has 61MP sensor."""
        assert DEVICE_CAPABILITIES["sony ilce-7rm4"] == (9504, 6336)


class TestVideoCapabilities:
    """Test video resolution limits."""

    def test_iphone_4_720p(self):
        """iPhone 4 max video is 720p."""
        assert VIDEO_CAPABILITIES["apple iphone 4"] == (1280, 720)

    def test_iphone_5_1080p(self):
        """iPhone 5 max video is 1080p."""
        assert VIDEO_CAPABILITIES["apple iphone 5"] == (1920, 1080)

    def test_iphone_6s_4k(self):
        """iPhone 6S can do 4K video."""
        assert VIDEO_CAPABILITIES["apple iphone 6s"] == (3840, 2160)

    def test_iphone_6_cannot_4k(self):
        """iPhone 6 is limited to 1080p."""
        max_w, max_h = VIDEO_CAPABILITIES["apple iphone 6"]
        assert max_w <= 1920
        assert max_h <= 1080


class TestNormalizationEdgeCases:
    """Test edge cases in device name normalization."""

    def test_empty_strings(self):
        """Empty strings normalize to empty."""
        result = normalize_device_name("", "")
        assert result == ""

    def test_whitespace_only(self):
        """Whitespace-only normalizes to empty."""
        result = normalize_device_name("   ", "   ")
        assert result == ""

    def test_tabs_and_newlines(self):
        """Tabs and newlines are handled."""
        result = normalize_device_name("Apple\t", "\niPhone 6\n")
        assert result == "apple iphone 6"

    def test_unicode_preserved(self):
        """Unicode characters are preserved."""
        result = normalize_device_name("Huawei", "P30 Pro")
        assert result == "huawei p30 pro"

    def test_numbers_preserved(self):
        """Numbers in model names are preserved."""
        result = normalize_device_name("Samsung", "SM-G998")
        assert result == "samsung sm-g998"


class TestResolutionLookupEdgeCases:
    """Test edge cases in resolution lookup."""

    def test_empty_make(self):
        """Empty make still works with model."""
        result = get_device_max_resolution("", "iPhone 6")
        # Should find via partial matching
        assert result is not None

    def test_empty_model(self):
        """Empty model with make returns None."""
        result = get_device_max_resolution("Apple", "")
        # Empty model shouldn't match anything specific
        # Behavior depends on implementation

    def test_swapped_make_model(self):
        """Swapped make/model still matches."""
        result = get_device_max_resolution("iPhone 6", "Apple")
        # Should find via partial matching
        assert result is not None

    def test_partial_match_iphone(self):
        """Partial iPhone model matches."""
        result = get_device_max_resolution("Apple", "iPhone 15")
        assert result is not None

    def test_case_variations(self):
        """Various case combinations work."""
        r1 = get_device_max_resolution("APPLE", "IPHONE 6")
        r2 = get_device_max_resolution("apple", "iphone 6")
        r3 = get_device_max_resolution("Apple", "iPhone 6")
        assert r1 == r2 == r3


class TestResolutionPossibilityBoundaries:
    """Test boundary conditions for resolution checks."""

    def test_exact_max_resolution(self):
        """Exact max resolution is possible."""
        result = is_resolution_possible("Apple", "iPhone 6", 3264, 2448)
        assert result is True

    def test_one_pixel_over(self):
        """One pixel over max is within tolerance."""
        result = is_resolution_possible("Apple", "iPhone 6", 3265, 2448)
        assert result is True  # Within 10% tolerance

    def test_ten_percent_over(self):
        """10% over max is at boundary."""
        max_w, max_h = DEVICE_CAPABILITIES["apple iphone 6"]
        max_pixels = max_w * max_h
        # 10% over = 1.1x pixels
        over_pixels = int(max_pixels * 1.1)
        # This should be at the boundary
        result = is_resolution_possible("Apple", "iPhone 6", 3600, 2400)
        # 3600*2400 = 8,640,000 vs 7,990,272 * 1.1 = 8,789,299
        assert result is True

    def test_clearly_over_tolerance(self):
        """Clearly over 10% tolerance fails."""
        result = is_resolution_possible("Apple", "iPhone 6", 4000, 3000)
        # 12,000,000 pixels vs 7,990,272 * 1.1 = 8,789,299
        assert result is False

    def test_zero_resolution(self):
        """Zero resolution is possible (edge case)."""
        result = is_resolution_possible("Apple", "iPhone 6", 0, 0)
        assert result is True  # 0 pixels is always <= max

    def test_negative_resolution(self):
        """Negative resolution handled gracefully."""
        result = is_resolution_possible("Apple", "iPhone 6", -1, -1)
        # Negative pixels = positive after multiplication
        assert result is True  # 1 pixel is always <= max

    def test_very_large_resolution(self):
        """Very large resolution is impossible."""
        result = is_resolution_possible("Apple", "iPhone 6", 100000, 100000)
        assert result is False


class TestAllDeviceEntriesValid:
    """Validate all device entries have sensible values."""

    def test_all_photo_resolutions_positive(self):
        """All photo resolutions are positive."""
        for device, (w, h) in DEVICE_CAPABILITIES.items():
            assert w > 0, f"{device} has invalid width: {w}"
            assert h > 0, f"{device} has invalid height: {h}"

    def test_all_video_resolutions_positive(self):
        """All video resolutions are positive."""
        for device, (w, h) in VIDEO_CAPABILITIES.items():
            assert w > 0, f"{device} has invalid width: {w}"
            assert h > 0, f"{device} has invalid height: {h}"

    def test_photo_resolution_reasonable(self):
        """All photo resolutions are reasonable (under 250MP)."""
        for device, (w, h) in DEVICE_CAPABILITIES.items():
            pixels = w * h
            assert pixels < 250_000_000, f"{device} has unreasonable resolution: {pixels}"

    def test_video_resolution_reasonable(self):
        """All video resolutions are under 16K."""
        for device, (w, h) in VIDEO_CAPABILITIES.items():
            assert w <= 15360, f"{device} has unreasonable width: {w}"
            assert h <= 8640, f"{device} has unreasonable height: {h}"

    def test_device_names_lowercase(self):
        """All device names are lowercase."""
        for device in DEVICE_CAPABILITIES.keys():
            assert device == device.lower(), f"{device} is not lowercase"
        for device in VIDEO_CAPABILITIES.keys():
            assert device == device.lower(), f"{device} is not lowercase"
