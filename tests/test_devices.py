"""
Tests for device capability database.

Verifies the forensic logic that catches impossibilities like
"iPhone 6 claiming 4K resolution."
"""

import pytest
from wu.dimensions.devices import (
    DEVICE_CAPABILITIES,
    VIDEO_CAPABILITIES,
    normalize_device_name,
    get_device_max_resolution,
    is_resolution_possible,
)


class TestDeviceDatabase:
    """Test device capability database."""

    def test_iphone_6_max_photo_resolution(self):
        """iPhone 6 has 8MP camera - max 3264x2448."""
        assert DEVICE_CAPABILITIES["apple iphone 6"] == (3264, 2448)

    def test_iphone_6_cannot_do_4k_photo(self):
        """iPhone 6 cannot produce 4K photos."""
        max_w, max_h = DEVICE_CAPABILITIES["apple iphone 6"]
        max_pixels = max_w * max_h
        four_k_pixels = 3840 * 2160
        assert max_pixels < four_k_pixels

    def test_iphone_6_max_video_resolution(self):
        """iPhone 6 video is limited to 1080p."""
        assert VIDEO_CAPABILITIES["apple iphone 6"] == (1920, 1080)

    def test_iphone_15_pro_max_resolution(self):
        """iPhone 15 Pro Max has 48MP camera."""
        assert DEVICE_CAPABILITIES["apple iphone 15 pro max"] == (8064, 6048)

    def test_samsung_s24_ultra_200mp(self):
        """Samsung S24 Ultra has 200MP camera."""
        assert DEVICE_CAPABILITIES["samsung sm-s928"] == (12000, 9000)


class TestDeviceNormalization:
    """Test device name normalization."""

    def test_normalize_basic(self):
        """Basic normalization combines make and model."""
        assert normalize_device_name("Apple", "iPhone 6") == "apple iphone 6"

    def test_normalize_extra_whitespace(self):
        """Extra whitespace is collapsed."""
        assert normalize_device_name("  Apple  ", "  iPhone   6  ") == "apple iphone 6"

    def test_normalize_case_insensitive(self):
        """Normalization is case-insensitive."""
        assert normalize_device_name("APPLE", "IPHONE 6") == "apple iphone 6"


class TestResolutionLookup:
    """Test device resolution lookup."""

    def test_lookup_exact_match(self):
        """Exact match returns correct resolution."""
        result = get_device_max_resolution("Apple", "iPhone 6")
        assert result == (3264, 2448)

    def test_lookup_partial_match(self):
        """Partial model match works."""
        result = get_device_max_resolution("Apple", "iPhone 6 Plus")
        assert result == (3264, 2448)

    def test_lookup_unknown_device(self):
        """Unknown device returns None."""
        result = get_device_max_resolution("Obscure", "ZZZ9999")
        assert result is None

    def test_lookup_video_mode(self):
        """Video mode returns video capabilities."""
        result = get_device_max_resolution("Apple", "iPhone 6", media_type="video")
        assert result == (1920, 1080)


class TestResolutionPossibility:
    """Test resolution possibility checker."""

    def test_iphone_6_possible_resolution(self):
        """Resolution within limits is possible."""
        result = is_resolution_possible("Apple", "iPhone 6", 3264, 2448)
        assert result is True

    def test_iphone_6_lower_resolution_possible(self):
        """Lower resolution is always possible."""
        result = is_resolution_possible("Apple", "iPhone 6", 1920, 1080)
        assert result is True

    def test_iphone_6_4k_video_impossible(self):
        """4K VIDEO from iPhone 6 is impossible (max is 1080p)."""
        # For video, iPhone 6 max is 1920x1080
        result = is_resolution_possible("Apple", "iPhone 6", 3840, 2160, media_type="video")
        assert result is False

    def test_iphone_6_8k_impossible(self):
        """8K from iPhone 6 is definitely impossible."""
        result = is_resolution_possible("Apple", "iPhone 6", 7680, 4320)
        assert result is False

    def test_unknown_device_returns_none(self):
        """Unknown device returns None (can't determine)."""
        result = is_resolution_possible("Obscure", "ZZZ9999", 3840, 2160)
        assert result is None

    def test_iphone_15_pro_4k_possible(self):
        """iPhone 15 Pro can do 4K photos."""
        result = is_resolution_possible("Apple", "iPhone 15 Pro Max", 3840, 2160)
        assert result is True

    def test_video_resolution_check(self):
        """Video resolution check uses video capabilities."""
        # iPhone 6 can't do 4K video
        result = is_resolution_possible(
            "Apple", "iPhone 6", 3840, 2160, media_type="video"
        )
        assert result is False

        # iPhone 6 can do 1080p video
        result = is_resolution_possible(
            "Apple", "iPhone 6", 1920, 1080, media_type="video"
        )
        assert result is True
