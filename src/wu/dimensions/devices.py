"""
Device capability database for forensic verification.

Maps device make/model to maximum photo/video capabilities.
Used to catch impossibilities like "iPhone 6 claiming 4K video."

Data sourced from manufacturer specifications.
"""

from typing import Optional, Tuple, Dict

# Device capabilities: (max_photo_width, max_photo_height)
# Video capabilities would be separate but photos are primary for now

DEVICE_CAPABILITIES: Dict[str, Tuple[int, int]] = {
    # Apple iPhones - Photo resolution
    "apple iphone 16 pro max": (8064, 6048),  # 48MP main
    "apple iphone 16 pro": (8064, 6048),
    "apple iphone 16 plus": (8064, 6048),  # 48MP Fusion
    "apple iphone 16": (8064, 6048),        # 48MP Fusion
    "apple iphone 15 pro max": (8064, 6048),  # 48MP
    "apple iphone 15 pro": (8064, 6048),
    "apple iphone 15 plus": (8064, 6048),
    "apple iphone 15": (8064, 6048),
    "apple iphone 14 pro max": (8064, 6048),
    "apple iphone 14 pro": (8064, 6048),
    "apple iphone 14 plus": (4032, 3024),
    "apple iphone 14": (4032, 3024),
    "apple iphone 13 pro max": (4032, 3024),
    "apple iphone 13 pro": (4032, 3024),
    "apple iphone 13": (4032, 3024),
    "apple iphone 13 mini": (4032, 3024),
    "apple iphone 12 pro max": (4032, 3024),
    "apple iphone 12 pro": (4032, 3024),
    "apple iphone 12": (4032, 3024),
    "apple iphone 12 mini": (4032, 3024),
    "apple iphone 11 pro max": (4032, 3024),
    "apple iphone 11 pro": (4032, 3024),
    "apple iphone 11": (4032, 3024),
    "apple iphone xr": (4032, 3024),
    "apple iphone xs max": (4032, 3024),
    "apple iphone xs": (4032, 3024),
    "apple iphone x": (4032, 3024),
    "apple iphone 8 plus": (4032, 3024),
    "apple iphone 8": (4032, 3024),
    "apple iphone 7 plus": (4032, 3024),
    "apple iphone 7": (4032, 3024),
    "apple iphone 6s plus": (4032, 3024),
    "apple iphone 6s": (4032, 3024),
    "apple iphone 6 plus": (3264, 2448),  # 8MP - CANNOT do 4K photo
    "apple iphone 6": (3264, 2448),        # 8MP - CANNOT do 4K photo
    "apple iphone 5s": (3264, 2448),
    "apple iphone 5c": (3264, 2448),
    "apple iphone 5": (3264, 2448),
    "apple iphone 4s": (3264, 2448),
    "apple iphone 4": (2592, 1936),

    # Samsung Galaxy S series
    "samsung sm-s938": (12000, 9000),  # S25 Ultra 200MP
    "samsung sm-s931": (4000, 3000),   # S25+
    "samsung sm-s928": (12000, 9000),  # S24 Ultra 200MP
    "samsung sm-s926": (4000, 3000),   # S24+
    "samsung sm-s921": (4000, 3000),   # S24
    "samsung sm-s918": (12000, 9000),  # S23 Ultra
    "samsung sm-s916": (4000, 3000),   # S23+
    "samsung sm-s911": (4000, 3000),   # S23
    "samsung sm-s908": (12000, 9000),  # S22 Ultra
    "samsung sm-s906": (4000, 3000),   # S22+
    "samsung sm-s901": (4000, 3000),   # S22
    "samsung sm-g998": (4000, 3000),   # S21 Ultra
    "samsung sm-g996": (4000, 3000),   # S21+
    "samsung sm-g991": (4000, 3000),   # S21
    "samsung sm-g988": (4000, 3000),   # S20 Ultra
    "samsung sm-g986": (4000, 3000),   # S20+
    "samsung sm-g981": (4000, 3000),   # S20
    "samsung sm-g975": (4000, 3000),   # S10+
    "samsung sm-g973": (4000, 3000),   # S10
    "samsung sm-g970": (4000, 3000),   # S10e
    "samsung sm-g965": (4032, 3024),   # S9+
    "samsung sm-g960": (4032, 3024),   # S9
    "samsung sm-g955": (4032, 3024),   # S8+
    "samsung sm-g950": (4032, 3024),   # S8
    "samsung sm-g935": (4032, 3024),   # S7 Edge
    "samsung sm-g930": (4032, 3024),   # S7
    "samsung sm-g925": (5312, 2988),   # S6 Edge
    "samsung sm-g920": (5312, 2988),   # S6

    # Samsung Galaxy A series (popular mid-range)
    "samsung sm-a556": (4000, 3000),   # A55
    "samsung sm-a546": (4000, 3000),   # A54
    "samsung sm-a536": (4000, 3000),   # A53
    "samsung sm-a526": (4000, 3000),   # A52
    "samsung sm-a336": (4000, 3000),   # A33

    # Samsung Galaxy Z Fold/Flip
    "samsung sm-f956": (4000, 3000),   # Z Fold 6
    "samsung sm-f946": (4000, 3000),   # Z Fold 5
    "samsung sm-f936": (4000, 3000),   # Z Fold 4
    "samsung sm-f741": (4000, 3000),   # Z Flip 6
    "samsung sm-f731": (4000, 3000),   # Z Flip 5

    # Google Pixel
    "google pixel 9 pro xl": (8160, 6120),  # 50MP
    "google pixel 9 pro": (8160, 6120),
    "google pixel 9 pro fold": (8160, 6120),
    "google pixel 9": (8160, 6120),
    "google pixel 8 pro": (8160, 6120),  # 50MP
    "google pixel 8": (8160, 6120),
    "google pixel 8a": (8160, 6120),
    "google pixel 7 pro": (8160, 6120),
    "google pixel 7": (8160, 6120),
    "google pixel 7a": (8160, 6120),
    "google pixel 6 pro": (8160, 6120),
    "google pixel 6": (8160, 6120),
    "google pixel 6a": (4032, 3024),
    "google pixel 5": (4032, 3024),
    "google pixel 5a": (4032, 3024),
    "google pixel 4 xl": (4032, 3024),
    "google pixel 4": (4032, 3024),
    "google pixel 4a": (4032, 3024),
    "google pixel 3 xl": (4032, 3024),
    "google pixel 3": (4032, 3024),
    "google pixel 3a": (4032, 3024),
    "google pixel 2 xl": (4032, 3024),
    "google pixel 2": (4032, 3024),
    "google pixel": (4048, 3036),

    # OnePlus
    "oneplus 12": (8160, 6120),    # 50MP main
    "oneplus 11": (8160, 6120),
    "oneplus 10 pro": (8000, 6000),
    "oneplus 9 pro": (8000, 6000),
    "oneplus 9": (8000, 6000),
    "oneplus 8 pro": (8000, 6000),
    "oneplus 8": (8000, 6000),

    # Xiaomi
    "xiaomi 14 ultra": (8192, 6144),  # 50MP 1" sensor
    "xiaomi 14 pro": (8192, 6144),
    "xiaomi 14": (8192, 6144),
    "xiaomi 13 ultra": (8192, 6144),
    "xiaomi 13 pro": (8192, 6144),
    "xiaomi 13": (8192, 6144),

    # Huawei
    "huawei p60 pro": (8192, 6144),
    "huawei p50 pro": (8192, 6144),
    "huawei p40 pro": (8192, 6144),
    "huawei mate 60 pro": (8192, 6144),
    "huawei mate 50 pro": (8192, 6144),

    # Common DSLRs - Canon
    "canon eos r5 mark ii": (8192, 5464),  # 45MP
    "canon eos r5": (8192, 5464),
    "canon eos r6 mark ii": (6000, 4000),
    "canon eos r6": (5472, 3648),
    "canon eos r8": (6000, 4000),
    "canon eos r3": (6000, 4000),
    "canon eos 5d mark iv": (6720, 4480),
    "canon eos 5d mark iii": (5760, 3840),
    "canon eos 5d mark ii": (5616, 3744),
    "canon eos 6d mark ii": (6240, 4160),
    "canon eos 6d": (5472, 3648),
    "canon eos 90d": (6960, 4640),
    "canon eos 80d": (6000, 4000),
    "canon eos 70d": (5472, 3648),
    "canon eos rebel t8i": (6000, 4000),
    "canon eos rebel t7i": (6000, 4000),
    "canon eos rebel t7": (6000, 4000),
    "canon eos rebel t6": (5184, 3456),
    "canon eos m50 mark ii": (6000, 4000),
    "canon eos m50": (6000, 4000),
    "canon powershot g7 x mark iii": (5472, 3648),
    "canon powershot g7 x mark ii": (5472, 3648),

    # Nikon DSLRs and Mirrorless
    "nikon z9": (8256, 5504),       # 45.7MP
    "nikon z8": (8256, 5504),
    "nikon z7 ii": (8256, 5504),
    "nikon z7": (8256, 5504),
    "nikon z6 iii": (6048, 4024),
    "nikon z6 ii": (6048, 4024),
    "nikon z6": (6048, 4024),
    "nikon z5": (6016, 4016),
    "nikon zf": (6048, 4024),
    "nikon z fc": (5568, 3712),
    "nikon d850": (8256, 5504),
    "nikon d810": (7360, 4912),
    "nikon d800": (7360, 4912),
    "nikon d780": (6048, 4024),
    "nikon d750": (6016, 4016),
    "nikon d610": (6016, 4016),
    "nikon d7500": (5568, 3712),
    "nikon d7200": (6000, 4000),
    "nikon d500": (5568, 3712),
    "nikon d3500": (6000, 4000),

    # Sony Alpha
    "sony ilce-1": (8640, 5760),     # A1 50.1MP
    "sony ilce-9m3": (6000, 4000),   # A9 III
    "sony ilce-7rm5": (10656, 7104), # A7R V 61MP
    "sony ilce-7rm4": (9504, 6336),  # A7R IV
    "sony ilce-7rm3": (7952, 5304),  # A7R III
    "sony ilce-7m4": (7008, 4672),   # A7 IV 33MP
    "sony ilce-7m3": (6000, 4000),   # A7 III
    "sony ilce-7cm2": (7008, 4672),  # A7C II
    "sony ilce-7c": (6000, 4000),    # A7C
    "sony ilce-6700": (6000, 4000),  # A6700
    "sony ilce-6600": (6000, 4000),  # A6600
    "sony ilce-6400": (6000, 4000),  # A6400
    "sony ilce-6100": (6000, 4000),  # A6100
    "sony zv-e10": (6000, 4000),     # ZV-E10
    "sony zv-1": (5472, 3648),       # ZV-1

    # Fujifilm
    "fujifilm x-h2": (9504, 7104),   # 40.2MP
    "fujifilm x-h2s": (6240, 4160),  # 26.1MP
    "fujifilm x-t5": (9504, 7104),
    "fujifilm x-t4": (6240, 4160),
    "fujifilm x-t3": (6240, 4160),
    "fujifilm x-t30 ii": (6240, 4160),
    "fujifilm x-s20": (6240, 4160),
    "fujifilm x-s10": (6240, 4160),
    "fujifilm x100vi": (9504, 7104),
    "fujifilm x100v": (6240, 4160),
    "fujifilm gfx 100s": (11648, 8736),  # 102MP medium format
    "fujifilm gfx 100 ii": (11648, 8736),
    "fujifilm gfx 50s ii": (8256, 6192),

    # Panasonic Lumix
    "panasonic dc-s5m2": (6000, 4000),   # S5 II
    "panasonic dc-s5": (6000, 4000),     # S5
    "panasonic dc-s1r": (8368, 5584),    # S1R 47.3MP
    "panasonic dc-s1": (6000, 4000),     # S1
    "panasonic dc-gh6": (6000, 4000),    # GH6
    "panasonic dc-gh5": (5184, 3888),    # GH5
    "panasonic dc-g9": (5184, 3888),     # G9

    # Olympus/OM System
    "olympus e-m1 mark iii": (5184, 3888),
    "olympus e-m1 mark ii": (5184, 3888),
    "olympus e-m5 mark iii": (5184, 3888),
    "om system om-1": (5184, 3888),
    "om system om-5": (5184, 3888),

    # Leica
    "leica q3": (9520, 6336),        # 60MP
    "leica q2": (8368, 5584),        # 47.3MP
    "leica m11": (9504, 6336),       # 60MP
    "leica m10": (6000, 4000),
    "leica sl2": (8368, 5584),

    # DJI Drones and Cameras
    "dji mavic 3 pro": (8064, 6048),  # 48MP Hasselblad
    "dji mavic 3": (5280, 3956),      # 20MP Hasselblad
    "dji mini 4 pro": (8064, 6048),   # 48MP
    "dji mini 3 pro": (8064, 6048),
    "dji air 3": (8064, 6048),
    "dji air 2s": (5472, 3648),
    "dji pocket 3": (4000, 3000),
    "dji osmo action 4": (4000, 3000),

    # GoPro
    "gopro hero12 black": (5568, 4176),  # 27MP
    "gopro hero11 black": (5568, 4176),
    "gopro hero10 black": (5568, 4176),
    "gopro hero9 black": (5568, 4176),
    "gopro hero8 black": (4000, 3000),
}

# Video capabilities: (max_width, max_height)
VIDEO_CAPABILITIES: Dict[str, Tuple[int, int]] = {
    # Apple iPhones - Video
    "apple iphone 15 pro max": (3840, 2160),  # 4K
    "apple iphone 15 pro": (3840, 2160),
    "apple iphone 15": (3840, 2160),
    "apple iphone 14 pro max": (3840, 2160),
    "apple iphone 14 pro": (3840, 2160),
    "apple iphone 14": (3840, 2160),
    "apple iphone 13 pro max": (3840, 2160),  # 4K ProRes
    "apple iphone 13 pro": (3840, 2160),
    "apple iphone 13": (3840, 2160),
    "apple iphone 12 pro max": (3840, 2160),
    "apple iphone 12": (3840, 2160),
    "apple iphone 11 pro max": (3840, 2160),
    "apple iphone 11": (3840, 2160),
    "apple iphone xs max": (3840, 2160),
    "apple iphone xs": (3840, 2160),
    "apple iphone x": (3840, 2160),
    "apple iphone 8 plus": (3840, 2160),
    "apple iphone 8": (3840, 2160),
    "apple iphone 7 plus": (3840, 2160),
    "apple iphone 7": (3840, 2160),
    "apple iphone 6s plus": (3840, 2160),
    "apple iphone 6s": (3840, 2160),
    "apple iphone 6 plus": (1920, 1080),  # 1080p MAX - CANNOT do 4K
    "apple iphone 6": (1920, 1080),        # 1080p MAX - CANNOT do 4K
    "apple iphone 5s": (1920, 1080),
    "apple iphone 5": (1920, 1080),
    "apple iphone 4s": (1920, 1080),
    "apple iphone 4": (1280, 720),         # 720p MAX
}


def normalize_device_name(make: str, model: str) -> str:
    """Normalize device make/model for lookup."""
    combined = f"{make} {model}".lower().strip()
    # Remove extra whitespace
    combined = " ".join(combined.split())
    return combined


def get_device_max_resolution(
    make: str,
    model: str,
    media_type: str = "photo"
) -> Optional[Tuple[int, int]]:
    """
    Get maximum resolution for a device.

    Args:
        make: Device manufacturer (e.g., "Apple")
        model: Device model (e.g., "iPhone 6")
        media_type: "photo" or "video"

    Returns:
        (max_width, max_height) or None if device unknown
    """
    device = normalize_device_name(make, model)

    # Try exact match first
    capabilities = VIDEO_CAPABILITIES if media_type == "video" else DEVICE_CAPABILITIES

    if device in capabilities:
        return capabilities[device]

    # Try partial matching for model variations
    for known_device, resolution in capabilities.items():
        # Check if model name appears in known device
        model_lower = model.lower()
        if model_lower in known_device:
            return resolution
        # Check if known device appears in our query
        if known_device in device:
            return resolution

    return None


def is_resolution_possible(
    make: str,
    model: str,
    width: int,
    height: int,
    media_type: str = "photo"
) -> Optional[bool]:
    """
    Check if a resolution is possible for the claimed device.

    Returns:
        True: Resolution is possible
        False: Resolution is impossible
        None: Device unknown (can't determine)
    """
    max_res = get_device_max_resolution(make, model, media_type)
    if max_res is None:
        return None

    max_width, max_height = max_res
    max_pixels = max_width * max_height
    actual_pixels = width * height

    # Allow 10% tolerance for aspect ratio differences
    return actual_pixels <= max_pixels * 1.1
