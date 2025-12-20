"""
EXIF Thumbnail forensic analysis.

Detects manipulation by comparing the embedded EXIF thumbnail
(created at capture time) with the main image.

When someone edits an image in Photoshop/GIMP/etc, they typically
modify the main image but forget the embedded thumbnail. This creates
a "before and after" comparison that's devastating in court.

Court relevance:
    "The embedded thumbnail created at capture time shows a person
    who does not appear in the submitted image. The main image was
    modified after capture while the thumbnail preserves the original."

References:
    Kee, E. & Farid, H. (2010). Digital Image Authentication from
        Thumbnails. SPIE Media Forensics and Security.
    JEITA CP-3451C - EXIF 2.3 specification (thumbnail requirements)
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


@dataclass
class ThumbnailComparison:
    """Result of thumbnail vs main image comparison."""
    has_thumbnail: bool
    thumbnail_size: Optional[Tuple[int, int]] = None
    main_size: Optional[Tuple[int, int]] = None
    similarity: float = 0.0  # 0-1, structural similarity
    mse: float = 0.0  # Mean squared error
    significant_difference: bool = False
    difference_regions: Optional[np.ndarray] = None  # Heatmap of differences


class ThumbnailAnalyzer:
    """
    Compares EXIF thumbnail to main image for manipulation detection.

    EXIF thumbnails are:
    1. Created at capture time by the camera
    2. Stored in the EXIF metadata block
    3. Typically 160x120 or similar small size
    4. Often forgotten when editing the main image

    If thumbnail differs significantly from main image:
    - Main image was edited after capture
    - Thumbnail shows the ORIGINAL scene
    - Strong forensic evidence of manipulation

    Limitations:
    - Some editors update thumbnails (Lightroom, Camera Raw)
    - Some images have no thumbnail
    - Very small thumbnails limit comparison detail
    - Cropping alone will cause mismatch (not necessarily manipulation)
    """

    # Similarity threshold - below this indicates manipulation
    SIMILARITY_THRESHOLD = 0.85
    MSE_THRESHOLD = 0.02  # Normalized MSE threshold

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Compare EXIF thumbnail to main image.

        Returns DimensionResult indicating consistency.
        """
        if not HAS_NUMPY or not HAS_PIL:
            return DimensionResult(
                dimension="thumbnail",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Dependencies not available",
                    explanation="numpy and PIL required for thumbnail analysis"
                )]
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="thumbnail",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        # Extract thumbnail and load main image
        try:
            thumbnail = self._extract_thumbnail(file_path)
            main_image = Image.open(file_path)
            if main_image.mode != 'RGB':
                main_image = main_image.convert('RGB')
        except Exception as e:
            return DimensionResult(
                dimension="thumbnail",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="Cannot load image",
                    explanation=str(e)
                )]
            )

        evidence = []

        # No thumbnail
        if thumbnail is None:
            evidence.append(Evidence(
                finding="No EXIF thumbnail found",
                explanation=(
                    "Image has no embedded thumbnail. This is common for "
                    "screenshots, web images, or images that have been "
                    "re-saved without EXIF data."
                )
            ))
            return DimensionResult(
                dimension="thumbnail",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=evidence,
                methodology="EXIF thumbnail extraction"
            )

        # Compare thumbnail to main image
        comparison = self._compare_images(thumbnail, main_image)

        evidence.append(Evidence(
            finding=f"Thumbnail found: {comparison.thumbnail_size[0]}x{comparison.thumbnail_size[1]}",
            explanation=f"Main image: {comparison.main_size[0]}x{comparison.main_size[1]}"
        ))

        evidence.append(Evidence(
            finding=f"Similarity: {comparison.similarity:.1%}",
            explanation=f"Structural similarity between thumbnail and main image"
        ))

        # Check for significant difference
        if comparison.significant_difference:
            evidence.append(Evidence(
                finding="THUMBNAIL MISMATCH DETECTED",
                explanation=(
                    "The embedded thumbnail (created at capture time) differs "
                    "significantly from the main image. This indicates the main "
                    "image was modified after capture while the thumbnail "
                    "preserves the original scene."
                ),
                citation="Kee & Farid (2010) - Digital image authentication from thumbnails"
            ))

            return DimensionResult(
                dimension="thumbnail",
                state=DimensionState.INCONSISTENT,
                confidence=Confidence.HIGH,
                evidence=evidence,
                methodology="EXIF thumbnail vs main image comparison",
                raw_data={
                    "thumbnail_size": comparison.thumbnail_size,
                    "main_size": comparison.main_size,
                    "similarity": comparison.similarity,
                    "mse": comparison.mse
                }
            )

        # Thumbnail matches main image
        evidence.append(Evidence(
            finding="Thumbnail consistent with main image",
            explanation="No evidence of post-capture modification detected via thumbnail"
        ))

        return DimensionResult(
            dimension="thumbnail",
            state=DimensionState.CONSISTENT,
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            methodology="EXIF thumbnail vs main image comparison"
        )

    def _extract_thumbnail(self, file_path: str) -> Optional[Image.Image]:
        """
        Extract EXIF thumbnail from image file.

        Tries multiple methods for maximum compatibility.
        """
        # Method 1: PIL's built-in EXIF thumbnail extraction
        try:
            with Image.open(file_path) as img:
                exif = img._getexif()
                if exif and 274 in exif:  # Orientation tag present = has EXIF
                    pass

                # Check for thumbnail in EXIF
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    # Tag 513 = JPEGInterchangeFormat (thumbnail offset)
                    # Tag 514 = JPEGInterchangeFormatLength
                    if exif_data and 513 in exif_data and 514 in exif_data:
                        pass  # Has thumbnail pointers

                # Try to get thumbnail via info
                if 'exif' in img.info:
                    exif_bytes = img.info['exif']
                    # Parse EXIF for thumbnail
                    thumb = self._parse_exif_thumbnail(exif_bytes)
                    if thumb:
                        return thumb
        except Exception:
            pass

        # Method 2: Use exifread library
        if HAS_EXIFREAD:
            try:
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f, details=True)

                    # Look for thumbnail
                    if 'JPEGThumbnail' in tags:
                        thumb_data = tags['JPEGThumbnail']
                        if thumb_data:
                            thumb = Image.open(BytesIO(thumb_data))
                            if thumb.mode != 'RGB':
                                thumb = thumb.convert('RGB')
                            return thumb
            except Exception:
                pass

        # Method 3: Manual EXIF parsing for JPEG
        try:
            thumb = self._extract_jpeg_thumbnail(file_path)
            if thumb:
                return thumb
        except Exception:
            pass

        return None

    def _parse_exif_thumbnail(self, exif_bytes: bytes) -> Optional[Image.Image]:
        """Parse EXIF bytes to extract embedded thumbnail."""
        try:
            # Look for JPEG thumbnail marker (FFD8)
            # Thumbnails are typically stored after the main EXIF data
            idx = exif_bytes.find(b'\xff\xd8', 10)  # Skip first bytes
            if idx > 0:
                # Find end of JPEG (FFD9)
                end_idx = exif_bytes.find(b'\xff\xd9', idx)
                if end_idx > idx:
                    thumb_data = exif_bytes[idx:end_idx+2]
                    thumb = Image.open(BytesIO(thumb_data))
                    if thumb.mode != 'RGB':
                        thumb = thumb.convert('RGB')
                    return thumb
        except Exception:
            pass
        return None

    def _extract_jpeg_thumbnail(self, file_path: str) -> Optional[Image.Image]:
        """Extract thumbnail from JPEG file by parsing APP1 marker."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(65536)  # Read first 64KB (enough for EXIF)

                # Find APP1 marker (EXIF)
                app1_start = data.find(b'\xff\xe1')
                if app1_start < 0:
                    return None

                # Get APP1 length
                length = int.from_bytes(data[app1_start+2:app1_start+4], 'big')

                # Extract APP1 data
                app1_data = data[app1_start+4:app1_start+2+length]

                # Look for embedded JPEG thumbnail
                return self._parse_exif_thumbnail(app1_data)
        except Exception:
            return None

    def _compare_images(
        self,
        thumbnail: Image.Image,
        main_image: Image.Image
    ) -> ThumbnailComparison:
        """
        Compare thumbnail to main image.

        Resizes main image to thumbnail size and computes similarity.
        """
        thumb_size = thumbnail.size
        main_size = main_image.size

        # Resize main image to thumbnail size
        main_resized = main_image.resize(thumb_size, Image.Resampling.LANCZOS)

        # Convert to numpy arrays
        thumb_arr = np.array(thumbnail, dtype=np.float64) / 255.0
        main_arr = np.array(main_resized, dtype=np.float64) / 255.0

        # Compute Mean Squared Error
        mse = np.mean((thumb_arr - main_arr) ** 2)

        # Compute Structural Similarity (simplified SSIM)
        similarity = self._compute_ssim(thumb_arr, main_arr)

        # Compute difference heatmap
        diff = np.mean(np.abs(thumb_arr - main_arr), axis=2)

        # Determine if difference is significant
        significant = (
            similarity < self.SIMILARITY_THRESHOLD or
            mse > self.MSE_THRESHOLD
        )

        return ThumbnailComparison(
            has_thumbnail=True,
            thumbnail_size=thumb_size,
            main_size=main_size,
            similarity=similarity,
            mse=mse,
            significant_difference=significant,
            difference_regions=diff
        )

    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Compute Structural Similarity Index (simplified).

        Full SSIM uses windowed computation; this is a global approximation.
        """
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Flatten to 1D for each channel and compute
        ssim_values = []

        for c in range(min(img1.shape[2], img2.shape[2]) if len(img1.shape) > 2 else 1):
            if len(img1.shape) > 2:
                x = img1[:, :, c].flatten()
                y = img2[:, :, c].flatten()
            else:
                x = img1.flatten()
                y = img2.flatten()

            mu_x = np.mean(x)
            mu_y = np.mean(y)
            sigma_x = np.std(x)
            sigma_y = np.std(y)
            sigma_xy = np.mean((x - mu_x) * (y - mu_y))

            ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))

            ssim_values.append(ssim)

        return float(np.mean(ssim_values))


def get_thumbnail_analyzer() -> ThumbnailAnalyzer:
    """Factory function for thumbnail analyzer."""
    return ThumbnailAnalyzer()
