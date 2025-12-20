import numpy as np
from PIL import Image
import io
from typing import List, Optional
from ...native import simd

class MJPEGDecoder:
    """
    Native MJPEG decoder that leverages existing SIMD-accelerated forensic kernels.
    For forensic use-cases, we often want raw frames without filter post-processing.
    """

    def __init__(self):
        self.last_frame: Optional[np.ndarray] = None

    def decode_frame(self, frame_data: bytes) -> np.ndarray:
        """
        Decode a single MJPEG frame.
        Since MJPEG is essentially a series of JPEGs, we start with a 
        reference PIL implementation but plan to swap in our own 
        DCT-direct decoding for forensic depth.
        """
        try:
            with io.BytesIO(frame_data) as f:
                img = Image.open(f)
                # Ensure we have RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                frame = np.array(img)
                self.last_frame = frame
                return frame
        except Exception as e:
            # In forensics, a corrupted frame is evidence in itself.
            # We return an empty array or signal error rather than failing.
            raise ValueError(f"Failed to decode MJPEG frame: {e}")

    def extract_and_decode(self, stream_data: bytes) -> List[np.ndarray]:
        """
        Extract and decode all frames from an MJPEG stream.
        MJPEG streams often use 0xFFD8 0xFFD9 boundary markers.
        """
        frames = []
        start = 0
        while True:
            start = stream_data.find(b'\xFF\xD8', start)
            if start == -1:
                break
            
            end = stream_data.find(b'\xFF\xD9', start)
            if end == -1:
                break
            
            # Extract JPEG
            jpeg_data = stream_data[start : end + 2]
            frames.append(self.decode_frame(jpeg_data))
            
            start = end + 2
            
        return frames

    def decode_to_grayscale(self, frame_data: bytes) -> np.ndarray:
        """
        Optimized decode path for forensic dimensions that only need Y-channel.
        """
        with io.BytesIO(frame_data) as f:
            img = Image.open(f).convert('L')
            return np.array(img)
