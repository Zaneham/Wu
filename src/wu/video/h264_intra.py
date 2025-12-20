import numpy as np
from typing import List, Optional, Tuple

class IntraPredictor:
    """
    Implements H.264 Intra-prediction modes (4x4).
    For forensic decoding, we must be bit-exact with the spec.
    """
    
    def predict_4x4(self, mode: int, top: Optional[np.ndarray], left: Optional[np.ndarray], top_left: Optional[int]) -> np.ndarray:
        """
        top: 4 pixels above
        left: 4 pixels to the left
        top_left: 1 pixel at top-left
        """
        pred = np.zeros((4, 4), dtype=np.uint8)
        
        if mode == 0: # Vertical
            if top is not None:
                for y in range(4): pred[y, :] = top
        elif mode == 1: # Horizontal
            if left is not None:
                for x in range(4): pred[:, x] = left
        elif mode == 2: # DC
            val = 128
            if top is not None and left is not None:
                val = (np.sum(top) + np.sum(left) + 4) >> 3
            elif top is not None:
                val = (np.sum(top) + 2) >> 2
            elif left is not None:
                val = (np.sum(left) + 2) >> 2
            pred.fill(val)
        # Modes 3-8 (Diagonal, etc.) would follow
        # For Baseline, these 3 cover most cases.
        return pred

    def predict_16x16(self, mode: int, top: Optional[np.ndarray], left: Optional[np.ndarray], top_left: Optional[int]) -> np.ndarray:
        """Intra-16x16 prediction."""
        pred = np.zeros((16, 16), dtype=np.uint8)
        # Vertical, Horizontal, DC, Planar
        if mode == 0: # Vertical
            for y in range(16): pred[y, :] = top
        elif mode == 1: # Horizontal
            for x in range(16): pred[:, x] = left
        elif mode == 2: # DC
            # Sum top and left...
            pass
        return pred
