from typing import List, Tuple
import numpy as np
from .bitstream import BitstreamReader

class CAVLCDecoder:
    """
    Context-Adaptive Variable-Length Coding (CAVLC) decoder for H.264.
    Used for decoding residual coefficients in Baseline Profile.
    """
    
    def __init__(self, bitstream: BitstreamReader):
        self.br = bitstream

    def decode_block(self, nC: int) -> np.ndarray:
        """
        Decode a 4x4 block of coefficients.
        nC is the context (predicted number of non-zero coefficients).
        """
        coeffs = np.zeros(16, dtype=np.int16)
        
        # 1. TotalCoeff and TrailingOnes
        tc, to = self._decode_coeff_token(nC)
        if tc == 0:
            return coeffs

        # 2. TrailingOnes signs
        to_signs = [0] * to
        for i in range(to):
            to_signs[i] = -1 if self.br.read_bit() == 1 else 1
            
        # 3. Levels (remaining TotalCoeff - TrailingOnes)
        levels = [0] * (tc - to)
        suffix_length = 0 if (tc > 10 and to < 3) else 1
        
        for i in range(tc - to):
            levels[i] = self._decode_level(suffix_length)
            # Update suffix length for next level
            abs_level = abs(levels[i])
            if suffix_length == 0:
                suffix_length = 1
            if abs_level > (3 << (suffix_length - 1)) and suffix_length < 6:
                suffix_length += 1
                
        # Combine levels and signs
        all_levels = [to_signs[i] for i in range(to)] + levels
        
        # 4. TotalZeros
        total_zeros = self._decode_total_zeros(tc)
        
        # 5. Runs (run_before)
        run_before = self._decode_run_before(tc, total_zeros)
        
        # Assemble coefficients in zig-zag order
        pos = 0
        rem_zeros = total_zeros
        for i in range(tc):
            pos += run_before[i]
            coeffs[self.ZIGZAG_SCAN[pos]] = all_levels[i]
            pos += 1
            
        return coeffs

    def _decode_coeff_token(self, nC: int) -> Tuple[int, int]:
        """Simplified coeff_token decoding (actual H.264 uses 4 tables)."""
        # For now, we use a placeholder or partial table implementation.
        # Full CAVLC involves mapping bits to (TotalCoeff, TrailingOnes)
        # using tables based on nC range: [0,2), [2,4), [4,8), and nC >= 8.
        
        # Placeholder: Reading ue() and mapping to something safe for testing
        # In a real decoder, this would be a large lookup table.
        val = self.br.read_ue()
        tc = (val % 17)
        to = min(tc, 3)
        return tc, to

    def _decode_level(self, suffix_length: int) -> int:
        """Decode a Level using suffix length adaptation."""
        prefix = 0
        while self.br.read_bit() == 0:
            prefix += 1
            
        level_code = (prefix << suffix_length)
        if suffix_length > 0:
            level_code += self.br.read_bits(suffix_length)
            
        if prefix == 15 and suffix_length == 0:
             level_code += self.br.read_bits(4) # Special case for 0-length
             
        # Map to signed level
        if level_code % 2 == 0:
            return (level_code + 2) >> 1
        else:
            return (-(level_code + 1)) >> 1

    def _decode_total_zeros(self, tc: int) -> int:
        """Decode TotalZeros based on TotalCoeff."""
        if tc == 16: return 0
        # Uses tables based on TotalCoeff (1..15)
        return self.br.read_ue() # Placeholder

    def _decode_run_before(self, tc: int, total_zeros: int) -> List[int]:
        """Decode run_before values."""
        runs = []
        zeros_left = total_zeros
        for i in range(tc - 1):
            if zeros_left <= 0:
                runs.append(0)
                continue
            run = self.br.read_ue() # Simplified
            run = min(run, zeros_left)
            runs.append(run)
            zeros_left -= run
        runs.append(zeros_left)
        return runs

    ZIGZAG_SCAN = [
        0,  1,  5,  6,
        2,  4,  7,  12,
        3,  8,  11, 13,
        9,  10, 14, 15
    ]
