import numpy as np
from typing import List, Optional, Iterator
from ..bitstream import BitstreamReader
from ..nal_extractor import NALExtractor, NALUnit
from ..h264_headers import H264HeaderParser
from ..h264_slice import SliceHeader
from ..cavlc import CAVLCDecoder
from ..h264_intra import IntraPredictor
from ..h264_inter import InterPredictor
from ...native import simd

class H264Decoder:
    """
    Main H.264 decoder (Baseline Profile).
    Hybrid: Python orchestration + Assembly kernels.
    Supports Dual-Mode (Forensic/Visual) output.
    """
    
    def __init__(self, deblock: bool = True):
        self.deblock = deblock
        self.header_parser = H264HeaderParser()
        self.intra_pred = IntraPredictor()
        self.inter_pred = InterPredictor()
        self.ref_frame: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0

    def decode_nal(self, nal: NALUnit) -> Optional[np.ndarray]:
        """Process a single NAL unit and return a frame if available."""
        if nal.type == 7: # SPS
            self.header_parser.parse_sps(nal)
            return None
        elif nal.type == 8: # PPS
            self.header_parser.parse_pps(nal)
            return None
        elif nal.type in (1, 5): # Slice (5=IDR, 1=Non-IDR)
            return self._decode_slice(nal)
        return None

    def _decode_slice(self, nal: NALUnit) -> np.ndarray:
        """Decode a slice into a frame."""
        br = BitstreamReader(nal.data[1:])
        sh = SliceHeader(br, self.header_parser)
        sh_info = sh.parse()
        
        pps = self.header_parser.pps[sh_info['pic_parameter_set_id']]
        sps = self.header_parser.sps[pps['sps_id']]
        
        self.width = sps['width']
        self.height = sps['height']
        
        # Initialize frame (grayscale for now)
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # CAVLC context... simplified
        cavlc = CAVLCDecoder(br)
        
        # Macroblock loop
        for mb_y in range(0, self.height, 16):
            for mb_x in range(0, self.width, 16):
                self._decode_macroblock(mb_x, mb_y, frame, cavlc, sh)
        
        # Deblocking filter logic
        if self.deblock:
            # Assembly call: simd.h264_deblock(frame)
            pass
            
        self.ref_frame = frame.copy()
        return frame

    def _decode_macroblock(self, mb_x: int, mb_y: int, frame: np.ndarray, cavlc: CAVLCDecoder, sh: SliceHeader):
        """Decode a single 16x16 macroblock."""
        # Baseline Intra/Inter logic...
        # For each 4x4 block:
        # 1. Decode residuals (CAVLC)
        # 2. Inverse Transform (Assembly: simd.h264_idct_4x4)
        # 3. Predict (Intra/Inter)
        # 4. Combine
        
        if sh.is_i_slice:
            self._decode_intra_mb(mb_x, mb_y, frame, cavlc)
        elif sh.is_p_slice:
            self._decode_inter_mb(mb_x, mb_y, frame, cavlc, sh)

    def _decode_inter_mb(self, mb_x: int, mb_y: int, frame: np.ndarray, cavlc: CAVLCDecoder, sh: SliceHeader):
        # Motion Compensation (P-frame)
        if self.ref_frame is None: return

        # MV decoding... simplified (assume 0 for now)
        mv_x, mv_y = 0, 0
        
        # 1. Prediction (Assembly-accelerated)
        pred = self.inter_pred.predict_p_block(self.ref_frame, mb_x, mb_y, mv_x, mv_y, 16, 16)
        
        # 2. Add residuals (Simplified path)
        for y in range(0, 16, 4):
            for x in range(0, 16, 4):
                coeffs = cavlc.decode_block(nC=0)
                simd.h264_idct_4x4(coeffs)
                pred[y:y+4, x:x+4] = np.clip(pred[y:y+4, x:x+4].astype(np.int16) + coeffs.reshape(4,4), 0, 255).astype(np.uint8)
        
        frame[mb_y:mb_y+16, mb_x:mb_x+16] = pred

    def _decode_intra_mb(self, mb_x: int, mb_y: int, frame: np.ndarray, cavlc: CAVLCDecoder):
        # Extremely simplified I-block reconstruction for Phase 3b.1
        for y in range(0, 16, 4):
            for x in range(0, 16, 4):
                # 1. Residuals
                coeffs = cavlc.decode_block(nC=0)
                # 2. Transform (Assembly)
                simd.h264_idct_4x4(coeffs)
                # 3. Predict (Simplified DC)
                pred = self.intra_pred.predict_4x4(2, None, None, None)
                # 4. Combine
                block = np.clip(pred.astype(np.int16) + coeffs.reshape(4,4), 0, 255).astype(np.uint8)
                frame[mb_y+y:mb_y+y+4, mb_x+x:mb_x+x+4] = block
