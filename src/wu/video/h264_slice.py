from typing import Dict, Any, Optional
from .bitstream import BitstreamReader
from .h264_headers import H264HeaderParser

class SliceHeader:
    """
    Parses H.264 Slice Header (Baseline Profile).
    """
    def __init__(self, br: BitstreamReader, header_parser: H264HeaderParser):
        self.br = br
        self.header_parser = header_parser
        self.info: Dict[str, Any] = {}

    def parse(self) -> Dict[str, Any]:
        """Parse the slice_header() as per spec."""
        self.info = {}
        
        self.info['first_mb_in_slice'] = self.br.read_ue()
        self.info['slice_type'] = self.br.read_ue() # 0: P, 2: I, 5: P, 7: I
        self.info['pic_parameter_set_id'] = self.br.read_ue()
        
        pps = self.header_parser.pps.get(self.info['pic_parameter_set_id'])
        if not pps:
            raise ValueError(f"PPS {self.info['pic_parameter_set_id']} not found")
        
        sps = self.header_parser.sps.get(pps['sps_id'])
        if not sps:
            raise ValueError(f"SPS {pps['sps_id']} not found")

        self.info['frame_num'] = self.br.read_bits(8) # simplified, should use sps.log2_max_frame_num
        
        if self.info['slice_type'] % 5 != 2: # Not an I-slice
            self.info['num_ref_idx_active_override'] = self.br.read_bit()
            if self.info['num_ref_idx_active_override']:
                self.info['num_ref_idx_l0_active'] = self.br.read_ue() + 1
        
        # Skip ref_pic_list_reordering and prediction_weight_table for Baseline
        # Skip dec_ref_pic_marking
        
        self.info['slice_qp_delta'] = self.br.read_se()
        
        # Deblocking filter control
        self.info['disable_deblocking_filter_idc'] = self.br.read_ue()
        if self.info['disable_deblocking_filter_idc'] != 1:
            self.info['slice_alpha_c0_offset_div2'] = self.br.read_se()
            self.info['slice_beta_offset_div2'] = self.br.read_se()
            
        return self.info

    @property
    def is_i_slice(self) -> bool:
        return self.info.get('slice_type', 0) % 5 == 2

    @property
    def is_p_slice(self) -> bool:
        return self.info.get('slice_type', 0) % 5 == 0
