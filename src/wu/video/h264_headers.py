from typing import Dict, Any, Optional
from .bitstream import BitstreamReader
from .nal_extractor import NALUnit

class H264HeaderParser:
    """
    Parses H.264 non-VCL NAL units: SPS (Sequence Parameter Set) 
    and PPS (Picture Parameter Set).
    """

    def __init__(self):
        self.sps: Dict[int, Dict[str, Any]] = {}
        self.pps: Dict[int, Dict[str, Any]] = {}

    def parse_sps(self, nal: NALUnit) -> Dict[str, Any]:
        """Parse Sequence Parameter Set (Type 7)."""
        # Remove emulation prevention bytes before parsing
        data = self._remove_emulation_prevention(nal.data[1:])
        br = BitstreamReader(data)
        
        profile_idc = br.read_bits(8)
        # Skip constraints and level_idc
        br.read_bits(16)
        sps_id = br.read_ue()
        
        log2_max_frame_num = br.read_ue() + 4
        pic_order_cnt_type = br.read_ue()
        
        if pic_order_cnt_type == 0:
            log2_max_pic_order_cnt_lsb = br.read_ue() + 4
        
        max_num_ref_frames = br.read_ue()
        gaps_in_frame_num_value_allowed = br.read_bit()
        
        pic_width_in_mbs = br.read_ue() + 1
        pic_height_in_map_units = br.read_ue() + 1
        frame_mbs_only_flag = br.read_bit()
        
        width = pic_width_in_mbs * 16
        height = (2 - frame_mbs_only_flag) * pic_height_in_map_units * 16
        
        sps_info = {
            'profile_idc': profile_idc,
            'sps_id': sps_id,
            'width': width,
            'height': height,
            'max_ref_frames': max_num_ref_frames
        }
        
        self.sps[sps_id] = sps_info
        return sps_info

    def parse_pps(self, nal: NALUnit) -> Dict[str, Any]:
        """Parse Picture Parameter Set (Type 8)."""
        data = self._remove_emulation_prevention(nal.data[1:])
        br = BitstreamReader(data)
        
        pps_id = br.read_ue()
        sps_id = br.read_ue()
        entropy_coding_mode_flag = br.read_bit() # 0 = CAVLC, 1 = CABAC
        
        pps_info = {
            'pps_id': pps_id,
            'sps_id': sps_id,
            'entropy_coding_mode': 'CABAC' if entropy_coding_mode_flag else 'CAVLC'
        }
        
        self.pps[pps_id] = pps_info
        return pps_info

    def _remove_emulation_prevention(self, data: bytes) -> bytes:
        """H.264 uses 0x000003 as an 'emulation prevention' sequence."""
        out = bytearray()
        i = 0
        while i < len(data):
            if i + 2 < len(data) and data[i] == 0 and data[i+1] == 0 and data[i+2] == 3:
                out.extend(data[i:i+2])
                i += 3
            else:
                out.append(data[i])
                i += 1
        return bytes(out)
