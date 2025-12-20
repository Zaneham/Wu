import struct
from typing import List, Optional

class NALUnit:
    """Represents an H.264 Network Abstraction Layer (NAL) Unit."""
    def __init__(self, type: int, data: bytes):
        self.type = type
        self.data = data

    @property
    def type_name(self) -> str:
        names = {
            1: "Coded slice of a non-IDR picture",
            5: "Coded slice of an IDR picture (I-frame)",
            7: "Sequence Parameter Set (SPS)",
            8: "Picture Parameter Set (PPS)",
            9: "Access Unit Delimiter (AUD)"
        }
        return names.get(self.type, f"Other ({self.type})")

class NALExtractor:
    """Logic to extract NAL units from raw sample data (AVCC or Annex B)."""
    
    def __init__(self, nalu_length_size: int = 4):
        self.nalu_length_size = nalu_length_size

    def extract_from_avcc(self, sample_data: bytes) -> List[NALUnit]:
        """Extract units from AVCC format (used in MP4/MOV) where each NAL is prefixed by its length."""
        nalus = []
        offset = 0
        while offset < len(sample_data):
            if offset + self.nalu_length_size > len(sample_data):
                break
            
            # Read length
            if self.nalu_length_size == 4:
                length = struct.unpack('>I', sample_data[offset : offset+4])[0]
            elif self.nalu_length_size == 1:
                length = sample_data[offset]
            elif self.nalu_length_size == 2:
                length = struct.unpack('>H', sample_data[offset : offset+2])[0]
            else:
                raise ValueError(f"Unsupported NALU length size: {self.nalu_length_size}")
                
            offset += self.nalu_length_size
            
            if offset + length > len(sample_data):
                # Potentially malformed or truncated
                break
                
            nalu_data = sample_data[offset : offset+length]
            if nalu_data:
                # First byte contains type (lower 5 bits)
                nal_type = nalu_data[0] & 0x1F
                nalus.append(NALUnit(nal_type, nalu_data))
            
            offset += length
            
        return nalus

    def to_annex_b(self, nal_unit: NALUnit) -> bytes:
        """Convert a NAL unit to Annex B format (start codes) for decoding."""
        return b'\x00\x00\x00\x01' + nal_unit.data
