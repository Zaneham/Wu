import struct
import io
from typing import List, Dict, Any, Optional, BinaryIO

class Box:
    """Represents an ISO BMFF Box (Atom)."""
    def __init__(self, type: str, size: int, offset: int, data_offset: int):
        self.type = type
        self.size = size
        self.offset = offset
        self.data_offset = data_offset
        self.children: List['Box'] = []
        self.data: Optional[bytes] = None

    def __repr__(self):
        return f"<Box {self.type} size={self.size} offset={self.offset}>"

class BoxParser:
    """Native parser for ISO Base Media File Format (ISO/IEC 14496-12)."""
    
    CONTAINER_TYPES = {'moov', 'trak', 'mdia', 'minf', 'stbl', 'udta'}

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def parse(self) -> List[Box]:
        """Parse the entire stream and return top-level boxes."""
        boxes = []
        self.stream.seek(0, io.SEEK_END)
        file_size = self.stream.tell()
        self.stream.seek(0)

        while self.stream.tell() < file_size:
            box = self._read_box()
            if not box:
                break
            boxes.append(box)
            # Skip to next box if not a container
            if box.type not in self.CONTAINER_TYPES:
                self.stream.seek(box.offset + box.size)
        
        return boxes

    def _read_box(self) -> Optional[Box]:
        """Read a single box header."""
        offset = self.stream.tell()
        header = self.stream.read(8)
        if len(header) < 8:
            return None

        size, box_type = struct.unpack('>I4s', header)
        box_type = box_type.decode('ascii', errors='ignore')
        
        data_offset = 8
        if size == 1:
            # 64-bit size
            larger_size = self.stream.read(8)
            size = struct.unpack('>Q', larger_size)[0]
            data_offset = 16
        elif size == 0:
            # Extends to end of file
            cur = self.stream.tell()
            self.stream.seek(0, io.SEEK_END)
            size = self.stream.tell() - offset
            self.stream.seek(cur)

        box = Box(box_type, size, offset, offset + data_offset)

        if box_type in self.CONTAINER_TYPES:
            # Recursively parse children
            while self.stream.tell() < (box.offset + box.size):
                child = self._read_box()
                if child:
                    box.children.append(child)
                    if child.type not in self.CONTAINER_TYPES:
                        self.stream.seek(child.offset + child.size)
                else:
                    break
        
        return box

    def find_all_boxes(self, boxes: List[Box], box_type: str) -> List[Box]:
        """Find all boxes of a specific type in the list."""
        return [b for b in boxes if b.type == box_type]

    def find_box(self, boxes: List[Box], type_path: str) -> Optional[Box]:
        """Find a box by its type path (e.g., 'moov/trak/mdia')."""
        parts = type_path.split('/')
        current_level = boxes
        
        target = None
        for part in parts:
            found = False
            for box in current_level:
                if box.type == part:
                    target = box
                    current_level = box.children
                    found = True
                    break
            if not found:
                return None
        return target

    def get_handler_type(self, trak_box: Box) -> Optional[str]:
        """Read the handler type (e.g. 'vide', 'soun') for a track."""
        hdlr = self.find_box([trak_box], 'mdia/hdlr')
        if not hdlr:
            return None
        
        self.stream.seek(hdlr.data_offset)
        # version (1) + flags (3) + component_type (4) + handler_type (4)
        data = self.stream.read(12)
        if len(data) < 12:
            return None
        
        handler_type = data[8:12].decode('ascii', errors='ignore')
        return handler_type

    def get_sample_info(self, trak_box: Box) -> Dict[str, Any]:
        """Extract sample (frame) metadata from a track box."""
        stbl = self.find_box([trak_box], 'mdia/minf/stbl')
        if not stbl:
            return {}

        info = {
            'offsets': [],
            'sizes': [],
        }
        
        # 1. Parse Sample Sizes (stsz)
        stsz = self.find_box(stbl.children, 'stsz')
        if stsz:
            self.stream.seek(stsz.data_offset)
            # version (1) + flags (3) + sample_size (4) + count (4)
            data = self.stream.read(12)
            _, sample_size, count = struct.unpack('>III', data[0:12])
            
            if sample_size > 0:
                info['sizes'] = [sample_size] * count
            else:
                raw_sizes = self.stream.read(count * 4)
                info['sizes'] = list(struct.unpack(f'>{count}I', raw_sizes))

        # 2. Parse Chunk Offsets (stco)
        stco = self.find_box(stbl.children, 'stco')
        if stco:
            self.stream.seek(stco.data_offset)
            # version (1) + flags (3) + count (4)
            data = self.stream.read(8)
            count = struct.unpack('>I', data[4:8])[0]
            raw_offsets = self.stream.read(count * 4)
            info['chunk_offsets'] = list(struct.unpack(f'>{count}I', raw_offsets))

        # 3. Parse Sample-to-Chunk (stsc)
        # Each entry: first_chunk (4), samples_per_chunk (4), sample_description_index (4)
        stsc = self.find_box(stbl.children, 'stsc')
        if stsc:
            self.stream.seek(stsc.data_offset)
            data = self.stream.read(8)
            count = struct.unpack('>I', data[4:8])[0]
            raw_stsc = self.stream.read(count * 12)
            info['stsc'] = []
            for i in range(count):
                entry = struct.unpack('>III', raw_stsc[i*12 : (i+1)*12])
                info['stsc'].append(entry)

        # 4. Parse Time-to-Sample (stts)
        # Each entry: sample_count (4), sample_delta (4)
        stts = self.find_box(stbl.children, 'stts')
        if stts:
            self.stream.seek(stts.data_offset)
            data = self.stream.read(8)
            count = struct.unpack('>I', data[4:8])[0]
            raw_stts = self.stream.read(count * 8)
            info['stts'] = []
            for i in range(count):
                entry = struct.unpack('>II', raw_stts[i*8 : (i+1)*8])
                info['stts'].append(entry)

        return info

    def calculate_frame_offsets(self, info: Dict[str, Any]) -> List[int]:
        """Convert chunk/sample mapping into absolute byte offsets for every frame."""
        if not info.get('chunk_offsets') or not info.get('sizes') or not info.get('stsc'):
            return []

        offsets = []
        sample_idx = 0
        stsc = info['stsc']
        chunk_offsets = info['chunk_offsets']
        sizes = info['sizes']

        for i in range(len(stsc)):
            first_chunk = stsc[i][0]
            samples_per_chunk = stsc[i][1]
            
            # Determine how many chunks this STSC entry applies to
            if i + 1 < len(stsc):
                last_chunk = stsc[i+1][0] - 1
            else:
                last_chunk = len(chunk_offsets)

            for chunk_idx in range(first_chunk - 1, last_chunk):
                current_offset = chunk_offsets[chunk_idx]
                for _ in range(samples_per_chunk):
                    if sample_idx < len(sizes):
                        offsets.append(current_offset)
                        current_offset += sizes[sample_idx]
                        sample_idx += 1
        
        return offsets
