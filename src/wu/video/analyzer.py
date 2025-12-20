from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import io
import numpy as np

from .box_parser import BoxParser
from .nal_extractor import NALExtractor
from .h264_headers import H264HeaderParser
from .decoders.mjpeg import MJPEGDecoder
from .decoders.h264 import H264Decoder
from ..state import DimensionResult, DimensionState, Confidence, Evidence

class VideoAnalyzer:
    """
    Orchestrates native container parsing and frame decoding for forensics.
    Handles MP4/MOV and MJPEG sources without external dependencies.
    """
    
    def __init__(self, use_simd: bool = True, deblock: bool = True):
        self.use_simd = use_simd
        self.deblock = deblock
        self.mjpeg_decoder = MJPEGDecoder()
        self.h264_decoder = H264Decoder(deblock=deblock)
        self.nal_extractor = NALExtractor()

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Perform high-level analysis of the video container and stream.
        """
        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="video",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(finding="File not found", explanation=str(path))]
            )

        try:
            with open(file_path, 'rb') as f:
                parser = BoxParser(f)
                boxes = parser.parse()
                
                # Basic container validation
                ftyp = parser.find_box(boxes, 'ftyp')
                moov = parser.find_box(boxes, 'moov')
                
                if not moov:
                    return DimensionResult(
                        dimension="video",
                        state=DimensionState.UNCERTAIN,
                        confidence=Confidence.NA,
                        evidence=[Evidence(finding="Invalid container", explanation="No moov box found")]
                    )

                # Find tracks
                traks = parser.find_all_boxes(moov.children, 'trak')
                video_trak = None
                audio_trak = None
                
                for trak in traks:
                    h_type = parser.get_handler_type(trak)
                    if h_type == 'vide' and not video_trak:
                        video_trak = trak
                    elif h_type == 'soun' and not audio_trak:
                        audio_trak = trak

                if not video_trak:
                     return DimensionResult(
                        dimension="video",
                        state=DimensionState.UNCERTAIN,
                        confidence=Confidence.NA,
                        evidence=[Evidence(finding="No video track", explanation="No usable video track found in container")]
                    )

                sample_info = parser.get_sample_info(video_trak)
                offsets = parser.calculate_frame_offsets(sample_info)
                
                return DimensionResult(
                    dimension="video",
                    state=DimensionState.CLEAN, # placeholder
                    confidence=Confidence.HIGH,
                    evidence=[
                        Evidence(
                            finding="Native container parsed",
                            explanation=f"Found {len(offsets)} frames in {len(traks)} tracks."
                        )
                    ]
                )

        except Exception as e:
            return DimensionResult(
                dimension="video",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(finding="Parsing error", explanation=str(e))]
            )

    def iter_frames(self, file_path: str) -> Iterator[np.ndarray]:
        """
        Yield decoded frames one by one for batch analysis.
        """
        with open(file_path, 'rb') as f:
            parser = BoxParser(f)
            boxes = parser.parse()
            moov = parser.find_box(boxes, 'moov')
            if not moov: return

            # Find video track natively
            traks = parser.find_all_boxes(moov.children, 'trak')
            video_trak = next((t for t in traks if parser.get_handler_type(t) == 'vide'), None)
            if not video_trak: return

            sample_info = parser.get_sample_info(video_trak)
            offsets = parser.calculate_frame_offsets(sample_info)
            sizes = sample_info.get('sizes', [])

            for i, offset in enumerate(offsets):
                if i >= len(sizes): break
                try:
                    f.seek(offset)
                    data = f.read(sizes[i])
                    
                    if data.startswith(b'\xFF\xD8'):
                        yield self.mjpeg_decoder.decode_frame(data)
                    else:
                        nalus = self.nal_extractor.extract_from_avcc(data)
                        for nalu in nalus:
                            nalu_type = nalu[0] & 0x1F
                            if nalu_type in [1, 5]:
                                frame = self.h264_decoder.decode_nal(nalu)
                                if frame is not None:
                                    yield frame
                except Exception:
                    continue

    def iter_audio_samples(self, file_path: str) -> Iterator[bytes]:
        """
        Yield raw audio samples (ADTS/MP3/etc) bit-exactly from the container.
        """
        with open(file_path, 'rb') as f:
            parser = BoxParser(f)
            boxes = parser.parse()
            moov = parser.find_box(boxes, 'moov')
            if not moov: return

            traks = parser.find_all_boxes(moov.children, 'trak')
            audio_trak = next((t for t in traks if parser.get_handler_type(t) == 'soun'), None)
            if not audio_trak: return

            sample_info = parser.get_sample_info(audio_trak)
            offsets = parser.calculate_frame_offsets(sample_info)
            sizes = sample_info.get('sizes', [])

            for i, offset in enumerate(offsets):
                if i >= len(sizes): break
                f.seek(offset)
                yield f.read(sizes[i])
