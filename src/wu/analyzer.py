"""
Main Wu analyzer orchestrating forensic analysis.

Coordinates dimension analyzers and aggregates results into
a court-ready WuAnalysis report.

Supports parallel execution of independent dimension analyzers
for improved performance on multi-core systems.

References:
    Daubert v. Merrell Dow Pharmaceuticals, 509 U.S. 579 (1993)
    Federal Rules of Evidence 702 (Expert Testimony)
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

from .state import (
    DimensionResult,
    DimensionState,
    Confidence,
    Evidence,
    WuAnalysis,
    OverallAssessment,
)
from .aggregator import EpistemicAggregator
from .dimensions import (
    MetadataAnalyzer,
    C2PAAnalyzer,
    VisualAnalyzer,
    ENFAnalyzer,
    CopyMoveAnalyzer,
    PRNUAnalyzer,
    BlockGridAnalyzer,
    LightingAnalyzer,
    AudioAnalyzer,
    ThumbnailAnalyzer,
    ShadowAnalyzer,
    PerspectiveAnalyzer,
    QuantizationAnalyzer,
    AIGenerationAnalyzer,
)
from .video.analyzer import VideoAnalyzer

__version__ = "1.1.0"

# Default number of workers for parallel execution
DEFAULT_MAX_WORKERS = min(8, (os.cpu_count() or 4))


class WuAnalyzer:
    """
    Main forensic analyzer for media files.

    Phase 0: Metadata, C2PA, and visual analysis (no ML dependencies)
    Phase 1: ENF (Electric Network Frequency) analysis for audio/video

    Supports parallel execution of dimension analyzers for improved
    performance on multi-core systems.

    Usage:
        analyzer = WuAnalyzer()
        result = analyzer.analyze("suspicious_photo.jpg")
        print(result.to_json())

        # For audio/video with ENF analysis:
        analyzer = WuAnalyzer(enable_enf=True)
        result = analyzer.analyze("recording.wav")

        # Enable parallel execution (default):
        analyzer = WuAnalyzer(parallel=True)

        # Control parallelism:
        analyzer = WuAnalyzer(parallel=True, max_workers=4)
    """

    def __init__(
        self,
        enable_metadata: bool = True,
        enable_c2pa: bool = True,
        enable_visual: bool = True,
        enable_enf: bool = False,  # Disabled by default (requires audio)
        enable_copymove: bool = False,  # Disabled by default (computationally expensive)
        enable_prnu: bool = False,  # Disabled by default (computationally expensive)
        enable_blockgrid: bool = False,  # Disabled by default (JPEG-specific)
        enable_lighting: bool = False,  # Disabled by default (computationally expensive)
        enable_audio: bool = False,  # Disabled by default (requires audio files)
        enable_thumbnail: bool = False,  # Disabled by default
        enable_shadows: bool = False,  # Disabled by default (computationally expensive)
        enable_perspective: bool = False,  # Disabled by default (computationally expensive)
        enable_quantization: bool = False,  # Disabled by default (JPEG-specific)
        enable_aigen: bool = False,  # Disabled by default (AI generation indicators)
        enable_video: bool = True,  # Enabled by default for video files
        parallel: bool = True,  # Enable parallel dimension execution
        max_workers: Optional[int] = None,  # Max parallel workers (None = auto)
    ):
        """
        Initialize analyzer with dimension configuration.

        Args:
            enable_metadata: Enable metadata forensics
            enable_c2pa: Enable C2PA content credential verification
            enable_visual: Enable visual forensics (JPEG artifacts, ELA)
            enable_enf: Enable ENF analysis (for audio/video files)
            enable_copymove: Enable copy-move (clone) detection
            enable_prnu: Enable PRNU (sensor fingerprint) analysis
            enable_blockgrid: Enable JPEG block grid alignment analysis
            enable_lighting: Enable lighting direction consistency analysis
            enable_audio: Enable audio forensics (spectral, noise floor, ENF)
            enable_thumbnail: Enable EXIF thumbnail mismatch detection
            enable_shadows: Enable shadow direction consistency analysis
            enable_perspective: Enable vanishing point consistency analysis
            enable_quantization: Enable JPEG quantization table forensics
            enable_aigen: Enable AI generation indicator analysis
            parallel: Enable parallel execution of dimension analyzers
            max_workers: Maximum number of parallel workers (None = auto)
        """
        self.enable_metadata = enable_metadata
        self.enable_c2pa = enable_c2pa
        self.enable_visual = enable_visual
        self.enable_enf = enable_enf
        self.enable_copymove = enable_copymove
        self.enable_prnu = enable_prnu
        self.enable_blockgrid = enable_blockgrid
        self.enable_lighting = enable_lighting
        self.enable_audio = enable_audio
        self.enable_thumbnail = enable_thumbnail
        self.enable_shadows = enable_shadows
        self.enable_perspective = enable_perspective
        self.enable_quantization = enable_quantization
        self.enable_aigen = enable_aigen
        self.parallel = parallel
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.aggregator = EpistemicAggregator()

        # Initialize dimension analyzers
        self._metadata_analyzer = MetadataAnalyzer() if enable_metadata else None
        self._c2pa_analyzer = C2PAAnalyzer() if enable_c2pa else None
        self._visual_analyzer = VisualAnalyzer() if enable_visual else None
        self._enf_analyzer = ENFAnalyzer() if enable_enf else None
        self._copymove_analyzer = CopyMoveAnalyzer() if enable_copymove else None
        self._prnu_analyzer = PRNUAnalyzer() if enable_prnu else None
        self._blockgrid_analyzer = BlockGridAnalyzer() if enable_blockgrid else None
        self._lighting_analyzer = LightingAnalyzer() if enable_lighting else None
        self._audio_analyzer = AudioAnalyzer() if enable_audio else None
        self._thumbnail_analyzer = ThumbnailAnalyzer() if enable_thumbnail else None
        self._shadow_analyzer = ShadowAnalyzer() if enable_shadows else None
        self._perspective_analyzer = PerspectiveAnalyzer() if enable_perspective else None
        self._quantization_analyzer = QuantizationAnalyzer() if enable_quantization else None
        self._aigen_analyzer = AIGenerationAnalyzer() if enable_aigen else None
        self._video_analyzer = VideoAnalyzer() if enable_video else None

        # Build list of enabled analyzers for parallel execution
        self._analyzer_config = self._build_analyzer_config()

    def _build_analyzer_config(self) -> List[Tuple[str, Any]]:
        """Build configuration list of enabled analyzers."""
        analyzers = {
            "metadata": self._metadata_analyzer,
            "c2pa": self._c2pa_analyzer,
            "visual": self._visual_analyzer,
            "enf": self._enf_analyzer,
            "copymove": self._copymove_analyzer,
            "prnu": self._prnu_analyzer,
            "blockgrid": self._blockgrid_analyzer,
            "lighting": self._lighting_analyzer,
            "audio": self._audio_analyzer,
            "thumbnail": self._thumbnail_analyzer,
            "shadows": self._shadow_analyzer,
            "perspective": self._perspective_analyzer,
            "quantization": self._quantization_analyzer,
            "aigen": self._aigen_analyzer,
            "video": self._video_analyzer,
        }
        return [(name, analyzer) for name, analyzer in analyzers.items() if analyzer]

    def analyze(self, file_path: str) -> WuAnalysis:
        """
        Perform forensic analysis on a media file.

        Returns WuAnalysis with:
        - Per-dimension results
        - Overall assessment
        - Human-readable summary
        - File hash for chain of custody

        When parallel=True, dimension analyzers run concurrently
        for improved performance.

        Args:
            file_path: Path to media file

        Returns:
            WuAnalysis containing all findings
        """
        path = Path(file_path)

        # Compute file hash for chain of custody
        file_hash = self._compute_hash(path)

        # Initialize analysis
        analysis = WuAnalysis(
            file_path=str(path.absolute()),
            file_hash=file_hash,
            analyzed_at=datetime.now(),
            wu_version=__version__,
        )

        # Run dimension analyzers
        if self.parallel and len(self._analyzer_config) > 1:
            results = self._analyze_parallel(str(path))
        else:
            results = self._analyze_sequential(str(path))

        # --- Phase 4: Cross-Modal Integration ---
        # If it's a video, we also want to run image analyzers on extracted frames and audio analyzer on bit-exact stream
        is_video = path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        if is_video and self._video_analyzer:
            try:
                import tempfile
                from .state import Evidence
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    # 1. Image Forensics on Decoded Frames
                    frame_count = 0
                    max_frames = 5
                    visual_tools = []
                    if self._visual_analyzer: visual_tools.append(self._visual_analyzer)
                    if self._copymove_analyzer: visual_tools.append(self._copymove_analyzer)
                    if self._prnu_analyzer: visual_tools.append(self._prnu_analyzer)
                    
                    for i, frame in enumerate(self._video_analyzer.iter_frames(str(path))):
                        if frame_count >= max_frames: break
                        frame_path = os.path.join(tmpdir, f"frame_{i}.png")
                        from PIL import Image
                        Image.fromarray(frame).save(frame_path)
                        
                        for tool in visual_tools:
                            tool_result = tool.analyze(frame_path)
                            if tool_result.state in [DimensionState.SUSPICIOUS, DimensionState.INCONSISTENT]:
                                if "video" in results:
                                    video_res = results["video"]
                                    video_res.state = DimensionState.SUSPICIOUS
                                    video_res.evidence.append(Evidence(
                                        finding=f"Visual Anomaly in Frame {i} ({tool_result.dimension})",
                                        explanation=tool_result.evidence[0].explanation if tool_result.evidence else "Anomaly detected in pixel data"
                                    ))
                        frame_count += 1

                    # 2. Hybrid Audio Forensics (Native Extraction -> External Decode)
                    if self._audio_analyzer:
                        # Locate audio trak natively
                        audio_samples = list(self._video_analyzer.iter_audio_samples(str(path)))
                        if audio_samples:
                            # Write raw samples to a temporary file for AudioAnalyzer
                            # Note: We append them, though the analyzer might need a proper header
                            # For MP3/AAC/ADTS, simple concatenation often works for pydub/ffmpeg
                            raw_audio_path = os.path.join(tmpdir, "extracted_audio.dat")
                            with open(raw_audio_path, "wb") as af:
                                for sample in audio_samples:
                                    af.write(sample)
                            
                            audio_result = self._audio_analyzer.analyze(raw_audio_path)
                            results["audio"] = audio_result
                            
                            # 3. Cross-Modal Corroboration
                            if "video" in results:
                                video_res = results["video"]
                                if audio_result.state == DimensionState.CONSISTENT and video_res.state == DimensionState.SUSPICIOUS:
                                    video_res.evidence.append(Evidence(
                                        finding="Cross-modal anomaly",
                                        explanation="Video shows splicing markers while audio appears continuous. Typical of deepfake/visual splicing."
                                    ))
            except Exception as e:
                if "video" in results:
                    results["video"].evidence.append(Evidence(
                        finding="Cross-modal analysis failed",
                        explanation=str(e)
                    ))

        # Assign results to analysis object
        dimension_results: List[DimensionResult] = []
        for dim_name, result in results.items():
            setattr(analysis, dim_name, result)
            dimension_results.append(result)

        # Aggregate results
        analysis.overall = self.aggregator.aggregate(dimension_results)
        analysis.findings_summary = self.aggregator.generate_summary(dimension_results)
        analysis.corroboration_summary = self.aggregator.generate_corroboration_summary(
            dimension_results
        )

        return analysis

    def _analyze_sequential(self, file_path: str) -> Dict[str, DimensionResult]:
        """Run all dimension analyzers sequentially."""
        results = {}
        for dim_name, analyzer in self._analyzer_config:
            results[dim_name] = analyzer.analyze(file_path)
        return results

    def _analyze_parallel(self, file_path: str) -> Dict[str, DimensionResult]:
        """
        Run dimension analyzers in parallel using ThreadPoolExecutor.

        Uses threads rather than processes because:
        1. Dimension analyzers are largely I/O bound (file reads)
        2. NumPy/PIL release the GIL during computation
        3. Lower overhead than process spawning
        """
        results = {}
        num_workers = min(self.max_workers, len(self._analyzer_config))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all analyzer tasks
            future_to_dim = {
                executor.submit(analyzer.analyze, file_path): dim_name
                for dim_name, analyzer in self._analyzer_config
            }

            # Collect results as they complete
            for future in as_completed(future_to_dim):
                dim_name = future_to_dim[future]
                try:
                    results[dim_name] = future.result()
                except Exception as e:
                    # If an analyzer fails, create an error result
                    results[dim_name] = DimensionResult(
                        dimension=dim_name,
                        state=DimensionState.UNCERTAIN,
                        confidence=Confidence.NA,
                        evidence=[
                            Evidence(
                                finding=f"Analysis failed: {type(e).__name__}",
                                explanation=str(e),
                            )
                        ],
                        methodology="Error during parallel execution",
                    )

        return results

    def analyze_batch(
        self,
        file_paths: List[str],
        parallel_files: bool = True,
        max_file_workers: Optional[int] = None,
    ) -> List[WuAnalysis]:
        """
        Analyze multiple files with optional file-level parallelism.

        Args:
            file_paths: List of paths to media files
            parallel_files: Enable parallel processing of files
            max_file_workers: Max workers for file-level parallelism

        Returns:
            List of WuAnalysis results in same order as input
        """
        if not file_paths:
            return []

        if not parallel_files or len(file_paths) == 1:
            return [self.analyze(fp) for fp in file_paths]

        # Use ThreadPoolExecutor for file-level parallelism
        # Each file analysis may itself use parallel dimension execution
        num_workers = max_file_workers or min(4, len(file_paths))

        # Store results with index to preserve order
        results: Dict[int, WuAnalysis] = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze, fp): idx
                for idx, fp in enumerate(file_paths)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Create error analysis for failed file
                    results[idx] = WuAnalysis(
                        file_path=file_paths[idx],
                        file_hash=f"ERROR:{type(e).__name__}",
                        analyzed_at=datetime.now(),
                        wu_version=__version__,
                        overall=OverallAssessment.INSUFFICIENT_DATA,
                        findings_summary=[f"Analysis failed: {str(e)}"],
                    )

        # Return results in original order
        return [results[i] for i in range(len(file_paths))]

    def _compute_hash(self, path: Path) -> str:
        """
        Compute SHA-256 hash for chain of custody.

        The hash provides:
        1. Proof that analysis was performed on specific file
        2. Detection if file is modified after analysis
        3. Reproducibility of analysis
        """
        if not path.exists():
            return "FILE_NOT_FOUND"

        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                # Read in 64KB chunks for large files
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            return f"HASH_ERROR:{type(e).__name__}"

    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported file formats.

        Phase 0 supports common image formats via PIL/exifread.
        Phase 1 adds audio/video for ENF analysis.
        """
        return [
            # Images
            ".jpg", ".jpeg", ".png", ".tiff", ".tif",
            ".heic", ".heif", ".webp", ".bmp", ".gif",
            # Raw formats
            ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng",
            # Audio (for ENF analysis)
            ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
            # Video (for ENF and visual analysis)
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        ]

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in WuAnalyzer.get_supported_formats()
