import pytest
import os
import numpy as np
import tempfile
from unittest.mock import MagicMock, patch
from wu.analyzer import WuAnalyzer
from wu.state import DimensionResult, DimensionState, Confidence

def test_cross_modal_coordination():
    """
    Verify that WuAnalyzer correctly dispatches image tools to video frames
    and handles cross-modal corroboration.
    """
    # 1. Setup mocks
    analyzer = WuAnalyzer(
        enable_video=True,
        enable_visual=True,
        enable_audio=True,
        parallel=False
    )
    
    # Mock VideoAnalyzer
    mock_video_analyzer = MagicMock()
    # Return 1 frame (800x600 green)
    mock_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    mock_frame[:, :, 1] = 255
    mock_video_analyzer.iter_frames.return_value = [mock_frame]
    # Return 1 audio sample
    mock_video_analyzer.iter_audio_samples.return_value = [b"fake_audio_data"]
    
    # Mock container analysis result
    mock_video_analyzer.analyze.return_value = DimensionResult(
        dimension="video",
        state=DimensionState.CONSISTENT,
        confidence=Confidence.HIGH,
        evidence=[]
    )
    
    analyzer._video_analyzer = mock_video_analyzer
    
    # Mock VisualAnalyzer to return SUSPICIOUS
    mock_visual_analyzer = MagicMock()
    mock_visual_analyzer.analyze.return_value = DimensionResult(
        dimension="visual",
        state=DimensionState.SUSPICIOUS,
        confidence=Confidence.MEDIUM,
        evidence=[MagicMock(finding="Splice", explanation="Splicing detected")]
    )
    analyzer._visual_analyzer = mock_visual_analyzer
    
    # Mock AudioAnalyzer to return CONSISTENT
    mock_audio_analyzer = MagicMock()
    mock_audio_analyzer.analyze.return_value = DimensionResult(
        dimension="audio",
        state=DimensionState.CONSISTENT,
        confidence=Confidence.HIGH,
        evidence=[]
    )
    analyzer._audio_analyzer = mock_audio_analyzer
    
    # Create a real empty file with .mp4 suffix
    with tempfile.TemporaryDirectory() as td:
        fake_video_raw = os.path.join(td, "test.mp4")
        with open(fake_video_raw, "wb") as f:
            f.write(b"minimal_mp4_header")

        # 2. Run analysis
        with patch("wu.analyzer.WuAnalyzer._compute_hash", return_value="abc"):
            result = analyzer.analyze(fake_video_raw)
            
            # 3. Verify Coordination
            # Video state should be SUSPICIOUS because of visual anomaly
            assert result.video.state == DimensionState.SUSPICIOUS
            # Audio should be present
            assert result.audio is not None
            assert result.audio.state == DimensionState.CONSISTENT
            
            # Check for cross-modal corroboration message
            explanations = [e.finding for e in result.video.evidence]
            assert "Cross-modal anomaly" in explanations
            print("\n[PASSED] Cross-modal coordination verified.")

if __name__ == "__main__":
    try:
        test_cross_modal_coordination()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
