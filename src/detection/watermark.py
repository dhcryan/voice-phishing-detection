"""
Voice Phishing Detection - Watermark Detection Module
AudioSeal integration for detecting AI-generated audio watermarks
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class WatermarkResult:
    """Result from watermark detection"""
    has_watermark: bool
    confidence: float
    watermark_type: Optional[str]
    localization: Optional[np.ndarray]  # Frame-level watermark presence
    processing_time_ms: float
    metadata: Dict[str, Any]


class AudioSealDetector:
    """
    AudioSeal Watermark Detector
    Reference: https://github.com/facebookresearch/audioseal
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.sample_rate = 16000
        
    def load_model(self):
        """Load AudioSeal detector model"""
        try:
            from audioseal import AudioSeal
            
            self.detector = AudioSeal.load_detector("audioseal_detector_16bits")
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            
            logger.info("AudioSeal detector loaded successfully")
            
        except ImportError:
            logger.warning(
                "AudioSeal not installed. Watermark detection will be disabled. "
                "Install with: pip install audioseal"
            )
            self.detector = None
        except Exception as e:
            logger.error(f"Failed to load AudioSeal: {e}")
            self.detector = None
            
    def detect(self, audio_path: str) -> WatermarkResult:
        """Detect watermark in audio file"""
        import time
        import librosa
        
        start_time = time.time()
        
        if self.detector is None:
            return WatermarkResult(
                has_watermark=False,
                confidence=0.0,
                watermark_type=None,
                localization=None,
                processing_time_ms=0.0,
                metadata={"error": "AudioSeal not loaded"}
            )
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0).float()
        waveform_tensor = waveform_tensor.to(self.device)
        
        # Detect watermark
        with torch.no_grad():
            result, message = self.detector.detect_watermark(
                waveform_tensor,
                sample_rate=self.sample_rate
            )
        
        # Process results
        detection_prob = result.item() if torch.is_tensor(result) else result
        has_watermark = detection_prob > 0.5
        
        # Frame-level localization if available
        localization = None
        if hasattr(self.detector, 'get_watermark_localization'):
            with torch.no_grad():
                loc = self.detector.get_watermark_localization(
                    waveform_tensor,
                    sample_rate=self.sample_rate
                )
                localization = loc.cpu().numpy() if torch.is_tensor(loc) else None
        
        processing_time = (time.time() - start_time) * 1000
        
        return WatermarkResult(
            has_watermark=has_watermark,
            confidence=detection_prob,
            watermark_type="AudioSeal" if has_watermark else None,
            localization=localization,
            processing_time_ms=processing_time,
            metadata={
                "message_bits": message.tolist() if message is not None else None,
                "model": "audioseal_detector_16bits"
            }
        )


class WatermarkDetectorFallback:
    """
    Fallback watermark detector using spectral analysis
    For when AudioSeal is not available
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def detect(self, audio_path: str) -> WatermarkResult:
        """Detect potential watermarks using spectral analysis"""
        import time
        import librosa
        
        start_time = time.time()
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Analyze spectral characteristics that might indicate synthetic audio
        stft = librosa.stft(waveform)
        magnitude = np.abs(stft)
        
        # Check for unusual periodic patterns (common in watermarked audio)
        spectral_flux = np.sqrt(np.mean(np.diff(magnitude, axis=1) ** 2))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=waveform))
        
        # High spectral flatness + low flux can indicate watermarked/synthetic audio
        suspicion_score = spectral_flatness * 0.5 + (1 - min(spectral_flux, 1)) * 0.5
        
        # Very rough heuristic
        has_watermark = suspicion_score > 0.7
        
        processing_time = (time.time() - start_time) * 1000
        
        return WatermarkResult(
            has_watermark=has_watermark,
            confidence=suspicion_score,
            watermark_type="Spectral Analysis (Heuristic)" if has_watermark else None,
            localization=None,
            processing_time_ms=processing_time,
            metadata={
                "method": "spectral_analysis",
                "spectral_flux": spectral_flux,
                "spectral_flatness": spectral_flatness,
                "note": "This is a heuristic fallback, not actual watermark detection"
            }
        )


def get_watermark_detector(use_audioseal: bool = True, device: str = "cuda"):
    """Factory function to get watermark detector"""
    if use_audioseal:
        detector = AudioSealDetector(device)
        detector.load_model()
        if detector.detector is not None:
            return detector
    
    logger.info("Using fallback spectral analysis watermark detector")
    return WatermarkDetectorFallback()
