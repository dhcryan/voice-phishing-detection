"""
Voice Phishing Detection - Detection Module Init
"""
from .detector import (
    DetectionResult,
    AudioPreprocessor,
    BaseDetector,
    RawNet2Detector,
    AASISTDetector,
    ECAPATDNNDetector,
    get_detector
)
from .watermark import (
    WatermarkResult,
    AudioSealDetector,
    WatermarkDetectorFallback,
    get_watermark_detector
)

__all__ = [
    "DetectionResult",
    "AudioPreprocessor",
    "BaseDetector",
    "RawNet2Detector",
    "AASISTDetector",
    "ECAPATDNNDetector",
    "get_detector",
    "WatermarkResult",
    "AudioSealDetector",
    "WatermarkDetectorFallback",
    "get_watermark_detector"
]
