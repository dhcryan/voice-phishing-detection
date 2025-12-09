"""
Voice Phishing Detection - Risk Scoring Module
Combines detection results, watermark analysis, and acoustic features
"""
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    risk_level: RiskLevel
    risk_score: float  # 0.0 - 1.0
    fake_probability: float
    watermark_detected: bool
    watermark_confidence: float
    acoustic_anomaly_score: float
    contributing_factors: List[Dict[str, Any]]
    recommendations: List[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


class RiskScorer:
    """
    Risk scoring engine combining multiple signals
    """
    
    def __init__(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
        detection_weight: float = 0.6,
        watermark_weight: float = 0.2,
        acoustic_weight: float = 0.2
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.detection_weight = detection_weight
        self.watermark_weight = watermark_weight
        self.acoustic_weight = acoustic_weight
        
        # Acoustic feature thresholds for anomaly detection
        self.acoustic_thresholds = {
            "flatness_mean": 0.5,  # High flatness indicates synthetic
            "zcr_std": 0.01,  # Low variation in ZCR
            "spectral_centroid_std": 100,  # Low spectral variation
        }
        
    def compute_acoustic_anomaly_score(
        self, 
        acoustic_features: Dict[str, float]
    ) -> float:
        """
        Compute anomaly score based on acoustic features
        Higher score = more synthetic/anomalous
        """
        scores = []
        
        # Spectral flatness (high = more synthetic)
        if "flatness_mean" in acoustic_features:
            flatness = acoustic_features["flatness_mean"]
            flatness_score = min(flatness / self.acoustic_thresholds["flatness_mean"], 1.0)
            scores.append(flatness_score)
        
        # ZCR variation (low = more synthetic)
        if "zcr_std" in acoustic_features:
            zcr_std = acoustic_features["zcr_std"]
            zcr_score = max(0, 1 - zcr_std / self.acoustic_thresholds["zcr_std"])
            scores.append(zcr_score)
        
        # Spectral centroid variation (low = more synthetic)
        if "spectral_centroid_std" in acoustic_features:
            sc_std = acoustic_features["spectral_centroid_std"]
            sc_score = max(0, 1 - sc_std / self.acoustic_thresholds["spectral_centroid_std"])
            scores.append(sc_score)
        
        if not scores:
            return 0.0
            
        return np.mean(scores)
    
    def compute_risk_score(
        self,
        fake_probability: float,
        watermark_detected: bool,
        watermark_confidence: float,
        acoustic_features: Dict[str, float]
    ) -> float:
        """
        Compute weighted risk score from multiple signals
        """
        # Detection component
        detection_score = fake_probability
        
        # Watermark component (if detected, increases risk but could be legitimate)
        if watermark_detected:
            # Watermark presence increases suspicion but with lower weight
            # because legitimate AI-generated content may have watermarks
            watermark_score = watermark_confidence * 0.8
        else:
            watermark_score = 0.0
        
        # Acoustic anomaly component
        acoustic_score = self.compute_acoustic_anomaly_score(acoustic_features)
        
        # Weighted combination
        risk_score = (
            self.detection_weight * detection_score +
            self.watermark_weight * watermark_score +
            self.acoustic_weight * acoustic_score
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < self.low_threshold:
            return RiskLevel.LOW
        elif risk_score < self.high_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def get_contributing_factors(
        self,
        fake_probability: float,
        watermark_detected: bool,
        watermark_confidence: float,
        acoustic_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify and explain contributing factors to risk"""
        factors = []
        
        # Detection model result
        if fake_probability > 0.5:
            factors.append({
                "factor": "AI 탐지 모델 결과",
                "severity": "HIGH" if fake_probability > 0.8 else "MEDIUM",
                "detail": f"가짜 음성 확률 {fake_probability:.1%}로 탐지됨",
                "contribution": fake_probability * self.detection_weight
            })
        else:
            factors.append({
                "factor": "AI 탐지 모델 결과",
                "severity": "LOW",
                "detail": f"정상 음성으로 판단됨 (가짜 확률: {fake_probability:.1%})",
                "contribution": fake_probability * self.detection_weight
            })
        
        # Watermark detection
        if watermark_detected:
            factors.append({
                "factor": "워터마크 탐지",
                "severity": "MEDIUM",
                "detail": f"AI 생성 워터마크 감지 (신뢰도: {watermark_confidence:.1%})",
                "contribution": watermark_confidence * self.watermark_weight,
                "note": "워터마크는 합법적인 AI 생성 콘텐츠에도 포함될 수 있음"
            })
        
        # Acoustic anomalies
        acoustic_score = self.compute_acoustic_anomaly_score(acoustic_features)
        if acoustic_score > 0.5:
            anomaly_details = []
            if acoustic_features.get("flatness_mean", 0) > 0.3:
                anomaly_details.append("높은 스펙트럼 평탄도 (합성음 특징)")
            if acoustic_features.get("zcr_std", 1) < 0.02:
                anomaly_details.append("낮은 영점교차율 변동 (부자연스러움)")
            
            factors.append({
                "factor": "음향 이상 패턴",
                "severity": "MEDIUM" if acoustic_score < 0.8 else "HIGH",
                "detail": "; ".join(anomaly_details) if anomaly_details else "음향 특성 이상 감지",
                "contribution": acoustic_score * self.acoustic_weight
            })
        
        return factors
    
    def get_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get recommended actions based on risk level"""
        recommendations = {
            RiskLevel.LOW: [
                "현재 음성은 정상으로 판단되나, 의심스러운 내용이 있다면 주의하세요.",
                "발신자의 신원을 공식 채널로 확인하는 것을 권장합니다.",
                "개인정보나 금융정보 요청 시 즉시 통화를 종료하세요."
            ],
            RiskLevel.MEDIUM: [
                "⚠️ 의심스러운 요소가 감지되었습니다.",
                "통화를 종료하고 공식 대표번호로 재확인하세요.",
                "금융거래나 개인정보 제공을 보류하세요.",
                "가족이나 지인에게 상황을 알리세요.",
                "금융감독원(1332)에 상담을 요청할 수 있습니다."
            ],
            RiskLevel.HIGH: [
                "🚨 가짜 음성(보이스피싱)으로 판단됩니다!",
                "즉시 통화를 종료하세요.",
                "어떠한 금융거래도 진행하지 마세요.",
                "경찰청(112)에 신고하세요.",
                "금융감독원(1332)에 피해 상담 및 지급정지를 요청하세요.",
                "관련 금융기관 고객센터에 즉시 연락하세요.",
                "통화 녹음 및 관련 증거를 보관하세요."
            ]
        }
        
        return recommendations.get(risk_level, recommendations[RiskLevel.MEDIUM])
    
    def assess_risk(
        self,
        fake_probability: float,
        watermark_detected: bool = False,
        watermark_confidence: float = 0.0,
        acoustic_features: Optional[Dict[str, float]] = None
    ) -> RiskAssessment:
        """
        Perform complete risk assessment
        """
        if acoustic_features is None:
            acoustic_features = {}
        
        # Compute scores
        acoustic_anomaly_score = self.compute_acoustic_anomaly_score(acoustic_features)
        risk_score = self.compute_risk_score(
            fake_probability,
            watermark_detected,
            watermark_confidence,
            acoustic_features
        )
        
        # Determine level
        risk_level = self.determine_risk_level(risk_score)
        
        # Get factors and recommendations
        contributing_factors = self.get_contributing_factors(
            fake_probability,
            watermark_detected,
            watermark_confidence,
            acoustic_features
        )
        recommendations = self.get_recommendations(risk_level)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            fake_probability=fake_probability,
            watermark_detected=watermark_detected,
            watermark_confidence=watermark_confidence,
            acoustic_anomaly_score=acoustic_anomaly_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            raw_data={
                "acoustic_features": acoustic_features,
                "weights": {
                    "detection": self.detection_weight,
                    "watermark": self.watermark_weight,
                    "acoustic": self.acoustic_weight
                }
            }
        )


# Convenience functions for risk level display
def get_risk_color(risk_level: RiskLevel) -> str:
    """Get display color for risk level"""
    colors = {
        RiskLevel.LOW: "#28a745",
        RiskLevel.MEDIUM: "#ffc107",
        RiskLevel.HIGH: "#dc3545"
    }
    return colors.get(risk_level, "#6c757d")


def get_risk_label(risk_level: RiskLevel) -> str:
    """Get Korean label for risk level"""
    labels = {
        RiskLevel.LOW: "저위험",
        RiskLevel.MEDIUM: "중위험", 
        RiskLevel.HIGH: "고위험"
    }
    return labels.get(risk_level, "알 수 없음")


def get_risk_emoji(risk_level: RiskLevel) -> str:
    """Get emoji for risk level"""
    emojis = {
        RiskLevel.LOW: "✅",
        RiskLevel.MEDIUM: "⚠️",
        RiskLevel.HIGH: "🚨"
    }
    return emojis.get(risk_level, "❓")
