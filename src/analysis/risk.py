"""
Risk Scoring Module
"""
from typing import Dict, Any
from src.config import settings

class RiskScorer:
    def calculate_risk(self, 
                       fake_probability: float, 
                       watermark_detected: bool, 
                       acoustic_anomalies: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate risk score based on multiple signals.
        
        Formula:
        Base Score = Fake Probability
        + Watermark Weight (if detected)
        + Acoustic Anomaly Weight (if anomalies found)
        """
        
        score = fake_probability
        
        # Add weight if watermark is detected (high indicator of synthetic speech)
        if watermark_detected:
            score += settings.WATERMARK_RISK_WEIGHT
            
        # Add weight for acoustic anomalies (simplified logic for now)
        # In a real scenario, we would analyze jitter, shimmer, etc.
        anomaly_score = acoustic_anomalies.get("anomaly_score", 0.0)
        if anomaly_score > 0.5:
            score += settings.ACOUSTIC_ANOMALY_WEIGHT
            
        # Cap score at 1.0
        score = min(score, 1.0)
        
        # Determine Risk Level
        if score >= settings.RISK_HIGH_THRESHOLD:
            level = "HIGH"
            color = "red"
        elif score >= settings.RISK_LOW_THRESHOLD:
            level = "MEDIUM"
            color = "yellow"
        else:
            level = "LOW"
            color = "green"
            
        return {
            "score": score,
            "level": level,
            "color": color,
            "factors": {
                "fake_probability": fake_probability,
                "watermark_detected": watermark_detected,
                "acoustic_anomaly_score": anomaly_score
            }
        }

risk_scorer = RiskScorer()
