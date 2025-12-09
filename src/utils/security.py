"""
Voice Phishing Detection - Security Utilities
Prompt injection detection, PII filtering, rate limiting
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """Result of security check"""
    is_safe: bool
    violations: List[str]
    sanitized_input: Optional[str]
    risk_score: float


class PromptInjectionDetector:
    """Detect potential prompt injection attacks"""
    
    # Patterns that may indicate prompt injection
    INJECTION_PATTERNS = [
        # Direct instruction override attempts
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"forget\s+(all\s+)?(previous|above|prior)",
        r"override\s+(all\s+)?(previous|above|prior)",
        
        # Role manipulation
        r"you\s+are\s+(now|actually|really)\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if|though)",
        r"switch\s+(to|into)\s+",
        
        # System prompt extraction
        r"(what|show|reveal|tell|display)\s+(is|me|your)\s+(system|initial)\s+(prompt|instruction)",
        r"print\s+(your\s+)?(system|initial)\s+(prompt|instruction)",
        r"repeat\s+(your\s+)?(system|initial)",
        
        # Jailbreak attempts
        r"(do|dan|aim|dev)\s*mode",
        r"no\s+restrictions",
        r"without\s+(any\s+)?(restrictions|limitations|filters)",
        r"bypass\s+(safety|content|ethical)",
        
        # Command injection
        r"```(bash|shell|cmd|powershell|python|javascript)",
        r"\$\([^)]+\)",
        r"`[^`]+`",
        
        # Korean injection patterns
        r"이전\s*(지시|명령|프롬프트).*무시",
        r"시스템\s*프롬프트.*보여",
        r"역할.*바꿔",
        r"제한.*없이",
    ]
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    def detect(self, text: str) -> Tuple[bool, List[str], float]:
        """
        Detect prompt injection attempts
        Returns: (is_injection, matched_patterns, risk_score)
        """
        matches = []
        
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                matches.append(self.INJECTION_PATTERNS[i])
        
        # Calculate risk score based on matches
        risk_score = min(len(matches) / 3, 1.0)
        is_injection = risk_score >= self.threshold
        
        return is_injection, matches, risk_score
    
    def sanitize(self, text: str) -> str:
        """Sanitize text by removing potential injection patterns"""
        sanitized = text
        
        for pattern in self.patterns:
            sanitized = pattern.sub("[FILTERED]", sanitized)
        
        return sanitized


class PIIFilter:
    """Filter personally identifiable information"""
    
    PII_PATTERNS = {
        "korean_rrn": {
            # Korean Resident Registration Number: 000000-0000000
            "pattern": r"\b\d{6}[-\s]?\d{7}\b",
            "replacement": "[주민번호]",
            "severity": "HIGH"
        },
        "phone_number": {
            # Korean phone numbers
            "pattern": r"\b(01[0-9]|02|0[3-9][0-9])[-\s]?\d{3,4}[-\s]?\d{4}\b",
            "replacement": "[전화번호]",
            "severity": "MEDIUM"
        },
        "email": {
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "replacement": "[이메일]",
            "severity": "MEDIUM"
        },
        "credit_card": {
            "pattern": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "replacement": "[카드번호]",
            "severity": "HIGH"
        },
        "bank_account": {
            # Korean bank account formats
            "pattern": r"\b\d{3,4}[-\s]?\d{2,4}[-\s]?\d{4,6}\b",
            "replacement": "[계좌번호]",
            "severity": "HIGH"
        },
        "korean_name": {
            # Korean names (2-4 characters, common surnames)
            "pattern": r"\b(김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전)[가-힣]{1,3}\b",
            "replacement": "[이름]",
            "severity": "LOW"
        },
        "address": {
            # Korean address patterns
            "pattern": r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)[시도]?\s*[가-힣]+[시군구]\s*[가-힣]+[읍면동로길]\s*\d+",
            "replacement": "[주소]",
            "severity": "MEDIUM"
        }
    }
    
    def __init__(self, filter_high: bool = True, filter_medium: bool = True, filter_low: bool = False):
        self.filter_levels = []
        if filter_high:
            self.filter_levels.append("HIGH")
        if filter_medium:
            self.filter_levels.append("MEDIUM")
        if filter_low:
            self.filter_levels.append("LOW")
        
        self.compiled_patterns = {
            name: re.compile(info["pattern"])
            for name, info in self.PII_PATTERNS.items()
        }
    
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text"""
        detections = []
        
        for name, pattern in self.compiled_patterns.items():
            info = self.PII_PATTERNS[name]
            
            if info["severity"] not in self.filter_levels:
                continue
            
            matches = pattern.findall(text)
            if matches:
                detections.append({
                    "type": name,
                    "count": len(matches),
                    "severity": info["severity"],
                    "matches": matches[:3]  # Limit for logging
                })
        
        return detections
    
    def filter(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Filter PII from text"""
        filtered_text = text
        detections = []
        
        for name, pattern in self.compiled_patterns.items():
            info = self.PII_PATTERNS[name]
            
            if info["severity"] not in self.filter_levels:
                continue
            
            matches = pattern.findall(filtered_text)
            if matches:
                detections.append({
                    "type": name,
                    "count": len(matches),
                    "severity": info["severity"]
                })
                filtered_text = pattern.sub(info["replacement"], filtered_text)
        
        return filtered_text, detections


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if request is allowed
        Returns: (is_allowed, remaining_requests)
        """
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Get or create request history for client
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Filter old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if t > window_start
        ]
        
        # Check limit
        current_count = len(self.requests[client_id])
        remaining = max(0, self.max_requests - current_count)
        
        if current_count >= self.max_requests:
            return False, 0
        
        # Record request
        self.requests[client_id].append(now)
        
        return True, remaining - 1
    
    def reset(self, client_id: str):
        """Reset rate limit for client"""
        if client_id in self.requests:
            del self.requests[client_id]


class SecurityManager:
    """Unified security manager"""
    
    def __init__(
        self,
        enable_injection_detection: bool = True,
        enable_pii_filtering: bool = True,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60
    ):
        self.injection_detector = PromptInjectionDetector() if enable_injection_detection else None
        self.pii_filter = PIIFilter() if enable_pii_filtering else None
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
    
    def check_request(
        self,
        text: str,
        client_id: str = "anonymous"
    ) -> SecurityCheckResult:
        """Perform all security checks on input"""
        violations = []
        risk_score = 0.0
        sanitized = text
        
        # Rate limiting
        is_allowed, remaining = self.rate_limiter.is_allowed(client_id)
        if not is_allowed:
            violations.append(f"Rate limit exceeded for client {client_id}")
            return SecurityCheckResult(
                is_safe=False,
                violations=violations,
                sanitized_input=None,
                risk_score=1.0
            )
        
        # Prompt injection check
        if self.injection_detector:
            is_injection, patterns, inj_risk = self.injection_detector.detect(text)
            if is_injection:
                violations.append(f"Prompt injection detected: {len(patterns)} patterns matched")
                risk_score = max(risk_score, inj_risk)
                sanitized = self.injection_detector.sanitize(sanitized)
        
        # PII filtering
        if self.pii_filter:
            sanitized, pii_detections = self.pii_filter.filter(sanitized)
            if pii_detections:
                high_severity = [d for d in pii_detections if d["severity"] == "HIGH"]
                if high_severity:
                    violations.append(f"High-severity PII detected: {len(high_severity)} types")
                    risk_score = max(risk_score, 0.7)
        
        is_safe = len(violations) == 0 or risk_score < 0.5
        
        return SecurityCheckResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_input=sanitized,
            risk_score=risk_score
        )


# Global security manager
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager"""
    global _security_manager
    if _security_manager is None:
        from src.config import settings
        _security_manager = SecurityManager(
            enable_injection_detection=settings.ENABLE_PROMPT_INJECTION_DETECTION,
            enable_pii_filtering=settings.ENABLE_PII_FILTERING,
            rate_limit_requests=settings.RATE_LIMIT_REQUESTS,
            rate_limit_window=settings.RATE_LIMIT_WINDOW
        )
    return _security_manager
