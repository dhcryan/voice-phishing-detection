"""
AI Voice Phishing Detection System - Configuration
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # === Application ===
    APP_NAME: str = "AI Voice Phishing Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    
    # === API Server ===
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    
    # === Audio Processing ===
    AUDIO_SAMPLE_RATE: int = Field(default=16000)
    AUDIO_MAX_DURATION: int = Field(default=60)  # seconds
    AUDIO_MIN_DURATION: float = Field(default=0.5)  # seconds
    SUPPORTED_AUDIO_FORMATS: List[str] = Field(default=["wav", "mp3", "flac", "ogg", "m4a"])
    
    # === Detection Models ===
    DETECTION_MODEL_TYPE: str = Field(default="simple")  # aasist, rawnet2, ecapa, simple
    DETECTION_MODEL_PATH: str = Field(default="checkpoints/simple_detector_best.pt")
    DETECTION_BATCH_SIZE: int = Field(default=8)
    DETECTION_THRESHOLD: float = Field(default=0.5)
    
    # === AudioSeal Watermark Detection ===
    ENABLE_WATERMARK_DETECTION: bool = Field(default=True)
    AUDIOSEAL_MODEL_PATH: Optional[str] = Field(default=None)
    
    # === Risk Scoring ===
    RISK_LOW_THRESHOLD: float = Field(default=0.3)
    RISK_HIGH_THRESHOLD: float = Field(default=0.7)
    WATERMARK_RISK_WEIGHT: float = Field(default=0.2)
    ACOUSTIC_ANOMALY_WEIGHT: float = Field(default=0.1)
    
    # === LLM Settings ===
    OPENAI_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_MAX_TOKENS: int = Field(default=2048)
    LLM_TIMEOUT: int = Field(default=60)
    
    # === RAG Settings ===
    VECTOR_DB_TYPE: str = Field(default="faiss")  # faiss, chromadb, qdrant
    VECTOR_DB_PATH: str = Field(default="data/vectors")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    RAG_TOP_K: int = Field(default=5)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.7)
    LEGAL_DOCS_PATH: str = Field(default="data/legal_docs")
    
    # === Langfuse Monitoring ===
    LANGFUSE_ENABLED: bool = Field(default=True)
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com")
    
    # === MLflow Tracking ===
    MLFLOW_ENABLED: bool = Field(default=True)
    MLFLOW_TRACKING_URI: str = Field(default="mlruns")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="voice-phishing-detection")
    
    # === Logging ===
    LOG_LEVEL: str = Field(default="INFO")
    LOG_PATH: str = Field(default="logs")
    LOG_ROTATION: str = Field(default="10 MB")
    LOG_RETENTION: str = Field(default="30 days")
    
    # === Security ===
    ENABLE_PROMPT_INJECTION_DETECTION: bool = Field(default=True)
    ENABLE_PII_FILTERING: bool = Field(default=True)
    MAX_REQUEST_SIZE_MB: int = Field(default=50)
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=60)  # seconds
    
    # === Streamlit Frontend ===
    STREAMLIT_PORT: int = Field(default=8501)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"  # .env의 추가 변수 무시
    }


# === Prompt Templates ===
class PromptTemplates:
    """Prompt templates for LLM interactions with version tracking"""
    
    DETECTION_SUMMARY_V1 = """당신은 음성 보이스피싱 탐지 전문가입니다.

## 탐지 결과
- 탐지 모델: {model_name}
- 가짜 음성 확률: {fake_probability:.2%}
- 리스크 레벨: {risk_level}
- 음향 이상 지표: {acoustic_anomalies}

## 요청
위 탐지 결과를 바탕으로:
1. 이 음성이 왜 {risk_level} 위험으로 분류되었는지 설명하세요.
2. 주요 탐지 근거를 일반인이 이해할 수 있게 설명하세요.
3. 권장 조치사항을 알려주세요.

간결하고 명확하게 한국어로 답변하세요."""

    DETECTION_SUMMARY_V2 = """## 역할
보이스피싱 탐지 AI 분석가

## 입력 데이터
| 항목 | 값 |
|------|-----|
| 모델 | {model_name} |
| 가짜 확률 | {fake_probability:.2%} |
| 리스크 | {risk_level} |
| 음향 이상 | {acoustic_anomalies} |
| 워터마크 | {watermark_detected} |

## 출력 형식
다음 형식으로 응답하세요:

### 📊 분석 요약
(위험도와 핵심 탐지 결과 2-3문장)

### 🔍 탐지 근거
(기술적 근거를 일반인 언어로 설명)

### ⚠️ 주의사항
(사용자가 취해야 할 조치)"""

    RAG_LEGAL_QUERY_V1 = """## 역할
당신은 보이스피싱 관련 법률 전문 AI입니다.

## 탐지 컨텍스트
- 리스크 레벨: {risk_level}
- 탐지 결과 요약: {detection_summary}

## 관련 법령
{legal_context}

## 사용자 질문
{user_question}

## 지시사항
1. 반드시 위 법령에서 관련 조항을 인용하세요.
2. 조항 번호와 내용을 명시하세요.
3. 구체적인 대응 절차를 안내하세요.
4. 피해 구제 방법이 있다면 포함하세요.

인용 형식: [법령명 제X조]"""

    CHECKLIST_GENERATION = """## 역할
보이스피싱 대응 체크리스트 생성기

## 상황
- 리스크 레벨: {risk_level}
- 탐지 확률: {fake_probability:.2%}
- 추정 유형: {fraud_type}

## 출력
해당 상황에 맞는 대응 체크리스트를 생성하세요:
1. 즉시 조치사항 (긴급)
2. 신고 절차
3. 증거 보전 방법
4. 피해 최소화 조치
5. 향후 예방 조치

각 항목은 구체적이고 실행 가능해야 합니다."""


# === Model Configurations ===
MODEL_CONFIGS = {
    "aasist": {
        "name": "AASIST",
        "architecture": "Graph Attention Network",
        "input_type": "raw_waveform",
        "sample_rate": 16000,
        "pretrained_url": "https://github.com/clovaai/aasist",
        "description": "RawNet2-based with graph attention for anti-spoofing"
    },
    "rawnet2": {
        "name": "RawNet2", 
        "architecture": "End-to-End CNN",
        "input_type": "raw_waveform",
        "sample_rate": 16000,
        "pretrained_url": "https://github.com/asvspoof-challenge/2021",
        "description": "End-to-end raw waveform anti-spoofing"
    },
    "ecapa": {
        "name": "ECAPA-TDNN",
        "architecture": "TDNN with Attentive Statistics",
        "input_type": "mel_spectrogram",
        "sample_rate": 16000,
        "pretrained_url": "speechbrain/spkrec-ecapa-voxceleb",
        "description": "Speaker embedding based detection"
    },
    "wav2vec2": {
        "name": "Wav2Vec2-VIB",
        "architecture": "Transformer with VIB",
        "input_type": "raw_waveform",
        "sample_rate": 16000,
        "pretrained_url": "facebook/wav2vec2-base-960h",
        "description": "Transfer learning with variational information bottleneck"
    }
}


# === Risk Level Definitions ===
RISK_LEVELS = {
    "LOW": {
        "label": "저위험",
        "color": "#28a745",
        "description": "정상 음성으로 판단되나 주의가 필요합니다.",
        "actions": ["통화 내용 확인", "발신자 신원 검증"]
    },
    "MEDIUM": {
        "label": "중위험", 
        "color": "#ffc107",
        "description": "의심스러운 요소가 감지되었습니다. 주의가 필요합니다.",
        "actions": ["즉시 통화 종료 권장", "공식 채널로 재확인", "가족/지인에게 알림"]
    },
    "HIGH": {
        "label": "고위험",
        "color": "#dc3545", 
        "description": "가짜 음성(합성/변조)으로 판단됩니다. 보이스피싱 가능성이 높습니다.",
        "actions": ["즉시 통화 종료", "금융거래 중단", "경찰청(112) 신고", "금융감독원(1332) 신고"]
    }
}


# === Legal Document Categories ===
LEGAL_DOC_CATEGORIES = {
    "criminal": {
        "name": "형법",
        "articles": ["제347조 (사기)", "제347조의2 (컴퓨터등 사용사기)"]
    },
    "electronic_finance": {
        "name": "전자금융거래법",
        "articles": ["제6조", "제9조", "제10조"]
    },
    "telecom_fraud": {
        "name": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법",
        "articles": ["제3조", "제4조", "제5조", "제13조"]
    },
    "aggravated_punishment": {
        "name": "특정경제범죄 가중처벌 등에 관한 법률",
        "articles": ["제3조"]
    }
}


# Global settings instance
settings = Settings()
prompts = PromptTemplates()
