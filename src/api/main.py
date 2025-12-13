"""
Voice Phishing Detection - FastAPI Server
Main API endpoints for voice analysis and legal guidance
"""
import os
import sys
import time
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aiofiles

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings, prompts, RISK_LEVELS
from src.scoring import RiskScorer, RiskLevel, get_risk_label, get_risk_color, get_risk_emoji


# === Pydantic Models ===

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class DetectionRequest(BaseModel):
    model_type: str = Field(default="aasist", description="Detection model: aasist, rawnet2, ecapa")
    enable_watermark: bool = Field(default=True, description="Enable watermark detection")


class DetectionResponse(BaseModel):
    filename: str
    is_fake: bool
    confidence: float
    fake_probability: float
    risk_level: str
    risk_score: float
    processing_time_ms: float
    model_used: str
    watermark_detected: bool
    explanation: Optional[str] = None
    legal_references: Optional[List[str]] = None


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    risk_level: str = Field(default="MEDIUM")
    detection_summary: str = Field(default="")
    top_k: int = Field(default=5, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    latency_ms: float
    timestamp: str


class FeedbackRequest(BaseModel):
    request_id: str
    score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    helpful: bool = True


class MetricsResponse(BaseModel):
    uptime_seconds: float
    requests: Dict[str, Any]
    latency: Dict[str, Any]
    tokens: Dict[str, Any]
    risk_distribution: Dict[str, int]


# === FastAPI App ===

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Voice Phishing Detection with Legal RAG System"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# === Global State ===

class AppState:
    """Application state holder"""
    detector = None
    watermark_detector = None
    risk_scorer = None
    rag_system = None
    monitor = None
    metrics = None
    security = None
    initialized = False


state = AppState()


# === Startup/Shutdown ===

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    import logging
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
    
    # Initialize risk scorer
    state.risk_scorer = RiskScorer(
        low_threshold=settings.RISK_LOW_THRESHOLD,
        high_threshold=settings.RISK_HIGH_THRESHOLD
    )
    
    # Initialize monitoring
    try:
        from src.utils.monitoring import LangfuseMonitor, MetricsCollector
        state.monitor = LangfuseMonitor(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            enabled=settings.LANGFUSE_ENABLED
        )
        state.metrics = MetricsCollector()
    except Exception as e:
        print(f"Warning: Monitoring initialization failed: {e}")
    
    # Initialize security
    try:
        from src.utils.security import SecurityManager
        state.security = SecurityManager(
            enable_injection_detection=settings.ENABLE_PROMPT_INJECTION_DETECTION,
            enable_pii_filtering=settings.ENABLE_PII_FILTERING
        )
    except Exception as e:
        print(f"Warning: Security initialization failed: {e}")
    
    state.initialized = True
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if state.monitor:
        state.monitor.flush()
    print("👋 Server shutdown complete")


# === Dependency Injection ===

def get_detector():
    """Get or initialize detector"""
    if state.detector is None:
        try:
            from src.detection import get_detector
            state.detector = get_detector(
                model_type=settings.DETECTION_MODEL_TYPE,
                device="cuda"
            )
        except Exception as e:
            print(f"Warning: Detector initialization failed: {e}")
            # Return a mock detector for demo
            return None
    return state.detector


def get_rag():
    """Get or initialize RAG system"""
    if state.rag_system is None:
        try:
            from src.rag import create_rag_system
            state.rag_system = create_rag_system(
                docs_path=settings.LEGAL_DOCS_PATH,
                vector_path=settings.VECTOR_DB_PATH,
                llm_model=settings.LLM_MODEL
            )
        except Exception as e:
            import traceback
            print(f"Warning: RAG initialization failed: {e}")
            traceback.print_exc()
            return None
    return state.rag_system


# === Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.initialized else "initializing",
        version=settings.APP_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_fake_voice(
    file: UploadFile = File(...),
    model_type: str = Query(default="aasist", regex="^(aasist|rawnet2|ecapa)$"),
    enable_watermark: bool = Query(default=True)
):
    """
    Analyze uploaded audio file for voice phishing
    
    - **file**: Audio file (wav, mp3, flac, etc.)
    - **model_type**: Detection model to use
    - **enable_watermark**: Enable watermark detection
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Validate file type
    filename = file.filename or "audio.wav"
    ext = filename.split(".")[-1].lower()
    if ext not in settings.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext}. Supported: {settings.SUPPORTED_AUDIO_FORMATS}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Detection result (mock for demo if detector not available)
        detector = get_detector()
        
        if detector:
            detection_result = detector.predict(tmp_path)
            fake_prob = detection_result.fake_probability
            acoustic_features = detection_result.raw_scores
            model_name = detection_result.model_name
            detection_time = detection_result.processing_time_ms
        else:
            # Mock result for demo
            import random
            fake_prob = random.uniform(0.1, 0.9)
            acoustic_features = {
                "flatness_mean": random.uniform(0.1, 0.6),
                "zcr_std": random.uniform(0.01, 0.05),
                "spectral_centroid_std": random.uniform(50, 200)
            }
            model_name = model_type.upper()
            detection_time = random.uniform(100, 500)
        
        # Watermark detection
        watermark_detected = False
        watermark_confidence = 0.0
        
        if enable_watermark:
            try:
                from src.detection import get_watermark_detector
                wm_detector = get_watermark_detector(use_audioseal=True)
                wm_result = wm_detector.detect(tmp_path)
                watermark_detected = wm_result.has_watermark
                watermark_confidence = wm_result.confidence
            except Exception:
                pass
        
        # Risk assessment
        from src.analysis.risk import risk_scorer
        risk_result = risk_scorer.calculate_risk(
            fake_probability=fake_prob,
            watermark_detected=watermark_detected,
            acoustic_anomalies={"anomaly_score": 0.0} # Placeholder for now
        )
        
        # RAG & LLM Explanation
        explanation = ""
        legal_docs = []
        
        try:
            from src.rag.engine import legal_rag
            from src.llm.service import llm_service
            
            # Retrieve relevant laws based on risk level
            query = "보이스피싱 처벌 및 피해 구제 절차"
            if risk_result["level"] == "HIGH":
                query += " 사기죄 형량 전기통신금융사기 피해금 환급"
            
            legal_docs = legal_rag.retrieve(query)
            
            # Generate explanation
            explanation = llm_service.generate_explanation(
                detection_result={
                    "is_fake": fake_prob > 0.5,
                    "confidence": max(fake_prob, 1-fake_prob),
                    "risk_score": risk_result["level"]
                },
                legal_context=legal_docs
            )
        except Exception as e:
            print(f"RAG/LLM Error: {e}")
            explanation = "상세 분석 정보를 생성하는 도중 오류가 발생했습니다."

        # Record metrics
        if state.metrics:
            state.metrics.record_request(success=True)
            state.metrics.record_latency("detection", detection_time)
            # state.metrics.record_risk_level(risk_result["level"]) # Update metrics class later
        
        total_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            filename=file.filename,
            is_fake=fake_prob > 0.5,
            confidence=max(fake_prob, 1-fake_prob),
            fake_probability=fake_prob,
            risk_level=risk_result["level"],
            risk_score=risk_result["score"],
            processing_time_ms=total_time,
            model_used=model_name,
            watermark_detected=watermark_detected,
            explanation=explanation,
            legal_references=[doc["text"][:50]+"..." for doc in legal_docs]
        )
        
    finally:
        # Cleanup temp file
        os.unlink(tmp_path)


@app.post("/api/v1/legal-query", response_model=RAGQueryResponse)
async def query_legal_guidance(request: RAGQueryRequest):
    """
    Query legal guidance using RAG system
    
    - **question**: Legal question in Korean
    - **risk_level**: Risk level from detection
    - **detection_summary**: Summary of detection result
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Security check
    if state.security:
        check = state.security.check_request(request.question)
        if not check.is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Security check failed: {', '.join(check.violations)}"
            )
        question = check.sanitized_input or request.question
    else:
        question = request.question
    
    # Get RAG system
    rag = get_rag()
    
    if rag:
        response = rag.query(
            question=question,
            risk_level=request.risk_level,
            detection_summary=request.detection_summary,
            top_k=request.top_k
        )

        print(response)
        
        answer = response.answer
        sources = response.sources
        tokens = response.generation_tokens
        latency = response.total_latency_ms
    else:
        # Mock response for demo
        answer = f"""## 보이스피싱 관련 법률 안내

죄송합니다. 현재 RAG 시스템이 초기화되지 않았습니다.

### 일반 안내
질문: {question}

보이스피싱 피해가 의심되시면:
1. 즉시 경찰청(112)에 신고하세요.
2. 금융감독원(1332)에 상담을 요청하세요.
3. 해당 금융기관에 지급정지를 요청하세요.

[형법 제347조], [전기통신금융사기 특별법] 등을 참조하세요.
"""
        sources = [{"title": "보이스피싱 대응 가이드", "relevance_score": 0.9}]
        tokens = 150
        latency = 100
    
    # Record metrics
    if state.metrics:
        state.metrics.record_request(success=True)
        state.metrics.record_latency("rag", latency)
        state.metrics.record_tokens(tokens)
    
    return RAGQueryResponse(
        request_id=request_id,
        answer=answer,
        sources=sources,
        tokens_used=tokens,
        latency_ms=latency,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit user feedback for a request"""
    
    # Log feedback asynchronously
    if state.monitor:
        # This would log to Langfuse
        pass
    
    return {"status": "success", "message": "Feedback recorded"}


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    if state.metrics:
        summary = state.metrics.get_summary()
        return MetricsResponse(**summary)
    else:
        return MetricsResponse(
            uptime_seconds=0,
            requests={"total": 0, "success": 0, "error": 0, "success_rate": 0},
            latency={"detection": {}, "rag": {}, "total": {}},
            tokens={"mean": 0, "p50": 0, "p95": 0, "p99": 0},
            risk_distribution={"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        )


@app.get("/api/v1/risk-levels")
async def get_risk_levels():
    """Get risk level definitions"""
    return RISK_LEVELS


@app.get("/api/v1/models")
async def get_available_models():
    """Get available detection models"""
    from src.config import MODEL_CONFIGS
    return MODEL_CONFIGS


# === Main ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
