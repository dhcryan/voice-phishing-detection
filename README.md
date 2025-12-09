# 🔒 AI 보이스피싱 대응 시스템

> AI 기반 가짜 음성 탐지 + 법적 근거 RAG 기반 리스크 스코어링 및 LLMOps 최적화 시스템

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 목차

- [개요](#-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [API 문서](#-api-문서)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)
- [데이터셋](#-데이터셋)
- [성능 지표](#-성능-지표)

---

## 🎯 개요

보이스 클로닝 기술의 발전으로 음성 합성/변조를 이용한 보이스피싱 사기가 급증하고 있습니다. 이 프로젝트는 다음을 제공합니다:

1. **가짜 음성 탐지**: AASIST, RawNet2, ECAPA-TDNN 등 최신 안티스푸핑 모델
2. **리스크 스코어링**: 탐지 결과, 워터마크, 음향 이상을 종합한 Low/Medium/High 분류
3. **법률 RAG 시스템**: 관련 법령(형법, 전기통신금융사기법 등) 인용 기반 대응 안내
4. **LLMOps 모니터링**: Langfuse 연동으로 프롬프트 버전/비용/지연 분석

---

## ✨ 주요 기능

### 🔍 음성 탐지
- **다양한 탐지 모델**: AASIST (그래프 어텐션), RawNet2 (End-to-End), ECAPA-TDNN (스피커 임베딩)
- **워터마크 탐지**: AudioSeal 기반 AI 생성 음성 워터마크 검출
- **음향 이상 분석**: MFCC, 스펙트럼 특성, ZCR 기반 이상 패턴 감지

### 📊 리스크 스코어링
- **멀티 시그널 융합**: 탐지 확률 + 워터마크 + 음향 이상
- **3단계 분류**: 저위험(🟢), 중위험(🟡), 고위험(🔴)
- **상세 기여도 분석**: 각 요인별 위험 기여도 표시

### ⚖️ 법률 RAG
- **한국 법령 기반**: 형법, 전자금융거래법, 전기통신금융사기 특별법 등
- **조항 인용**: 관련 법 조항 번호와 내용 명시
- **대응 체크리스트**: 상황별 구체적 행동 가이드

### 📈 LLMOps 모니터링
- **Langfuse 연동**: 요청/응답, 토큰 사용량, 지연시간 추적
- **프롬프트 버전 관리**: A/B 테스트 및 품질 비교
- **비용 분석**: 모델별/요청별 비용 집계

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Streamlit)                     │
│                    - 음성 업로드 - 결과 시각화 - 법률 상담        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP/REST
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                           │
│           - /detect - /legal-query - /feedback - /metrics        │
└──────┬──────────────┬───────────────┬──────────────┬────────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌──────────┐  ┌───────────┐  ┌────────────┐  ┌─────────────┐
│ Detection│  │   Risk    │  │  Legal RAG │  │  Langfuse   │
│ Models   │  │  Scorer   │  │   System   │  │  Monitoring │
│ ────────│  │ ─────────│  │ ──────────│  │ ───────────│
│ AASIST   │  │ Multi-sig │  │ FAISS/     │  │ Traces      │
│ RawNet2  │  │ fusion    │  │ ChromaDB   │  │ Generations │
│ ECAPA    │  │           │  │            │  │ Metrics     │
└──────────┘  └───────────┘  └────────────┘  └─────────────┘
       │              │               │              │
       └──────────────┴───────────────┴──────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │   OpenAI GPT API   │
                    │   (Embeddings/LLM) │
                    └────────────────────┘
```

---

## 🚀 설치 방법

### 사전 요구사항
- Python 3.10+
- CUDA 지원 GPU (선택사항, CPU도 가능)
- OpenAI API 키

### 1. 저장소 클론 및 환경 설정

```bash
cd voice-phishing-detection

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 설정하세요:

```env
OPENAI_API_KEY=your-openai-api-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
```

### 3. 실행

```bash
# API 서버 실행
uvicorn src.api.main:app --reload --port 8000

# 별도 터미널에서 Streamlit 실행
streamlit run frontend/app.py --server.port 8501
```

또는 통합 실행:

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

---

## 💻 사용 방법

### 웹 인터페이스

1. 브라우저에서 `http://localhost:8501` 접속
2. **음성 분석** 탭에서 음성 파일 업로드
3. 분석 결과 및 리스크 레벨 확인
4. **법률 상담** 탭에서 관련 질문

### API 직접 호출

```python
import requests

# 음성 탐지
files = {"file": open("audio.wav", "rb")}
response = requests.post(
    "http://localhost:8000/api/v1/detect",
    files=files,
    params={"model_type": "aasist"}
)
result = response.json()
print(f"가짜 확률: {result['fake_probability']:.1%}")
print(f"리스크 레벨: {result['risk_level_label']}")

# 법률 상담
response = requests.post(
    "http://localhost:8000/api/v1/legal-query",
    json={
        "question": "보이스피싱 피해 신고는 어떻게 하나요?",
        "risk_level": "HIGH"
    }
)
print(response.json()["answer"])
```

---

## 📚 API 문서

서버 실행 후 `http://localhost:8000/docs`에서 Swagger UI 확인

### 주요 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/detect` | 음성 파일 분석 |
| POST | `/api/v1/legal-query` | 법률 질의 |
| POST | `/api/v1/feedback` | 피드백 제출 |
| GET | `/api/v1/metrics` | 시스템 메트릭 |
| GET | `/api/v1/risk-levels` | 리스크 레벨 정의 |
| GET | `/api/v1/models` | 사용 가능 모델 목록 |

---

## 📁 프로젝트 구조

```
voice-phishing-detection/
├── src/
│   ├── api/                 # FastAPI 서버
│   │   ├── __init__.py
│   │   └── main.py          # API 엔드포인트
│   ├── detection/           # 음성 탐지 모듈
│   │   ├── __init__.py
│   │   ├── detector.py      # AASIST, RawNet2, ECAPA
│   │   └── watermark.py     # AudioSeal 워터마크 탐지
│   ├── rag/                  # 법률 RAG 시스템
│   │   ├── __init__.py
│   │   └── engine.py        # ChromaDB 벡터 검색 엔진
│   ├── llm/                  # LLM 서비스
│   │   ├── __init__.py
│   │   └── service.py       # OpenAI & Langfuse 연동
│   ├── analysis/             # 분석 및 스코어링
│   │   ├── __init__.py
│   │   └── risk.py          # 리스크 스코어링 로직
│   ├── utils/                # 유틸리티
│   │   ├── __init__.py
│   │   └── monitoring.py    # 모니터링 도구
│   └── config.py             # 설정 및 프롬프트
├── frontend/
│   └── app.py                # Streamlit UI
├── data/
│   ├── audio/                # 음성 파일
│   ├── legal_docs/           # 법률 문서
│   └── vectors/              # 벡터 인덱스
├── models/
│   ├── checkpoints/          # 모델 가중치
│   └── configs/              # 모델 설정
├── tests/
│   └── test_main.py          # 테스트
├── scripts/
│   ├── setup.sh              # 설치 스크립트
│   └── run.sh                # 실행 스크립트
├── notebooks/                # 실험 노트북
├── logs/                     # 로그
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠 기술 스택

### Backend
- **FastAPI**: 비동기 API 서버
- **PyTorch / TorchAudio**: 딥러닝 프레임워크
- **Librosa**: 오디오 처리
- **OpenAI API**: 임베딩 및 LLM

### RAG
- **FAISS / ChromaDB**: 벡터 데이터베이스
- **LangChain**: RAG 파이프라인

### Frontend
- **Streamlit**: 대화형 웹 UI
- **Plotly**: 데이터 시각화

### MLOps / LLMOps
- **Langfuse**: LLM 모니터링
- **MLflow**: 모델 실험 추적

### 보안
- **프롬프트 인젝션 탐지**: 정규식 기반 필터링
- **PII 필터링**: 개인정보 마스킹
- **레이트 리미터**: 요청 제한

---

## 📊 데이터셋

### 탐지 모델 학습/평가

| 데이터셋 | 설명 | 링크 |
|---------|------|------|
| ASVspoof 2021 | 공식 안티스푸핑 벤치마크 (LA/PA/DF) | [asvspoof.org](https://www.asvspoof.org/index2021.html) |
| MLAAD | 23개 언어, 52개 TTS 다국어 | [HuggingFace](https://huggingface.co/papers/2401.09512) |
| WaveFake | TTS/VC 기반 딥페이크 음성 | [arXiv](https://arxiv.org/abs/2111.02813) |

### 법률 RAG 코퍼스

- 형법 제347조 (사기), 제347조의2 (컴퓨터등사용사기)
- 전자금융거래법
- 전기통신금융사기 피해 방지 및 환급 특별법
- 특정경제범죄 가중처벌법

---

## 📈 성능 지표

### 탐지 성능 목표

| 지표 | 목표 | 비고 |
|------|------|------|
| EER | < 5% | ASVspoof 2021-LA 기준 |
| min-tDCF | 개선 | 베이스라인 대비 |
| ROC-AUC | > 0.95 | 전체 테스트셋 |

### 운영 성능 목표

| 지표 | 목표 |
|------|------|
| 평균 응답 지연 | < 2초 |
| 토큰 비용 | 10% 절감 |
| 법령 인용률 | 100% |

---

## 📝 라이선스

MIT License

---

## 🙏 참고 자료

- [AASIST 논문](https://arxiv.org/abs/2110.01200)
- [RawNet2 논문](https://arxiv.org/abs/2011.01108)
- [ASVspoof Challenge](https://www.asvspoof.org/)
- [AudioSeal](https://github.com/facebookresearch/audioseal)
- [국가법령정보센터](https://www.law.go.kr/)

---

## 📞 문의

프로젝트에 대한 질문이나 제안은 Issue를 통해 남겨주세요.

---

## 🚀 개발자 매뉴얼 (Developer Manual)

이 프로젝트를 클론하고 실행하기 위한 단계별 가이드입니다.

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd voice-phishing-detection

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습 (Quick Start)

실제 데이터셋(ASVspoof 등)이 없는 경우, 합성 데이터를 생성하여 빠르게 모델을 학습시킬 수 있습니다.

```bash
# 합성 데이터 생성 및 Simple CNN 모델 학습
python scripts/quick_train.py
```
위 명령어를 실행하면 `checkpoints/simple_detector_best.pt` 모델이 생성됩니다.

### 3. 서버 실행

기본적으로 API 서버는 8001 포트, 웹 인터페이스는 8501 포트를 사용합니다.

```bash
# 1. API 서버 실행 (Port 8001)
uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# 2. 웹 인터페이스 실행 (Port 8501)
# 새 터미널에서 실행하세요
export API_URL="http://localhost:8001"
streamlit run frontend/app.py --server.port 8501
```

### 4. RAG 및 LLM 설정 (선택 사항)

법률 자문 및 상세 분석 기능을 사용하려면 추가 설정이 필요합니다.

#### 4.1. API 키 설정
`.env` 파일을 생성하거나 `src/config.py`를 수정하여 API 키를 입력하세요.

```bash
export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

#### 4.2. 법률 데이터베이스 구축
기본 제공되는 스크립트를 실행하여 법률 벡터 데이터베이스를 생성합니다.

```bash
python scripts/build_rag_db.py
```
이 명령어를 실행하면 `data/chromadb` 폴더에 벡터 DB가 생성되며, 이후 서버 실행 시 자동으로 로드됩니다.

---

## 📥 향후 작업 및 데이터셋 다운로드

현재 시스템은 합성 데이터를 기반으로 작동합니다. 실제 보이스피싱 탐지 성능을 확보하기 위해 다음 작업이 필요합니다.

### 1. 필수 데이터셋 다운로드
다음 데이터셋을 다운로드하여 `data/raw/` 폴더에 위치시켜야 합니다.
- **ASVspoof 2019 LA**: [다운로드 링크](https://datashare.ed.ac.uk/handle/10283/3336)
- **WaveFake**: [Kaggle 링크](https://www.kaggle.com/datasets/testing123/wavefake)
- **In-the-wild**: 실제 보이스피싱 사례 데이터

### 2. 추가 개발 필요 사항
- [ ] **실제 데이터 학습**: `scripts/train_model.py`를 사용하여 대규모 데이터셋 학습 수행
- [ ] **RAG 벡터 DB 구축**: 법률 문서 데이터베이스 구축 및 임베딩 (ChromaDB/FAISS)
- [ ] **모델 고도화**: RawNet2, AASIST 모델의 하이퍼파라미터 튜닝
