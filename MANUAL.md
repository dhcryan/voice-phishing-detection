# 📘 AI 보이스피싱 탐지 시스템 - 프로젝트 수행 매뉴얼

> **목표**: 가짜 음성 탐지 + 법적 근거 RAG 기반 리스크 스코어링 및 LLMOps 최적화

---

## 📋 목차

1. [Phase 1: 환경 설정 및 데이터 준비](#phase-1-환경-설정-및-데이터-준비-1-2일)
2. [Phase 2: 탐지 모델 구축](#phase-2-탐지-모델-구축-3-5일)
3. [Phase 3: RAG 시스템 구축](#phase-3-rag-시스템-구축-2-3일)
4. [Phase 4: API 서버 및 프론트엔드](#phase-4-api-서버-및-프론트엔드-2-3일)
5. [Phase 5: LLMOps 모니터링](#phase-5-llmops-모니터링-2-3일)
6. [Phase 6: 평가 및 최적화](#phase-6-평가-및-최적화-3-5일)
7. [Phase 7: 배포 및 문서화](#phase-7-배포-및-문서화-1-2일)

---

## Phase 1: 환경 설정 및 데이터 준비 (1-2일)

### 1.1 개발 환경 설정

```bash
# 1. 프로젝트 디렉토리 이동
cd /home/dhc99/voice-phishing-detection

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. GPU 확인 (선택사항)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.2 데이터셋 다운로드

#### ASVspoof 2021 (필수 - 메인 벤치마크)
```bash
# 1. 공식 사이트 등록: https://www.asvspoof.org/index2021.html
# 2. 다운로드 링크 이메일 수신
# 3. LA (Logical Access) 트랙 다운로드

mkdir -p data/audio/asvspoof2021
cd data/audio/asvspoof2021

# 다운로드 (받은 링크로 대체)
wget <YOUR_DOWNLOAD_LINK>/LA.zip
unzip LA.zip
```

#### MLAAD (다국어 테스트용)
```bash
# Hugging Face에서 다운로드
pip install datasets

python << 'EOF'
from datasets import load_dataset
ds = load_dataset("Habs/MLAAD", split="train")
ds.save_to_disk("data/audio/mlaad")
print(f"✅ MLAAD 다운로드 완료: {len(ds)} samples")
EOF
```

#### WaveFake (추가 평가용)
```bash
mkdir -p data/audio/wavefake
cd data/audio/wavefake
wget https://zenodo.org/record/5642694/files/generated_audio.zip
unzip generated_audio.zip
```

### 1.3 데이터 구조 확인

```bash
# 데이터 구조 확인
tree data/audio -L 2

# 예상 구조:
# data/audio/
# ├── asvspoof2021/
# │   ├── LA/
# │   │   ├── ASVspoof2021_LA_train/
# │   │   ├── ASVspoof2021_LA_dev/
# │   │   └── ASVspoof2021_LA_eval/
# ├── mlaad/
# └── wavefake/
```

### ✅ Phase 1 체크리스트
- [ ] 가상환경 활성화 및 패키지 설치 완료
- [ ] `.env` 파일 API 키 설정 완료
- [ ] ASVspoof 2021 LA 데이터셋 다운로드
- [ ] 데이터 디렉토리 구조 확인

---

## Phase 2: 탐지 모델 구축 (3-5일)

### 2.1 데이터 전처리 파이프라인

```python
# notebooks/01_data_preprocessing.ipynb 생성

import librosa
import numpy as np
from pathlib import Path
import pandas as pd

# ASVspoof 프로토콜 파일 로드
def load_protocol(protocol_path):
    """ASVspoof 프로토콜 파일 파싱"""
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            data.append({
                'speaker_id': parts[0],
                'audio_file': parts[1],
                'system_id': parts[3],  # bonafide or spoof system
                'label': parts[4]  # bonafide / spoof
            })
    return pd.DataFrame(data)

# 오디오 전처리
def preprocess_audio(audio_path, sr=16000, max_duration=4):
    """오디오 로드 및 정규화"""
    waveform, _ = librosa.load(audio_path, sr=sr)
    
    # 길이 맞추기
    max_samples = sr * max_duration
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    else:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))
    
    # 정규화
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    
    return waveform

# 데이터셋 클래스
class ASVspoofDataset:
    def __init__(self, audio_dir, protocol_path):
        self.audio_dir = Path(audio_dir)
        self.protocol = load_protocol(protocol_path)
        
    def __len__(self):
        return len(self.protocol)
    
    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        audio_path = self.audio_dir / f"{row['audio_file']}.flac"
        waveform = preprocess_audio(audio_path)
        label = 0 if row['label'] == 'bonafide' else 1
        return waveform, label
```

### 2.2 모델 학습 (AASIST 또는 RawNet2)

```python
# notebooks/02_model_training.ipynb 생성

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.detection.detector import RawNet2, AASIST

# 설정
CONFIG = {
    'model': 'rawnet2',  # 'rawnet2' or 'aasist'
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 모델 초기화
if CONFIG['model'] == 'rawnet2':
    model = RawNet2(num_classes=2)
else:
    model = AASIST(num_classes=2)

model = model.to(CONFIG['device'])

# 학습
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (audio, labels) in enumerate(dataloader):
        audio = audio.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader), correct / total

# 학습 루프
for epoch in range(CONFIG['epochs']):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 체크포인트 저장
    if val_acc > best_acc:
        torch.save(model.state_dict(), f"models/checkpoints/{CONFIG['model']}_best.pt")
        best_acc = val_acc
```

### 2.3 모델 평가 (EER, min-tDCF)

```python
# notebooks/03_model_evaluation.ipynb 생성

from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def compute_eer(y_true, y_scores):
    """Equal Error Rate 계산"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # EER: FPR = FNR 지점
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold

def compute_min_tdcf(y_true, y_scores, Pspoof=0.05, Cmiss=1, Cfa=10):
    """min t-DCF 계산 (ASVspoof 표준)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # t-DCF 계산
    Ptar = 1 - Pspoof
    dcf = Cmiss * fnr * Ptar + Cfa * fpr * Pspoof
    min_tdcf = np.min(dcf)
    
    return min_tdcf

# 평가 실행
model.eval()
all_scores = []
all_labels = []

with torch.no_grad():
    for audio, labels in test_loader:
        audio = audio.to(CONFIG['device'])
        outputs = model(audio)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # spoof 확률
        
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

# 메트릭 계산
eer, threshold = compute_eer(all_labels, all_scores)
min_tdcf = compute_min_tdcf(all_labels, all_scores)
auc = roc_auc_score(all_labels, all_scores)

print(f"📊 평가 결과:")
print(f"  EER: {eer*100:.2f}%")
print(f"  min t-DCF: {min_tdcf:.4f}")
print(f"  AUC: {auc:.4f}")
```

### ✅ Phase 2 체크리스트
- [ ] 데이터 전처리 파이프라인 구축
- [ ] RawNet2 또는 AASIST 모델 학습
- [ ] ASVspoof 2021-LA dev set 평가
- [ ] EER < 5% 달성 확인
- [ ] 모델 체크포인트 저장 (`models/checkpoints/`)

---

## Phase 3: RAG 시스템 구축 (2-3일)

### 3.1 법률 문서 수집 및 인덱싱

```python
# notebooks/04_rag_setup.ipynb 생성

from src.rag.legal_rag import LegalDocumentLoader, VectorStore

# 1. 법률 문서 로드
loader = LegalDocumentLoader(docs_path="data/legal_docs")
documents = loader.get_documents()

print(f"📚 로드된 문서 수: {len(documents)}")
for doc in documents:
    print(f"  - {doc.metadata['title']}")

# 2. 벡터 인덱스 생성
vector_store = VectorStore(
    embedding_model="text-embedding-3-small",
    index_path="data/vectors"
)

# 인덱스 빌드
vector_store.build_index(documents)
vector_store.save_index("legal_docs")

print("✅ 벡터 인덱스 생성 완료!")
```

### 3.2 RAG 파이프라인 테스트

```python
from src.rag.legal_rag import LegalRAG, create_rag_system

# RAG 시스템 생성
rag = create_rag_system(
    docs_path="data/legal_docs",
    vector_path="data/vectors",
    llm_model="gpt-4o-mini"
)

# 테스트 질의
test_questions = [
    "보이스피싱 피해를 당했을 때 어떻게 해야 하나요?",
    "보이스피싱 사기죄의 형량은 어떻게 되나요?",
    "지급정지 신청 절차를 알려주세요.",
    "통장 명의를 빌려준 경우 처벌받나요?"
]

for question in test_questions:
    print(f"\n❓ 질문: {question}")
    response = rag.query(question, risk_level="HIGH")
    print(f"📜 답변: {response.answer[:200]}...")
    print(f"📚 참조: {[s['title'] for s in response.sources]}")
    print(f"⏱️ 응답시간: {response.total_latency_ms:.0f}ms")
```

### 3.3 추가 법령 문서 확장 (선택)

```bash
# 법제처에서 추가 법령 수집
# https://www.law.go.kr 에서 다음 법령 검색하여 JSON 추가:
# - 정보통신망 이용촉진 및 정보보호 등에 관한 법률
# - 개인정보 보호법
# - 금융실명거래 및 비밀보장에 관한 법률
```

### ✅ Phase 3 체크리스트
- [ ] 법률 문서 JSON 파일 생성 완료
- [ ] 벡터 인덱스 빌드 완료
- [ ] RAG 질의 테스트 성공
- [ ] 법령 인용 정확성 확인

---

## Phase 4: API 서버 및 프론트엔드 (2-3일)

### 4.1 FastAPI 서버 실행 및 테스트

```bash
# 1. API 서버 실행
cd /home/dhc99/voice-phishing-detection
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000

# 2. 새 터미널에서 API 테스트
curl http://localhost:8000/health

# 3. Swagger UI 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

### 4.2 API 엔드포인트 테스트

```python
# notebooks/05_api_test.ipynb 생성

import requests

API_URL = "http://localhost:8000"

# 1. 헬스체크
response = requests.get(f"{API_URL}/health")
print(f"Health: {response.json()}")

# 2. 음성 탐지 테스트
with open("data/audio/test_sample.wav", "rb") as f:
    files = {"file": ("test.wav", f, "audio/wav")}
    response = requests.post(
        f"{API_URL}/api/v1/detect",
        files=files,
        params={"model_type": "aasist"}
    )
    
result = response.json()
print(f"탐지 결과:")
print(f"  가짜 확률: {result['fake_probability']:.1%}")
print(f"  리스크: {result['risk_level_label']}")

# 3. 법률 질의 테스트
response = requests.post(
    f"{API_URL}/api/v1/legal-query",
    json={
        "question": "보이스피싱 신고는 어디에 하나요?",
        "risk_level": "HIGH"
    }
)
print(f"\n법률 답변:\n{response.json()['answer'][:300]}...")
```

### 4.3 Streamlit 프론트엔드 실행

```bash
# 1. Streamlit 실행 (별도 터미널)
streamlit run frontend/app.py --server.port 8501

# 2. 브라우저에서 접속
# http://localhost:8501
```

### 4.4 통합 실행 스크립트

```bash
# API + Streamlit 동시 실행
./scripts/run.sh
```

### ✅ Phase 4 체크리스트
- [ ] FastAPI 서버 정상 실행
- [ ] `/api/v1/detect` 엔드포인트 동작 확인
- [ ] `/api/v1/legal-query` 엔드포인트 동작 확인
- [ ] Streamlit UI 정상 표시
- [ ] 음성 업로드 → 분석 → 결과 표시 플로우 확인

---

## Phase 5: LLMOps 모니터링 (2-3일)

### 5.1 Langfuse 대시보드 설정

```bash
# 1. Langfuse 클라우드 접속
# https://cloud.langfuse.com

# 2. 프로젝트 생성 (이미 완료된 경우 스킵)

# 3. API 키 확인 (.env에 이미 설정됨)
cat .env | grep LANGFUSE
```

### 5.2 Langfuse 트레이싱 확인

```python
# notebooks/06_langfuse_monitoring.ipynb 생성

from langfuse import Langfuse

# Langfuse 클라이언트
langfuse = Langfuse()

# 트레이스 생성 테스트
trace = langfuse.trace(
    name="test-detection-flow",
    user_id="test-user",
    metadata={"test": True}
)

# 스팬 추가
span = trace.span(
    name="audio-preprocessing",
    input={"audio_file": "test.wav"},
    output={"duration_ms": 1000}
)

# Generation 로깅
trace.generation(
    name="legal-response",
    model="gpt-4o-mini",
    input="보이스피싱 신고 방법",
    output="경찰청 112 또는 금융감독원 1332로 신고하세요.",
    usage={"input": 50, "output": 100}
)

langfuse.flush()
print("✅ Langfuse 트레이싱 성공!")
print("   대시보드에서 확인: https://cloud.langfuse.com")
```

### 5.3 프롬프트 버전 관리 및 A/B 테스트

```python
# 프롬프트 버전별 성능 비교

from src.config import prompts
import time

prompt_versions = {
    "v1": prompts.DETECTION_SUMMARY_V1,
    "v2": prompts.DETECTION_SUMMARY_V2
}

results = []

for version, prompt_template in prompt_versions.items():
    start = time.time()
    
    # LLM 호출 (실제 구현)
    response = call_llm(prompt_template.format(
        model_name="AASIST",
        fake_probability=0.85,
        risk_level="HIGH",
        acoustic_anomalies="높은 스펙트럼 평탄도",
        watermark_detected="없음"
    ))
    
    latency = (time.time() - start) * 1000
    
    results.append({
        "version": version,
        "latency_ms": latency,
        "tokens": response.usage.total_tokens,
        "response_length": len(response.content)
    })
    
    # Langfuse에 기록
    trace.generation(
        name=f"summary-{version}",
        model="gpt-4o-mini",
        metadata={"prompt_version": version}
    )

# 비교 결과
import pandas as pd
df = pd.DataFrame(results)
print(df.to_markdown())
```

### 5.4 메트릭 대시보드 확인

```python
# API 메트릭 조회
response = requests.get(f"{API_URL}/api/v1/metrics")
metrics = response.json()

print("📊 시스템 메트릭:")
print(f"  총 요청: {metrics['requests']['total']}")
print(f"  성공률: {metrics['requests']['success_rate']:.1%}")
print(f"  평균 응답시간: {metrics['latency']['total']['mean']:.0f}ms")
print(f"  리스크 분포: {metrics['risk_distribution']}")
```

### ✅ Phase 5 체크리스트
- [ ] Langfuse 대시보드 접속 확인
- [ ] API 요청 트레이스 기록 확인
- [ ] LLM 토큰 사용량 추적 확인
- [ ] 프롬프트 버전별 성능 비교 완료
- [ ] 메트릭 대시보드 데이터 확인

---

## Phase 6: 평가 및 최적화 (3-5일)

### 6.1 탐지 모델 성능 벤치마크

```python
# notebooks/07_benchmark.ipynb 생성

import pandas as pd

# 모델별 성능 비교
models = ['aasist', 'rawnet2', 'ecapa']
datasets = ['asvspoof2021_la', 'mlaad', 'wavefake']

results = []
for model in models:
    for dataset in datasets:
        eer, min_tdcf, rtf = evaluate_model(model, dataset)
        results.append({
            'model': model,
            'dataset': dataset,
            'EER': eer,
            'min_tDCF': min_tdcf,
            'RTF': rtf  # Real-Time Factor
        })

df = pd.DataFrame(results)
print("📊 모델 성능 비교:")
print(df.pivot_table(index='model', columns='dataset', values='EER'))
```

### 6.2 Latency 최적화

```python
# 단계별 Latency 분해

import time

def profile_pipeline(audio_path, question):
    """전체 파이프라인 프로파일링"""
    timings = {}
    
    # 1. 오디오 전처리
    start = time.time()
    waveform = preprocess_audio(audio_path)
    timings['preprocess'] = (time.time() - start) * 1000
    
    # 2. 탐지 추론
    start = time.time()
    detection_result = detector.predict(audio_path)
    timings['detection'] = (time.time() - start) * 1000
    
    # 3. 리스크 스코어링
    start = time.time()
    risk_result = scorer.assess_risk(detection_result.fake_probability)
    timings['scoring'] = (time.time() - start) * 1000
    
    # 4. LLM 요약
    start = time.time()
    summary = generate_summary(detection_result, risk_result)
    timings['llm_summary'] = (time.time() - start) * 1000
    
    # 5. RAG 검색 + 생성
    start = time.time()
    rag_response = rag.query(question, risk_result.risk_level.value)
    timings['rag'] = (time.time() - start) * 1000
    
    timings['total'] = sum(timings.values())
    
    return timings

# 프로파일링 실행
timings = profile_pipeline("test.wav", "신고 방법은?")
print("⏱️ Latency 분해:")
for step, ms in timings.items():
    pct = ms / timings['total'] * 100
    print(f"  {step}: {ms:.0f}ms ({pct:.1f}%)")
```

### 6.3 토큰/비용 최적화

```python
# 프롬프트 최적화로 토큰 절감

import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

# 기존 프롬프트
original_tokens = len(enc.encode(prompts.DETECTION_SUMMARY_V1))

# 최적화 프롬프트 (더 간결하게)
optimized_prompt = """탐지결과: {model_name}, 가짜확률 {fake_probability:.0%}, {risk_level}
요청: 1) 위험 이유 2) 탐지 근거 3) 권장 조치를 간결히 설명"""

optimized_tokens = len(enc.encode(optimized_prompt))

print(f"토큰 절감: {original_tokens} → {optimized_tokens} ({(1-optimized_tokens/original_tokens)*100:.1f}% 절감)")
```

### 6.4 RAG 품질 평가

```python
# LLM-as-Judge 평가

evaluation_criteria = """
다음 기준으로 1-5점 평가:
1. 법령 인용 정확성: 조항 번호가 정확한가?
2. 답변 관련성: 질문에 적절히 답했는가?
3. 실용성: 구체적 행동 가이드를 제공하는가?
4. 명확성: 일반인이 이해할 수 있는가?
"""

def evaluate_rag_response(question, response, sources):
    """LLM으로 RAG 응답 품질 평가"""
    eval_prompt = f"""
{evaluation_criteria}

질문: {question}
답변: {response}
인용 출처: {sources}

각 기준별 점수와 이유를 JSON으로 출력하세요.
"""
    # LLM 호출하여 평가
    result = call_llm(eval_prompt)
    return parse_evaluation(result)

# 테스트셋 평가
test_cases = [
    ("보이스피싱 신고 방법", "HIGH"),
    ("피해금 환급 절차", "HIGH"),
    ("사기죄 형량", "MEDIUM"),
]

scores = []
for question, risk_level in test_cases:
    response = rag.query(question, risk_level)
    eval_result = evaluate_rag_response(question, response.answer, response.sources)
    scores.append(eval_result)

avg_score = sum(s['total'] for s in scores) / len(scores)
print(f"📊 RAG 평균 품질 점수: {avg_score:.2f}/5.0")
```

### ✅ Phase 6 체크리스트
- [ ] 모델별 EER/min-tDCF 벤치마크 완료
- [ ] Latency 병목 구간 식별 및 개선
- [ ] 토큰 사용량 10% 이상 절감
- [ ] RAG 품질 평가 (LLM-as-Judge) 실행
- [ ] 법령 인용률 100% 확인

---

## Phase 7: 배포 및 문서화 (1-2일)

### 7.1 Docker 컨테이너화

```dockerfile
# Dockerfile 생성
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY src/ src/
COPY frontend/ frontend/
COPY data/legal_docs/ data/legal_docs/

# 환경 변수
ENV PYTHONPATH=/app

# 포트
EXPOSE 8000 8501

# 실행
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Docker Compose

```yaml
# docker-compose.yml 생성
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./models/checkpoints:/app/models/checkpoints
      - ./data/vectors:/app/data/vectors
    
  frontend:
    build: .
    command: streamlit run frontend/app.py --server.port 8501
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

### 7.3 배포 실행

```bash
# Docker 빌드 및 실행
docker-compose up --build -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 7.4 최종 문서 점검

```bash
# 문서 목록 확인
ls -la *.md

# README.md - 프로젝트 개요
# MANUAL.md - 이 매뉴얼
```

### ✅ Phase 7 체크리스트
- [ ] Dockerfile 작성 및 빌드 성공
- [ ] Docker Compose 설정 완료
- [ ] 컨테이너 실행 테스트
- [ ] README.md 최종 업데이트
- [ ] 프로젝트 GitHub 업로드 (선택)

---

## 📊 최종 성과 목표 체크리스트

### 탐지 성능
- [ ] ASVspoof 2021-LA EER < 5%
- [ ] min-tDCF 베이스라인 대비 개선
- [ ] Real-Time Factor < 1.0 (실시간 처리)

### 운영 최적화
- [ ] 평균 응답 Latency < 2초
- [ ] 토큰 비용 10% 절감
- [ ] 에러율 < 1%

### RAG 품질
- [ ] 법령 인용률 100%
- [ ] LLM-as-Judge 평균 4.0/5.0 이상

### 모니터링
- [ ] Langfuse 대시보드 활성화
- [ ] 프롬프트 A/B 테스트 결과 분석
- [ ] 사용자 피드백 수집 파이프라인

---

## 🆘 문제 해결

### GPU 메모리 부족
```bash
# 배치 사이즈 줄이기
# src/config.py에서 DETECTION_BATCH_SIZE = 4 로 변경
```

### API 연결 오류
```bash
# 포트 확인
netstat -tlnp | grep 8000

# 방화벽 확인
sudo ufw allow 8000
```

### OpenAI API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY

# 잔액 확인
# https://platform.openai.com/usage
```

---

## 📞 참고 자료

- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [국가법령정보센터](https://www.law.go.kr/)

---

**마지막 업데이트**: 2025년 12월 1일
