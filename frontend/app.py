"""
Voice Phishing Detection - Streamlit Frontend
Interactive UI for voice analysis and legal guidance
"""
import streamlit as st
import requests
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="AI 보이스피싱 탐지 시스템",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8001")

# Custom CSS
st.markdown("""
<style>
    .risk-low { 
        background-color: #d4edda; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #28a745;
    }
    .risk-medium { 
        background-color: #fff3cd; 
        padding: 20px; 
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .risk-high { 
        background-color: #f8d7da; 
        padding: 20px; 
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .source-card {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("🔒 보이스피싱 탐지")
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ 설정")
        model_type = st.selectbox(
            "탐지 모델",
            ["aasist", "rawnet2", "ecapa"],
            help="음성 위변조 탐지에 사용할 AI 모델"
        )
        
        enable_watermark = st.checkbox(
            "워터마크 탐지 활성화",
            value=True,
            help="AI 생성 음성의 워터마크 탐지"
        )
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ 정보")
        st.markdown("""
        **지원 형식**: WAV, MP3, FLAC, OGG, M4A
        
        **탐지 모델**:
        - AASIST: 그래프 어텐션 기반
        - RawNet2: End-to-End CNN
        - ECAPA: 스피커 임베딩 활용
        
        **연락처**:
        - 경찰청: 112
        - 금융감독원: 1332
        """)
        
        st.markdown("---")
        st.caption(f"API: {API_URL}")

    # Main content
    st.title("🔍 AI 보이스피싱 탐지 시스템")
    st.markdown("음성 파일을 업로드하여 가짜 음성(합성/변조)을 탐지하고, 관련 법률 안내를 받으세요.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 음성 분석", "⚖️ 법률 상담", "📊 대시보드"])
    
    # Tab 1: Voice Analysis
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 음성 파일 업로드")
            uploaded_file = st.file_uploader(
                "음성 파일을 선택하세요",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="최대 60초 분량의 음성 파일"
            )
            
            if uploaded_file:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
                
                if st.button("🔍 분석 시작", type="primary", use_container_width=True):
                    with st.spinner("음성을 분석하고 있습니다..."):
                        try:
                            # Call API
                            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                            params = {
                                "model_type": model_type,
                                "enable_watermark": enable_watermark
                            }
                            
                            response = requests.post(
                                f"{API_URL}/api/v1/detect",
                                files=files,
                                params=params,
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state["detection_result"] = result
                                st.success("분석이 완료되었습니다!")
                            else:
                                st.error(f"분석 실패: {response.text}")
                                
                        except requests.exceptions.ConnectionError:
                            import traceback
                            traceback.print_exc()
                            st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
                            # Demo result for testing
                            st.session_state["detection_result"] = create_demo_result()
                            st.info("데모 결과를 표시합니다.")
                        except Exception as e:
                            st.error(f"오류 발생: {str(e)}")
        
        with col2:
            st.subheader("📊 분석 결과")
            
            if "detection_result" in st.session_state:
                result = st.session_state["detection_result"]
                
                # Risk level display
                risk_level = result.get("risk_level", "MEDIUM")
                risk_label = result.get("risk_level_label", "중위험")
                risk_score = result.get("risk_score", 0.5)
                
                risk_class = f"risk-{risk_level.lower()}"
                emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(risk_level, "❓")
                st.markdown(f"""
                <div class="{risk_class}">
                    <h2>{emoji} <span style='color:black'>{risk_label}</span></h2>
                    <p><span style='color:black'><strong>리스크 점수:</strong> {risk_score:.1%}</span></p>
                    <p><span style='color:black'><strong>가짜 음성 확률:</strong> {result.get('fake_probability', 0):.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "탐지 모델",
                        result.get("model_used", "N/A")
                    )
                
                with col_b:
                    wm = "감지됨" if result.get("watermark_detected") else "없음"
                    st.metric("워터마크", wm)
                
                with col_c:
                    st.metric(
                        "처리 시간",
                        f"{result.get('processing_time_ms', 0):.0f}ms"
                    )
                
                # Contributing factors
                st.markdown("### 📋 분석 요인")
                for factor in result.get("contributing_factors", []):
                    severity = factor.get("severity", "LOW")
                    icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(severity, "⚪")
                    st.markdown(f"{icon} **{factor.get('factor')}**: {factor.get('detail')}")
                
                # Recommendations
                st.markdown("### 💡 권장 조치")
                for rec in result.get("recommendations", []):
                    st.markdown(f"- {rec}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "리스크 점수"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d4edda"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("음성 파일을 업로드하고 분석을 시작하세요.")
    
    # Tab 2: Legal Query
    with tab2:
        st.subheader("⚖️ 법률 상담 AI")
        st.markdown("보이스피싱 관련 법률 및 대응 방법에 대해 질문하세요.")
        
        # Pre-defined questions
        quick_questions = [
            "보이스피싱 피해를 당했을 때 어떻게 해야 하나요?",
            "보이스피싱 사기죄의 형량은 어떻게 되나요?",
            "피해금 환급 절차를 알려주세요.",
            "지급정지 신청은 어떻게 하나요?"
        ]
        
        st.markdown("**빠른 질문:**")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            if st.button(quick_questions[0], use_container_width=True):
                st.session_state["legal_question"] = quick_questions[0]
            if st.button(quick_questions[2], use_container_width=True):
                st.session_state["legal_question"] = quick_questions[2]
        
        with col_q2:
            if st.button(quick_questions[1], use_container_width=True):
                st.session_state["legal_question"] = quick_questions[1]
            if st.button(quick_questions[3], use_container_width=True):
                st.session_state["legal_question"] = quick_questions[3]
        
        st.markdown("---")
        
        # Custom question
        question = st.text_area(
            "질문 입력",
            value=st.session_state.get("legal_question", ""),
            height=100,
            placeholder="예: 보이스피싱 범죄자의 처벌 수위는 어떻게 되나요?"
        )
        
        # Risk context from detection
        risk_level = "MEDIUM"
        detection_summary = ""
        
        if "detection_result" in st.session_state:
            result = st.session_state["detection_result"]
            risk_level = result.get("risk_level", "MEDIUM")
            detection_summary = f"가짜 음성 확률 {result.get('fake_probability', 0):.1%}"
        
        if st.button("📤 질문하기", type="primary", disabled=len(question) < 5):
            with st.spinner("답변을 생성하고 있습니다..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/legal-query",
                        json={
                            "question": question,
                            "risk_level": risk_level,
                            "detection_summary": detection_summary,
                            "top_k": 5
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        st.session_state["legal_response"] = response.json()
                    else:
                        st.error(f"오류: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    import traceback
                    traceback.print_exc()
                    st.error("API 서버에 연결할 수 없습니다.")
                    st.session_state["legal_response"] = create_demo_legal_response(question)
                    st.info("데모 응답을 표시합니다.")
        
        # Display response
        if "legal_response" in st.session_state:
            response = st.session_state["legal_response"]
            
            st.markdown("### 📜 답변")
            st.markdown(response.get("answer", ""))
            
            st.markdown("---")
            
            # Sources
            st.markdown("### 📚 참조 법령")
            for source in response.get("sources", []):
                with st.expander(f"📖 {source.get('title', 'Unknown')}"):
                    st.markdown(f"**참조 문서:** {source.get('raw', 'N/A')}")
            
            # Metadata
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.caption(f"토큰 사용량: {response.get('tokens_used', 0)}")
            with col_m2:
                st.caption(f"응답 시간: {response.get('latency_ms', 0):.0f}ms")
    
    # Tab 3: Dashboard
    with tab3:
        st.subheader("📊 시스템 대시보드")
        
        # Fetch metrics
        try:
            response = requests.get(f"{API_URL}/api/v1/metrics", timeout=10)
            if response.status_code == 200:
                metrics = response.json()
            else:
                metrics = create_demo_metrics()
        except:
            metrics = create_demo_metrics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 요청",
                metrics["requests"]["total"],
                delta=None
            )
        
        with col2:
            success_rate = metrics["requests"]["success_rate"] * 100
            st.metric(
                "성공률",
                f"{success_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "평균 응답시간",
                f"{metrics['latency']['total'].get('mean', 0):.0f}ms"
            )
        
        with col4:
            uptime_hours = metrics["uptime_seconds"] / 3600
            st.metric(
                "가동시간",
                f"{uptime_hours:.1f}h"
            )
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 리스크 탐지 결과 분포")
            risk_data = metrics.get("risk_distribution", {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
            
            fig = px.pie(
                values=list(risk_data.values()),
                names=["저위험", "중위험", "고위험"],
                color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.markdown("### RAG 응답 시간 분포")
            latency = metrics.get("latency", {}).get("rag", {})
            
            fig = go.Figure(data=[
                go.Bar(
                    x=["평균", "P50", "P95", "P99"],
                    y=[
                        latency.get("mean", 0),
                        latency.get("p50", 0),
                        latency.get("p95", 0),
                        latency.get("p99", 0)
                    ],
                    marker_color=["#007bff", "#17a2b8", "#ffc107", "#dc3545"]
                )
            ])
            fig.update_layout(yaxis_title="밀리초(ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Token usage
        st.markdown("### 💰 토큰 사용량")
        tokens = metrics.get("tokens", {})
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        
        with col_t1:
            st.metric("평균", f"{tokens.get('mean', 0):.0f}")
        with col_t2:
            st.metric("P50", f"{tokens.get('p50', 0):.0f}")
        with col_t3:
            st.metric("P95", f"{tokens.get('p95', 0):.0f}")
        with col_t4:
            st.metric("P99", f"{tokens.get('p99', 0):.0f}")


def create_demo_result():
    """Create demo detection result"""
    import random
    
    fake_prob = random.uniform(0.3, 0.9)
    
    if fake_prob < 0.3:
        risk_level = "LOW"
        risk_label = "저위험"
    elif fake_prob < 0.7:
        risk_level = "MEDIUM"
        risk_label = "중위험"
    else:
        risk_level = "HIGH"
        risk_label = "고위험"
    
    return {
        "request_id": "demo-123",
        "is_fake": fake_prob > 0.5,
        "fake_probability": fake_prob,
        "risk_level": risk_level,
        "risk_level_label": risk_label,
        "risk_score": fake_prob * 0.8,
        "watermark_detected": random.choice([True, False]),
        "watermark_confidence": random.uniform(0.1, 0.9),
        "model_used": "AASIST (Demo)",
        "recommendations": [
            "⚠️ 의심스러운 요소가 감지되었습니다.",
            "통화를 종료하고 공식 대표번호로 재확인하세요.",
            "금융거래나 개인정보 제공을 보류하세요."
        ],
        "contributing_factors": [
            {
                "factor": "AI 탐지 모델 결과",
                "severity": "MEDIUM" if fake_prob < 0.7 else "HIGH",
                "detail": f"가짜 음성 확률 {fake_prob:.1%}로 탐지됨"
            },
            {
                "factor": "음향 이상 패턴",
                "severity": "LOW",
                "detail": "일부 비정상 스펙트럼 패턴 감지"
            }
        ],
        "processing_time_ms": random.uniform(200, 800),
        "timestamp": datetime.now().isoformat()
    }


def create_demo_legal_response(question):
    """Create demo legal response"""
    return {
        "request_id": "demo-legal-123",
        "answer": f"""## 보이스피싱 대응 안내

질문: {question}

### 📌 관련 법령

**[형법 제347조 (사기)]**
사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처합니다.

**[전기통신금융사기 특별법 제3조]**
금융회사는 피해자로부터 피해 신고를 받은 경우 즉시 해당 계좌에 대한 지급정지 조치를 해야 합니다.

### ✅ 권장 조치

1. **즉시 신고**: 경찰청(112), 금융감독원(1332)
2. **지급정지 요청**: 송금 금융기관 고객센터
3. **증거 보전**: 통화 녹음, 문자 캡처, 거래 내역 확보
4. **피해 환급 신청**: 채권소멸절차 완료 후 신청 가능

### 📞 연락처
- 경찰청 사이버수사국: 182
- 금융감독원: 1332
- 한국인터넷진흥원: 118
""",
        "sources": [
            {"title": "형법 제347조 (사기)", "category": "criminal", "relevance_score": 0.95},
            {"title": "전기통신금융사기 특별법", "category": "telecom_fraud", "relevance_score": 0.88},
            {"title": "보이스피싱 대응 가이드", "category": "guide", "relevance_score": 0.82}
        ],
        "tokens_used": 850,
        "latency_ms": 1234,
        "timestamp": datetime.now().isoformat()
    }


def create_demo_metrics():
    """Create demo metrics"""
    return {
        "uptime_seconds": 3600,
        "requests": {
            "total": 150,
            "success": 145,
            "error": 5,
            "success_rate": 0.967
        },
        "latency": {
            "detection": {"mean": 350, "p50": 300, "p95": 600, "p99": 800},
            "rag": {"mean": 1200, "p50": 1000, "p95": 2000, "p99": 2500},
            "total": {"mean": 1550, "p50": 1300, "p95": 2600, "p99": 3300}
        },
        "tokens": {"mean": 800, "p50": 750, "p95": 1200, "p99": 1500},
        "risk_distribution": {"LOW": 45, "MEDIUM": 70, "HIGH": 35}
    }


if __name__ == "__main__":
    main()
