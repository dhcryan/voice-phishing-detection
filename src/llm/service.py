"""
LLM Service with Langfuse Monitoring
"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import Langfuse
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.LLM_MODEL
        
        # Initialize Langfuse if keys are present
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            self.client = LangfuseOpenAI(
                api_key=self.api_key
            )
            self.langfuse = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST
            )
            logger.info("Langfuse monitoring enabled.")
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.langfuse = None
            logger.warning("Langfuse keys not found. Monitoring disabled.")

    def generate_explanation(self, 
                             detection_result: Dict[str, Any], 
                             legal_context: List[Dict[str, Any]]) -> str:
        """
        Generate a plain Korean explanation of the detection result with legal advice.
        """
        if not self.api_key:
            return "LLM API Key가 설정되지 않아 상세 설명을 생성할 수 없습니다."

        is_fake = detection_result.get("is_fake", False)
        confidence = detection_result.get("confidence", 0.0)
        risk_score = detection_result.get("risk_score", "Unknown")
        
        # Format legal context
        context_str = "\n\n".join([f"- {doc['text']} (출처: {doc['metadata'].get('source', 'Unknown')})" for doc in legal_context])
        
        system_prompt = """
        당신은 보이스피싱 탐지 시스템의 AI 어시스턴트입니다. 
        사용자에게 탐지 결과를 알기 쉽게 설명하고, 관련된 법적 근거를 바탕으로 대응 가이드를 제공해야 합니다.
        
        다음 지침을 따르세요:
        1. 결과가 '가짜(Voice Phishing)'인 경우, 즉시 경고하고 위험성을 알립니다.
        2. 결과가 '진짜(Real)'인 경우, 안심시키되 주의는 필요함을 알립니다.
        3. 제공된 법률 정보를 근거로 인용하여(조항 번호 포함) 법적 대응 방안을 제시하세요.
        4. 전문 용어보다는 일반인이 이해하기 쉬운 언어를 사용하세요.
        """
        
        user_prompt = f"""
        [탐지 결과]
        - 판정: {"보이스피싱 의심 (가짜 음성)" if is_fake else "정상 음성 (진짜 음성)"}
        - 신뢰도: {confidence:.2%}
        - 위험도 등급: {risk_score}
        
        [관련 법률 정보]
        {context_str}
        
        위 정보를 바탕으로 사용자에게 종합적인 분석 리포트를 작성해주세요.
        """
        
        try:
            # Create trace if Langfuse is enabled
            trace_id = None
            if self.langfuse:
                trace = self.langfuse.trace(
                    name="generate_explanation",
                    metadata={"is_fake": is_fake, "risk_score": risk_score}
                )
                trace_id = trace.id
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                name="explanation_generation",
                trace_id=trace_id
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "설명 생성 중 오류가 발생했습니다."

# Singleton instance
llm_service = LLMService()
