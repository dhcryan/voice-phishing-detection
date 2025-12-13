"""
Voice Phishing Detection - RAG System
Legal document retrieval and generation for Korean voice phishing laws
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import hashlib
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv(verbose=True)

@dataclass
class Document:
    """Document structure for RAG"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    documents: List[Document]
    query: str
    # scores: List[float]
    metas: List[dict]
    total_tokens: int


@dataclass
class RAGResponse:
    """Complete RAG response"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_result: RetrievalResult
    generation_tokens: int
    total_latency_ms: float


class LegalDocumentLoader:
    """Load and parse Korean legal documents"""
    
    # Korean voice phishing related laws
    LEGAL_DOCUMENTS = {
        "criminal_code_347": {
            "title": "형법 제347조 (사기)",
            "content": """
제347조(사기) 
① 사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처한다.
② 전항의 방법으로 제삼자로 하여금 재물의 교부를 받게 하거나 재산상의 이익을 취득하게 한 때에도 전항의 형과 같다.

[해설]
- 보이스피싱의 기본 적용 법조항
- 기망행위: 거짓 신분(수사기관, 금융기관 사칭), 허위 사실 고지
- 재물 교부: 현금 이체, 계좌 정보 취득 등
- 형량: 10년 이하 징역 또는 2천만원 이하 벌금
""",
            "category": "criminal",
            "keywords": ["사기", "기망", "재물", "징역", "벌금", "보이스피싱"]
        },
        "criminal_code_347_2": {
            "title": "형법 제347조의2 (컴퓨터등 사용사기)",
            "content": """
제347조의2(컴퓨터등사용사기)
컴퓨터등 정보처리장치에 허위의 정보 또는 부정한 명령을 입력하거나 권한 없이 정보를 입력·변경하여 정보처리를 하게 함으로써 재산상의 이익을 취득하거나 제3자로 하여금 취득하게 한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처한다.

[해설]
- 전자금융 관련 보이스피싱에 적용
- 원격제어 앱을 통한 계좌 조작
- 피해자 명의 대출 실행 등
""",
            "category": "criminal",
            "keywords": ["컴퓨터", "정보처리", "전자금융", "원격제어", "대출"]
        },
        "telecom_fraud_special_act": {
            "title": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법",
            "content": """
제1조(목적)
이 법은 전기통신을 이용한 금융사기 범죄로 인한 피해를 예방하고 피해자의 재산적 피해를 신속히 회복하는 것을 목적으로 한다.

제2조(정의)
1. "전기통신금융사기"란 전기통신을 이용하여 타인을 기망·공갈함으로써 재산상의 이익을 취하거나 제3자에게 재산상의 이익을 취하게 하는 다음 각 목의 행위를 말한다.
   가. 자금을 송금·이체하도록 하는 행위
   나. 개인정보를 알아내어 자금을 송금·이체하는 행위

제3조(지급정지)
① 금융회사등은 피해자 또는 수사기관으로부터 피해 구제의 신청을 받은 경우 지체 없이 해당 사기이용계좌에 대한 지급정지 조치를 하여야 한다.
② 지급정지 요청시 필요한 정보: 피해자 인적사항, 피해금액, 송금일시, 사기이용계좌 정보

제4조(피해금 환급)
① 금융회사등은 지급정지된 사기이용계좌의 채권소멸절차가 완료된 경우 피해자에게 피해금을 환급하여야 한다.
② 환급 비율은 지급정지된 금액과 피해자별 피해액을 기준으로 산정

제13조(명의대여 금지 및 처벌)
누구든지 전기통신금융사기에 이용될 것을 알면서 접근매체(통장, 체크카드 등)를 대여하여서는 아니 된다.
위반시: 3년 이하의 징역 또는 3천만원 이하의 벌금

[피해 구제 절차]
1. 피해 인지 즉시 경찰(112) 또는 금융감독원(1332) 신고
2. 해당 금융기관 고객센터로 지급정지 요청
3. 신분증, 피해사실확인서 등 서류 제출
4. 채권소멸절차(2개월) 후 환급 심사
5. 피해액 환급 (잔액 범위 내)
""",
            "category": "telecom_fraud",
            "keywords": ["전기통신금융사기", "지급정지", "환급", "피해구제", "신고", "보이스피싱"]
        },
        "electronic_finance_act": {
            "title": "전자금융거래법",
            "content": """
제6조(접근매체의 선정과 사용 및 관리)
① 금융회사등은 전자금융거래를 위하여 접근매체를 선정하거나 사용 및 관리하는 경우 이용자의 신원, 권한 및 거래지시의 내용 등을 확인하여야 한다.
② 이용자는 접근매체를 제3자에게 대여하거나 양도 또는 담보의 목적으로 제공하여서는 아니 된다.

제9조(금융회사등의 책임)
① 금융회사등은 접근매체의 위조나 변조로 발생한 사고, 계약체결 또는 거래지시의 전자적 전송이나 처리과정에서 발생한 사고로 인하여 이용자에게 손해가 발생한 경우에는 그 손해를 배상할 책임을 진다.
② 다만, 이용자의 고의나 중대한 과실이 있는 경우에는 그 책임의 전부 또는 일부를 이용자가 부담할 수 있다.

제10조(이용자의 책임)
① 이용자는 접근매체를 제3자에게 대여하거나 그 사용을 위임한 경우 또는 접근매체를 제3자에게 양도한 경우에는 그로 인하여 발생하는 손해에 대하여 책임을 진다.

[보이스피싱 관련 해석]
- 피해자가 기망에 의해 접근매체 정보를 제공한 경우
- 이용자의 고의·중과실 여부 판단이 핵심
- 최근 판례는 피해자 보호 강화 추세
""",
            "category": "electronic_finance",
            "keywords": ["접근매체", "전자금융", "손해배상", "책임", "이용자"]
        },
        "aggravated_punishment_act": {
            "title": "특정경제범죄 가중처벌 등에 관한 법률 제3조",
            "content": """
제3조(특정재산범죄의 가중처벌)
① 「형법」 제347조(사기)의 죄를 범한 자는 그 범죄행위로 인하여 취득하거나 제3자로 하여금 취득하게 한 재물 또는 재산상 이익의 가액(이하 이 조에서 "이득액"이라 한다)이 다음 각 호의 어느 하나에 해당하는 경우에는 각 호에서 정한 형에 처한다.

1. 이득액이 50억원 이상인 때: 무기 또는 5년 이상의 징역
2. 이득액이 5억원 이상 50억원 미만인 때: 3년 이상의 유기징역

② 제1항의 경우 이득액 이하에 상당하는 벌금을 병과할 수 있다.

[적용 예시]
- 조직적 보이스피싱: 피해액 합산시 가중처벌 대상
- 총책, 콜센터 운영자 등 주도적 역할자에게 적용
- 인출책, 전달책 등 하위 조직원은 일반 사기죄 적용 가능
""",
            "category": "aggravated_punishment",
            "keywords": ["가중처벌", "특정경제범죄", "이득액", "무기징역", "조직"]
        },
        "response_guide": {
            "title": "보이스피싱 대응 가이드",
            "content": """
[보이스피싱 의심 상황 즉시 대응]

1. 즉시 조치
   - 통화 즉시 종료
   - 어떤 정보도 제공하지 말 것
   - 금융거래 중단

2. 신고 및 지급정지
   - 경찰청: 112
   - 금융감독원: 1332
   - 해당 금융기관 고객센터
   - 계좌 지급정지 요청 (24시간 이내가 중요)

3. 이미 송금한 경우
   - 즉시 112 신고 후 사건번호 확보
   - 송금 은행에 지급정지 요청
   - 피해사실확인서 발급
   - 금융감독원 피해상담 (1332)

4. 증거 보전
   - 통화 녹음 (가능한 경우)
   - 문자메시지 캡처
   - 계좌 거래내역 확보
   - 피해 경위서 작성

5. 피해 구제
   - 지급정지 신청 (피해금 동결)
   - 채권소멸절차 (약 2개월)
   - 피해환급 신청
   - 민사소송 고려 (범인 검거시)

[주요 연락처]
- 경찰청 사이버수사국: 182
- 금융감독원 보이스피싱 상담: 1332
- 검찰청: 1301
- 한국인터넷진흥원: 118
""",
            "category": "guide",
            "keywords": ["대응", "신고", "지급정지", "피해구제", "연락처", "증거"]
        }
    }
    
    def __init__(self, docs_path: str = "data/legal_docs"):
        self.docs_path = Path(docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        
    def get_documents(self) -> List[Document]:
        """Load all legal documents"""
        documents = []
        
        for doc_id, doc_data in self.LEGAL_DOCUMENTS.items():
            doc = Document(
                id=doc_id,
                content=doc_data["content"],
                metadata={
                    "title": doc_data["title"],
                    "category": doc_data["category"],
                    "keywords": doc_data["keywords"]
                }
            )
            documents.append(doc)
        
        # Also load any external documents
        external_docs = self._load_external_documents()
        documents.extend(external_docs)
        
        return documents
    
    def _load_external_documents(self) -> List[Document]:
        """Load documents from files"""
        documents = []
        
        for file_path in self.docs_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = Document(
                        id=data.get("id", file_path.stem),
                        content=data.get("content", ""),
                        metadata=data.get("metadata", {})
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {e}")
        
        return documents
    
    def save_document(self, doc: Document):
        """Save document to file"""
        file_path = self.docs_path / f"{doc.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False, indent=2)


class VectorStore:
    """Simple vector store using FAISS"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
        index_path: str = "data/vectors"
    ):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.documents: List[Document] = []
        self.openai_client = None
        
    def _get_openai_client(self):
        """Lazy load OpenAI client"""
        if self.openai_client is None:
            from openai import OpenAI
            self.openai_client = OpenAI()
        return self.openai_client
    
    def _embed_text(self, text: str) -> List[float]:
        """Get embedding for text"""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index from documents"""
        import faiss
        import numpy as np
        
        self.documents = documents
        
        # Get embeddings
        texts = [doc.content for doc in documents]
        embeddings = self._embed_texts(texts)
        
        # Store embeddings in documents
        for doc, emb in zip(self.documents, embeddings):
            doc.embedding = emb
        
        # Build FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        logger.info(f"Built index with {len(documents)} documents")
        
    def save_index(self, name: str = "legal_docs"):
        """Save index to disk"""
        import faiss
        
        index_file = self.index_path / f"{name}.index"
        docs_file = self.index_path / f"{name}_docs.json"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_file))
        
        # Save documents
        docs_data = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            }
            for doc in self.documents
        ]
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False)
        
        logger.info(f"Saved index to {index_file}")
    
    def load_index(self, name: str = "legal_docs") -> bool:
        """Load index from disk"""
        import faiss
        
        index_file = self.index_path / f"{name}.index"
        docs_file = self.index_path / f"{name}_docs.json"
        
        if not index_file.exists() or not docs_file.exists():
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Load documents
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        
        self.documents = [
            Document(
                id=d["id"],
                content=d["content"],
                metadata=d["metadata"],
                embedding=d.get("embedding")
            )
            for d in docs_data
        ]
        
        logger.info(f"Loaded index with {len(self.documents)} documents")
        return True
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> Tuple[List[Document], List[float]]:
        """Search for relevant documents"""
        import numpy as np
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get query embedding
        query_embedding = np.array([self._embed_text(query)]).astype('float32')
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Filter and return
        results = []
        result_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                results.append(self.documents[idx])
                result_scores.append(float(score))
        
        return results, result_scores


class LegalRAG:
    """RAG system for legal document retrieval and generation"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        self.vector_store = vector_store

        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from chromadb.utils import embedding_functions
        from src.config import settings
        import chromadb
        import os, chromadb

        
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.RAG_EMBEDDING_MODEL
        )

        persist_dir = os.path.abspath("./data/chromadb")
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_collection(
            name=settings.RAG_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

        self.llm_model = llm_model
        self.temperature = temperature
        self.openai_client = None
        
    def _get_openai_client(self):
        """Lazy load OpenAI client"""
        if self.openai_client is None:
            from openai import OpenAI
            self.openai_client = OpenAI()
        return self.openai_client
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> RetrievalResult:
        """Retrieve relevant documents"""
        import tiktoken
        
        # documents, scores = self.vector_store.search(query, top_k)
        results = self.collection.query(
            query_texts=[query],
            n_results=20
        )
        documents = results["documents"][0]
        print("??")
        print(results["distances"])
        # print(results["documents"][0])
        # print(results["documents"][0][0])
        print("??")
        # Count tokens
        enc = tiktoken.encoding_for_model(self.llm_model)
        total_tokens = sum(len(enc.encode(doc)) for doc in documents)

        return RetrievalResult(
            documents=results["documents"][0],
            query=query,
            # scores=scores,
            metas=results["metadatas"][0],
            total_tokens=total_tokens
        )
    
    def generate(
        self,
        query: str,
        context: str,
        risk_level: str = "MEDIUM",
        detection_summary: str = ""
    ) -> Tuple[str, int]:
        """Generate response using LLM"""
        
        system_prompt = """당신은 보이스피싱 관련 법률 전문 AI 상담사입니다.

규칙:
1. 반드시 제공된 법령 내용만 인용하세요.
2. 조항 번호를 명시하세요 (예: [형법 제347조])
3. 실질적이고 구체적인 대응 방안을 제시하세요.
4. 피해자 보호와 구제를 최우선으로 고려하세요.
5. 불확실한 내용은 전문가 상담을 권고하세요."""

        user_prompt = f"""## 탐지 컨텍스트
- 리스크 레벨: {risk_level}
- 탐지 결과 요약: {detection_summary}

## 관련 법령
{context}

## 사용자 질문
{query}

## 요청사항
위 법령을 참조하여 질문에 답변하세요. 반드시 관련 조항을 인용하고, 구체적인 대응 절차를 안내하세요."""

        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=2048
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return answer, tokens_used
    
    def query(
        self,
        question: str,
        risk_level: str = "MEDIUM",
        detection_summary: str = "",
        top_k: int = 5
    ) -> RAGResponse:
        """Complete RAG query: retrieve + generate"""
        import time
        
        start_time = time.time()
        
        # Retrieve
        retrieval_result = self.retrieve(question, top_k)

        # Build context
        context_parts = []
        sources = []
        
        for doc, meta in zip(retrieval_result.documents, retrieval_result.metas):
            context_parts.append(f"### {f"{meta["law_name_ko"]}"}\n{doc}")
            sources.append({
                "id": meta.get("source_record_id"),
                "title": meta.get("law_name_ko"),
                "article_title": meta.get("article_title", "unknown"),
                # "relevance_score": score
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate
        answer, gen_tokens = self.generate(
            question,
            context,
            risk_level,
            detection_summary
        )
        
        total_latency = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_result=retrieval_result,
            generation_tokens=gen_tokens,
            total_latency_ms=total_latency
        )

def create_rag_system(
    docs_path: str = "data/legal_docs",
    vector_path: str = "data/vectors",
    llm_model: str = "gpt-4o-mini"
) -> LegalRAG:
    """Factory function to create RAG system"""
    
    # Load documents
    loader = LegalDocumentLoader(docs_path)
    documents = loader.get_documents()
    
    # Create vector store
    vector_store = VectorStore(index_path=vector_path)
    
    # Try to load existing index
    if not vector_store.load_index():
        # Build new index
        logger.info("Building new vector index...")
        vector_store.build_index(documents)
        vector_store.save_index()
    
    # Create RAG
    rag = LegalRAG(vector_store, llm_model)
    
    return rag
