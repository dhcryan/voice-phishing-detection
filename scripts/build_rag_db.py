import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.engine import legal_rag
from src.utils.chunking import split_record_text_only

# Sample legal data related to Voice Phishing (Telecommunications-based Financial Fraud)
# In a real scenario, this would be fetched from the National Law Information Center API
LAW_DATA = [
    {
        "id": "criminal_act_347",
        "text": "형법 제347조(사기) ①사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처한다. ②전항의 방법으로 제삼자로 하여금 재물의 교부를 받게 하거나 재산상의 이익을 취득하게 한 때에도 전항의 형과 같다.",
        "metadata": {"source": "형법", "article": "제347조", "category": "사기"}
    },
    {
        "id": "criminal_act_347_2",
        "text": "형법 제347조의2(컴퓨터등 사용사기) 컴퓨터등 정보처리장치에 허위의 정보 또는 부정한 명령을 입력하거나 권한 없이 정보를 입력ㆍ변경하여 정보처리를 하게 함으로써 재산상의 이익을 취득하거나 제3자로 하여금 취득하게 한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처한다.",
        "metadata": {"source": "형법", "article": "제347조의2", "category": "사기"}
    },
    {
        "id": "telecom_fraud_act_1",
        "text": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법 제1조(목적) 이 법은 전기통신금융사기를 예방하고 그 피해금을 신속하게 환급하는 등 피해자를 구제함으로써 금융거래의 안전성과 신뢰성을 확보하고 국민생활의 안정에 이바지함을 목적으로 한다.",
        "metadata": {"source": "통신사기피해환급법", "article": "제1조", "category": "보이스피싱"}
    },
    {
        "id": "telecom_fraud_act_3",
        "text": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법 제3조(피해구제의 신청) ① 피해자는 피해금을 송금ㆍ이체한 계좌를 관리하는 금융회사 또는 사기이용계좌를 관리하는 금융회사에 대하여 사기이용계좌의 지급정지 등 피해구제를 신청할 수 있다.",
        "metadata": {"source": "통신사기피해환급법", "article": "제3조", "category": "피해구제"}
    },
    {
        "id": "telecom_fraud_act_15_2",
        "text": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법 제15조의2(벌칙) 전기통신금융사기를 행한 자는 1년 이상의 유기징역 또는 범죄수익의 3배 이상 5배 이하에 상당하는 벌금에 처한다.",
        "metadata": {"source": "통신사기피해환급법", "article": "제15조의2", "category": "처벌"}
    },
    {
        "id": "electronic_financial_act_6",
        "text": "전자금융거래법 제6조(접근매체의 선정과 사용 및 관리) ③ 누구든지 접근매체를 사용 및 관리함에 있어서 다른 법률에 특별한 규정이 없는 한 다음 각 호의 행위를 하여서는 아니 된다. 1. 접근매체를 양도하거나 양수하는 행위 2. 대가를 수수(授受)ㆍ요구 또는 약속하면서 접근매체를 대여받거나 대여하는 행위",
        "metadata": {"source": "전자금융거래법", "article": "제6조", "category": "대포통장"}
    }
]

def load_external_data(file_path: str):
    """Load legal data from an external JSON file."""
    if not os.path.exists(file_path):
        return None
    try:
        import json
        data_list = []
        data_dict = {}
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append( json.loads(line) )

        # id 중복 처리
        data_ids_set = set([doc["id"] for doc in data_list])
        for d in data_list:
            if data_dict.get(d["id"]) is None:
                data_dict[d["id"]] = d
                # 청킹
                chunked_data = split_record_text_only(d, chunk_tokens=1500, overlap_tokens=250)
                data = data + chunked_data
        
        print(f"chunked result: {len(data_ids_set)} -> {len(data)}")

        if isinstance(data, list) and len(data) > 0 and "text" in data[0]:
            return data
        else:
            print(f"Warning: Invalid format in {file_path}. Expected a list of objects with 'text' field.")
            return None
    except Exception as e:
        print(f"Error loading external data from {file_path}: {e}")
        return None

def main():
    # 1. Try to load real data from data/raw/laws.json
    # data/lawopendata/law_rag_voicephishing.jsonl
    # data/raw/laws.json
    external_data_path = os.path.join(Path(__file__).parent.parent, "data/lawopendata/law_rag_voicephishing.jsonl")
    data_to_ingest = load_external_data(external_data_path)
    
    if data_to_ingest:
        print(f"Found external legal data at {external_data_path}")
        print(f"Ingesting {len(data_to_ingest)} documents...")
    else:
        print("External data not found or invalid. Using default sample data...")
        data_to_ingest = LAW_DATA
        print(f"Ingesting {len(data_to_ingest)} sample documents...")

    try:
        legal_rag.ingest_documents(data_to_ingest)
        print("Successfully built RAG database.")
        print(f"Database location: {os.path.abspath('data/chromadb')}")
    except Exception as e:
        print(f"Error building RAG database: {e}")

if __name__ == "__main__":
    main()
