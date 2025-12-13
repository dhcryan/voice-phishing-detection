"""
국가법령정보 OPEN API를 이용해서
- 현행법령(시행일 기준) 목록 조회 (lawSearch.do?target=eflaw)
- 각 법령 ID로 본문 조회 (lawService.do?target=eflaw)
- 조문/항/호 텍스트를 RAG용 JSONL로 저장

사용 예시:
python crawl_law_for_rag.py \
  --query 보이스피싱 \
  --query 통신사기 \
  --query 금융사기 \
  --query 전기통신금융사기 \
  --query 전자금융거래 \
  --query 사기 \
  --query 사기죄 \
  --query "피해금 환급" \
  --query 금융회사 \
  --query 지급정지 \
  --query 계좌정지 \
  --out law_rag_voicephishing.jsonl \
  --max-pages 3 \
  --oc (본인 국가법령정보 API oc)
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, List, Any, Iterable

import requests

BASE_SEARCH_URL = "http://www.law.go.kr/DRF/lawSearch.do"
BASE_SERVICE_URL = "http://www.law.go.kr/DRF/lawService.do"


class LawOpenClient:
    def __init__(self, oc: str, delay: float = 0.2):
        """
        :param oc: open.law.go.kr OC 값 (이메일 아이디 부분)
        :param delay: 요청 사이 딜레이(초)
        """
        self.oc = oc
        self.delay = delay

    # ------------------------
    # 1) 목록 검색 (현행법령)
    # ------------------------
    def search_eflaw(
        self,
        query: str,
        display: int = 100,
        max_pages: int = 10,
        search: int = 1,
        sort: str = "efdes",
    ) -> List[Dict[str, Any]]:
        """
        현행법령(시행일) 목록 조회 API
        - target=eflaw, type=JSON
        - query: 검색어 (법령명/본문)
        - search: 1=법령명, 2=본문검색
        """
        all_laws: List[Dict[str, Any]] = []

        # 1페이지 먼저 호출해서 totalCnt 확인
        params = {
            "OC": self.oc,
            "target": "eflaw",
            "type": "JSON",
            "query": query,
            "display": display,
            "page": 1,
            "search": search,
            "sort": sort,
        }

        first = self._get_json(BASE_SEARCH_URL, params)
        law_search = first.get("LawSearch", {})
        total_cnt = int(law_search.get("totalCnt", 0))
        page_size = int(params["display"])

        if total_cnt == 0:
            return []

        total_pages = min(math.ceil(total_cnt / page_size), max_pages)

        print(f"[search_eflaw] query='{query}' total={total_cnt}, pages={total_pages}")

        # 첫 페이지
        all_laws.extend(self._extract_law_list(first))

        # 나머지 페이지
        for page in range(2, total_pages + 1):
            params["page"] = page
            data = self._get_json(BASE_SEARCH_URL, params)
            all_laws.extend(self._extract_law_list(data))

        return all_laws

    @staticmethod
    def _extract_law_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        lawSearch JSON에서 법령 리스트 부분만 꺼내기.
        응답 구조 예시(가이드 기준):contentReference[oaicite:1]{index=1}:
            {
              "LawSearch": {
                "totalCnt": "123",
                "page": "1",
                "law": [
                  { "법령ID": "1747", "법령명한글": "...", ... },
                  ...
                ]
              }
            }
        """
        law_search = data.get("LawSearch", {})
        laws = law_search.get("law", [])
        if isinstance(laws, dict):
            # 검색 결과가 1건일 때 dict로 오는 경우 방지
            laws = [laws]
        return laws

    # ------------------------
    # 2) 본문 조회 (조문 단위)
    # ------------------------
    def get_eflaw_detail_by_id(self, law_id: str) -> Dict[str, Any]:
        """
        현행법령(시행일) 본문 조회 API
        - target=eflaw, ID=법령ID, type=JSON
        """
        params = {
            "OC": self.oc,
            "target": "eflaw",
            "type": "JSON",
            "ID": law_id,
        }
        return self._get_json(BASE_SERVICE_URL, params)

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay)
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            print("JSON decode error. Raw text:", resp.text[:500], file=sys.stderr)
            raise

# --------------------------------
# 3) RAG용 chunk 생성 유틸
# --------------------------------

def iter_law_chunks_from_detail(detail_json: Dict[str, Any], law_header: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    lawService.do (target=eflaw, type=JSON) 응답에서
    조문/항/호 기준 텍스트 chunk를 만들어낸다.

    가이드 기준 주요 필드:contentReference[oaicite:2]{index=2}:
      - 법령명_한글, 법령ID, 시행일자, 소관부처
      - 조문번호, 조문제목, 조문내용
      - 항번호, 항내용
      - 호번호, 호내용
      - 부칙내용, 개정문내용, 제개정이유내용 등

    실제 응답 구조를 한 번 print 해보고
    아래에서 'rows' 부분을 수정해도 된다.
    """

    # 응답이 어떤 형태로 오는지에 따라 수정 필요
    # 아래는 "조문 리스트가 배열로 flatten 되어 있다"는 가정.
    rows: List[Dict[str, Any]] = []

    # 가장 바깥 구조: 예를 들어 {"Law": {...}} 이런 형태일 수도 있다.
    # 안전하게 전체 트리를 순회하면서 "조문내용" 키를 가진 dict들을 모은다.
    def collect_rows(obj: Any):
        if isinstance(obj, dict):
            # 조문/항/호 정보가 담긴 dict라고 가정
            if any(k in obj for k in ["조문내용", "항내용", "호내용", "부칙내용", "제개정이유내용"]):
                rows.append(obj)
            # 재귀적으로 순회
            for v in obj.values():
                collect_rows(v)
        elif isinstance(obj, list):
            for v in obj:
                collect_rows(v)
    collect_rows(detail_json)

    # 법령 전체 공통 메타데이터 추출 (가이드 상 법령 레벨 필드들):contentReference[oaicite:3]{index=3}
    law_meta = {
        "law_id": law_header.get("법령ID"),
        "law_name_ko": law_header.get("법령명_한글") or law_header.get("법령명한글"),
        "law_name_hanja": law_header.get("법령명_한자"),
        "law_name_abbrev": law_header.get("법령명약칭") or law_header.get("법령약칭명"),
        "effective_date": law_header.get("시행일자"),
        "promulgation_date": law_header.get("공포일자"),
        "promulgation_number": law_header.get("공포번호"),
        "ministry": law_header.get("소관부처") or law_header.get("소관부처명"),
    }


    # 조문/항/호별 chunk 생성
    for idx, row in enumerate(rows):
        article_no = row.get("조문번호")
        article_title = row.get("조문제목")
        article_text = row.get("조문내용")

        paragraph_no = row.get("항번호")
        paragraph_text = row.get("항내용")

        sub_no = row.get("호번호")
        sub_text = row.get("호내용")

        supplement_text = row.get("부칙내용")
        revision_reason = row.get("제개정이유내용")
        revision_text = row.get("개정문내용")

        parts = []

        # 제목
        title_line = ""
        if article_no:
            title_line += f"제{article_no}조 "
        if article_title:
            title_line += str(article_title)
        if title_line:
            parts.append(title_line)

        # 본문
        if article_text:
            parts.append(str(article_text))
        if paragraph_no or paragraph_text:
            parts.append(f"항 {paragraph_no}: {paragraph_text}")
        if sub_no or sub_text:
            parts.append(f"호 {sub_no}: {sub_text}")
        if supplement_text:
            parts.append(f"[부칙] {supplement_text}")
        if revision_text:
            parts.append(f"[개정문] {revision_text}")
        if revision_reason:
            parts.append(f"[제·개정이유] {revision_reason}")

        text = "\n".join(p for p in parts if p)

        if not text.strip():
            continue

        # RAG용 하나의 chunk 레코드
        chunk = {
            "id": f"{law_meta['law_id']}_{idx}",
            "title": law_meta["law_name_ko"],
            "text": text,
            "metadata": {
                **law_meta,
                "chunk_index": idx,
                "article_no": article_no,
                "article_title": article_title,
                "paragraph_no": paragraph_no,
                "sub_no": sub_no,
            },
        }
        yield chunk


# --------------------------------
# 4) 전체 파이프라인 (검색 → 본문 → JSONL)
# --------------------------------

def build_rag_corpus(
    oc: str,
    queries: List[str],
    out_path: str,
    max_pages_per_query: int = 5,
) -> None:
    client = LawOpenClient(oc=oc, delay=0.3)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for query in queries:
            # 1) 목록 검색
            laws = client.search_eflaw(
                query=query,
                display=100,
                max_pages=max_pages_per_query,
                search=1,   # 1=법령명 기준 검색
                sort="efdes",  # 시행일자 내림차순
            )

            print(f"[build_rag_corpus] query='{query}' -> {len(laws)} laws")

            # 2) 각 법령에 대해 본문 조회 + chunking
            for law in laws:
                law_id = str(law.get("법령ID") or law.get("법령일련번호"))
                law_name = law.get("법령명한글") or law.get("법령명_한글")

                if not law_id:
                    continue

                print(f"  - Fetch detail: {law_id} ({law_name})")
                try:
                    detail = client.get_eflaw_detail_by_id(law_id)
                except Exception as e:
                    print(f"    ! Failed to fetch detail for law_id={law_id}: {e}", file=sys.stderr)
                    continue

                # 3) 조문/항/호 단위 chunk 생성
                for chunk in iter_law_chunks_from_detail(detail, law):
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# --------------------------------
# 5) CLI 인터페이스
# --------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="국가법령정보 OPEN API → RAG용 JSONL 크롤러")
    parser.add_argument(
        "--oc",
        type=str,
        default=os.environ.get("LAW_OC"),
        help="법제처 Open API OC 값(필수, 환경변수 LAW_OC로도 지정 가능)",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        required=True,
        help="법령 검색 키워드 (여러 번 지정 가능, 예: --query 보이스피싱 --query 전기통신금융사기)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="키워드당 검색할 최대 페이지 수 (기본 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.oc:
        print("ERROR: --oc 또는 환경변수 LAW_OC 를 설정해야 합니다.", file=sys.stderr)
        sys.exit(1)

    print(f"OC={args.oc}, queries={args.query}, out={args.out}")
    build_rag_corpus(
        oc=args.oc,
        queries=args.query,
        out_path=args.out,
        max_pages_per_query=args.max_pages,
    )
    print("done.")
