from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Union, Optional

import tiktoken


def chunk_text_tokenwise(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[str]:
    """
    Token-based chunking using tiktoken.
    - Safe for max context length limits.
    - overlap_tokens helps preserve continuity for retrieval.
    """
    if not text:
        return []

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    chunks = []
    start = 0
    step = chunk_tokens - overlap_tokens

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def split_record_text_only(
    record: Dict[str, Any],
    *,
    model: str = "text-embedding-3-small",
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
    text_key: str = "text",
    id_key: str = "id",
    add_fields_in_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Given one record like:
      {"id": "...", "title": "...", "text": "...", "metadata": {...}}
    Return a list of records where only `text` is chunked.
    Other fields are preserved, and ids are made unique.

    Output record id example:
      "010843_30__c000", "010843_30__c001", ...
    """
    base_id = str(record.get(id_key, ""))
    text = record.get(text_key, "") or ""
    chunks = chunk_text_tokenwise(
        text,
        model=model,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    # If already short enough / empty, return as-is (optionally add metadata fields)
    if len(chunks) <= 1:
        out = deepcopy(record)
        if add_fields_in_metadata:
            out.setdefault("metadata", {})
            out["metadata"] = deepcopy(out["metadata"]) if isinstance(out["metadata"], dict) else {}
            out["metadata"]["text_chunk_index"] = 0
            out["metadata"]["text_chunk_count"] = max(1, len(chunks))
            out["metadata"]["source_record_id"] = base_id
        # Keep original text (or the single chunk if you prefer)
        if chunks:
            out[text_key] = chunks[0]
        return [out]

    # Split into multiple records
    out_records: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        r = deepcopy(record)

        # unique id per chunk
        r[id_key] = f"{base_id}__c{i:03d}"

        # chunked text
        r[text_key] = chunk

        # keep metadata, add chunk info
        if add_fields_in_metadata:
            r.setdefault("metadata", {})
            r["metadata"] = deepcopy(r["metadata"]) if isinstance(r["metadata"], dict) else {}
            r["metadata"]["text_chunk_index"] = i
            r["metadata"]["text_chunk_count"] = len(chunks)
            r["metadata"]["source_record_id"] = base_id
            # optional: store token config for debugging
            r["metadata"]["chunk_tokens"] = chunk_tokens
            r["metadata"]["overlap_tokens"] = overlap_tokens

        out_records.append(r)

    return out_records


def split_records_text_only(
    records: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    model: str = "text-embedding-3-small",
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Accept a single record or a list of records, return flattened list.
    """
    if isinstance(records, dict):
        return split_record_text_only(
            records,
            model=model,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

    out: List[Dict[str, Any]] = []
    for rec in records:
        out.extend(
            split_record_text_only(
                rec,
                model=model,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return out


# -------------------------
# Example usage (your record)
# -------------------------




if __name__ == "__main__":
    sample = {
        "id": "010843_30",
        "title": "군사기지 및 군사시설 보호법 시행령",
        "text": "[부칙] [['부칙 <제34676호,2024.7.9>', '이 영은 2024년 7월 17일부터 시행한다.']]",
        "metadata": {
            "law_id": "010843",
            "law_name_ko": "군사기지 및 군사시설 보호법 시행령",
            "law_name_hanja": None,
            "law_name_abbrev": "군사기지법 시행령",
            "effective_date": "20240717",
            "promulgation_date": "20240709",
            "promulgation_number": "34676",
            "ministry": "국방부",
            "chunk_index": 30,
            "article_no": None,
            "article_title": None,
            "paragraph_no": None,
            "sub_no": None,
        },
    }

    sample =  {
    "id": "009718_5", 
    "title": "공무원 인사기록ㆍ통계 및 인사사무 처리 규정", 
    "text": "[부칙] [['부칙(행정안전부와 그 소속기관 직제) <제20741호,2008.2.29>', '제1조(시행일) 이 영은 공포한 날부터 시행한다. \
        <단서 생략>', '제2조 부터 제5조까지 생략', '제6조(다른 법령의 개정) ① 부터 ⑬ 까지 생략', '  ⑭ 공무원인사기록 및 인사사무처리 규정 일부를 다음과 같이 개정한다.', \
        '  제4조제1항제12호 \"호적초본\"을 \"가족관계등록부의 기본증명서\"로 한다.', '  제6조제3항 중 \"중앙인사위원회는\"을 \"행정안전부장관은\"으로 한다.', \
        '  제6조의2제3항 중 \"중앙인사위원회가\"를 \"행정안전부장관이\"로 한다.', '  제8조제2항 중 \"중앙인사위원회가\"를 \"행정안전부장관이\"로 한다.', \
        '  제8조의3제1항 중 \"중앙인사위원회에\"를 \"행정안전부장관 소속 하에\"로 하고, 같은 조 제2항 중 \"중앙인사위원회가\"를 \"행정안전부장관이\"로 한다.', \
        '  제9조제4항 중 \"행정자치부장관\"을 \"행정안전부장관\"으로 한다. ', '  제15조제5항제2호 중 \"호적등본ㆍ초본\"을 \"가족관계증명서ㆍ기본증명서\"로 한다.', \
        '  제27조제3항 중 \"중앙인사위원회가\"를 \"행정안전부장관이\"로 한다.', '  제28조 중 \"중앙인사위원회에\"를 \"행정안전부장관에게\"로 한다.', \
        '  제29조 중 \"행정자치부\"를 \"행정안전부\"로 한다. ', '  제31조 중 \"중앙인사위원회에\"를 \"행정안전부장관에게\"로 한다.', \
        '  제33조제1항 및 제2항 중 \"중앙인사위원회가\"를 각각 \"행정안전부장관이\"로 한다.', '  별표 2를 다음과 같이 한다.', \
        '  [별표 2]                                                                                            ', \
        '  인사발령을 위한 구비서류(제15조제1항 관련)                                                          ', \
        '┏━━┯━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━┓', '┃순번│발령구분    │구비서류                                    │비고                            ┃', \
        '┠──┼──────┼──────────────────────┼────────────────┨', '┃1.  │신규채용    │인사 및 성과 기록 출력물 (5급 이상은 2통)   │                                ┃', \
        '┃    │            │                                 1통        │                                ┃', '┃    │            │가족관계증명서ㆍ기본증명서        각 1통    \
        │                                ┃', '┃    │            │신원조회확인 회보서              1통        │○시ㆍ구ㆍ읍ㆍ면장 발행         ┃', '┃    │            \
        │최종학력증명서 또는 학력증서 사본 1통       │                                ┃', '┃    │            │경력증명서                    1통           │                                \
        ┃', '┃    │            │채용신체검사서                1통           │○「공무원 채용신체검사 규정」  ┃', '┃    │            │                                            \
        │제3조제1항 각 호의 의료기관     ┃', '┃    │            │                                            │발행                            ┃', '┃    │   \
        │신원조사회보서                1통           │○국가정보원장(3급 이상 및 이에 ┃', '┃    │            │                                   \
        │상당하는 공무원과 고위공무원    ┃', '┃    │            │신원진술서                      2통         │단에 속하는 공무원) 또는 경찰   ┃', '┃    │   \
        │사진(반명함판 상반신 탈모 3매)              │청장(4급 이하 및 이에 상당하    ┃', '┃    │            │                   \
        │는 공무원) 발행                 ┃', '┠──┼─┬────┼──────────────────────┼────────────────┨',\
        '┃2.  │승│일반승진│인사 및 성과 기록 출력물      1통           │                          \
        ┃', '┃    │진│        │일반승진시험합격통지서        1통           │               \
        ┃', '┃    │  ├────┼──────────────────────┼────────────────┨', '┃  \
        │  │기타승진│승진순위표                    1통           │             \
        ┃', '┠──┼─┴────┼──────────────────────┼────────────────┨', \
        '┃3.  │전직        │인사 및 성과 기록 출력물      1통           │○「공무원임용령」 제30조제5호  ┃', \
        '┃    │            │전직시험합격통지서            1통           │  해당자                        ┃', \
        '┃    │            │연구 및 기술직렬의 해당분야 확인서 1통      │                                ┃', \
        '┠──┼─┬────┼──────────────────────┼────────────────┨', '┃4.  │면│의원면직│사직원서(자필)                1통           │                                ┃', \
        '┃    │직├────┼──────────────────────┼────────────────┨', '┃    │  │직권면직│징계위원회동의서ㆍ진단서ㆍ직권              │                                ┃', \
        '┃    │  │        │면직사유설명서 또는 직권면직사유            │                                ┃', \
        '┃    │  │        │를 증명할 서류                1통           │                                ┃', \
        '┃    │  ├────┼──────────────────────┼────────────────┨', \
        '┃    │  │당연퇴직│판결문 사본                   1통           │                                ┃', \
        '┠──┼─┴────┼──────────────────────┼────────────────┨', \
        '┃5.  │강임        │강임동의서(자필) 또는 직제                  │                                ┃', \
        '┃    │            │개폐ㆍ예산감소의 관계서류       1통         │                                ┃', \
        '┠──┼──────┼──────────────────────┼────────────────┨', '┃6.  │징계        │징계의결서 사본               1통           │                                ┃', \
        '┠──┼──────┼──────────────────────┼────────────────┨', '┃7.  │추서        │공적조사서                    1통           │                                ┃', \
        '┃    │            │사망진단서                    1통           │                                ┃', \
            '┃    │            │사망경위서                    1통           │                                ┃', '┠──┼──────┼──────────────────────┼────────────────┨', \
                '┃8.  │휴직 및     │진단서 또는 판결문 사본       1통           │○「공무원 채용신체검사 규정」  ┃', \
                    '┃    │복직        │현역증서 사본, 입영통지서 사본              │제3조제1항 각 호의 의료기관ㆍ   ┃',\
                         '┃    │            │또는 휴직사유를 증명할 만한 서류 1통        │보건소장 및 「공무원연금법」    ┃', \
                            '┃    │            │                                            │에 따라 지정된 공무원 요양기    ┃', '┃    │            │                                            │관 발행                         ┃', \
                                '┃    │            │                                            │                                ┃', '┠──┼──────┼──────────────────────┼────────────────┨', \
                                    '┃9.  │전 출 입    │전출입동의서                  1통           │                                ┃', '┠──┼──────┼──────────────────────┼────────────────┨', \
                                        '┃10. │직위해제    │직위해제사유서                1통           │                                ┃', '┠──┼──────┼──────────────────────┼────────────────┨', \
                                            '┃11. │시보임용    │시보임용단축기간 산출표       1통           │○「공무원임용령」 제25조제1항  ┃', \
                                                '┃    │            │                                            │해당자                          ┃', \
                                                    '┗━━┷━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━┛', '  ⑮ 부터 <105> 까지 생략']]", 
    "metadata": {
        "law_id": "009718", 
        "law_name_ko": "공무원 인사기록ㆍ통계 및 인사사무 처리 규정", 
        "law_name_hanja": None, 
        "law_name_abbrev": "", 
        "effective_date": "20240101", 
        "promulgation_date": "20231212", 
        "promulgation_number": "33962", 
        "ministry": "인사혁신처", 
        "chunk_index": 5, 
        "article_no": None, 
        "article_title": None, 
        "paragraph_no": None, 
        "sub_no": None}
    }

    # chunk_tokens는 embedding 모델 한계(8192)보다 훨씬 작게 잡는 게 안전합니다.
    chunked = split_record_text_only(sample, chunk_tokens=1500, overlap_tokens=250)

    for r in chunked:
        print(r["id"], r["metadata"]["text_chunk_index"], "/", r["metadata"]["text_chunk_count"])
    print(len(chunked))
