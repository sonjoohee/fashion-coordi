"""1단계 질의 분석기 비교: OpenAI SDK 버전 vs LangChain 버전.

같은 질의를 두 분석기에 넣고 검색 조건이 같은지 확인한다.
검색 단계(Qdrant)와 심사 단계는 돌리지 않으므로 Qdrant 서버 없이 실행할 수 있다.

실행:
    pip install langchain-openai
    python compare_parsers.py
"""
import json

from core.query_parser import QueryParser
from core.query_parser_lc import LangChainQueryParser

# test_search.py에서 켜 둔 질의 8종 + 패션 무관 가드레일 질의 1종
TEST_QUERIES = [
    "12월 캐나다 갈 건데 옷 추천좀",
    "캐나다 갈건데 옷 추천좀",
    "7월 발리에 입고 갈 옷 추천해줘",
    "8월에 입고 다닐 옷 추천해줘",
    "오늘 날씨에 적합한 옷 추천해줘",
    "2만원 이하 모자 추천해줘",
    "친한 친구 결혼식 깔끔한 하객룩 추천해줘",
    "여름 휴양지에서 편하게 신을 신발 추천해줘",
    "오늘 강남역 근처 분위기 좋은 카페 알려줘",
]

# 검색 결과를 바꾸는 필드만 비교한다.
# tpo_summary와 clip_query_en은 자연어라 표현이 조금만 달라도 다르게 나오므로 따로 출력만 한다.
STRICT_FIELDS = ["search_type", "season", "total_budget", "is_outdoor"]


def slot_signature(intent):
    return sorted(
        (s.category, s.color, s.max_price) for s in intent.slots
    )


def main():
    sdk = QueryParser()
    lc = LangChainQueryParser()
    same_count = 0
    rows = []

    for q in TEST_QUERIES:
        a = sdk.parse(q)
        b = lc.parse(q)
        diffs = [f for f in STRICT_FIELDS if getattr(a, f) != getattr(b, f)]
        if slot_signature(a) != slot_signature(b):
            diffs.append("slots")
        same = not diffs
        same_count += same
        rows.append({
            "query": q,
            "same": same,
            "diffs": diffs,
            "sdk": a.model_dump(),
            "langchain": b.model_dump(),
        })

        print("=" * 80)
        print(f"질의: {q}")
        print(f"  결과: {'일치' if same else '불일치 ' + str(diffs)}")
        for f in STRICT_FIELDS:
            print(f"  {f:13s} SDK={getattr(a, f)!r:12} LangChain={getattr(b, f)!r}")
        print(f"  slots         SDK={slot_signature(a)}")
        print(f"                LangChain={slot_signature(b)}")
        for sa, sb in zip(a.slots, b.slots):
            print(f"  clip_query_en [{sa.category}] SDK: {sa.clip_query_en}")
            print(f"                [{sb.category}] LC : {sb.clip_query_en}")

    print("=" * 80)
    print(f"검색 조건 일치: {same_count}/{len(TEST_QUERIES)}")

    with open("compare_parsers_result.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("상세 결과: compare_parsers_result.json")


if __name__ == "__main__":
    main()
