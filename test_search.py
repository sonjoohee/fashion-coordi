from coordinator import FashionPipelineCoordinator

def main():
    coordinator = FashionPipelineCoordinator()

    test_queries = [
        # # 1. 단품 질의
        # "오늘 날씨에 딱 맞는 아우터 추천해줘",
        # "여자친구랑 피크닉 갈 때 신을 신발 추천해줘",
        
        # 2. 코디 질의
        # "동남아 여행룩 추천해줘",
        # "결혼식 하객룩 추천해줘"
        # "운동할 때 입을 옷 추천해줘",
        # "여름 휴양지 룩 추천해줘",
        # "내일 친구랑 놀러가는데 의상 추천해줘",
        # "오늘 점심 뭐 먹지"
        # 1. 패션 무관 가드레일 검증 (검색/비전 없이 즉시 조기 종료되는지)
        # "오늘 강남역 근처 분위기 좋은 카페 알려줘",
        "12월 캐나다 갈 건데 옷 추천좀",
        "캐나다 갈건데 옷 추천좀",
        "7월 발리에 입고 갈 옷 추천해줘",
        "8월에 입고 다닐 옷 추천해줘",
        "오늘 날씨에 적합한 옷 추천해줘",
        # "Can you recommend me shirts under 500000won?",
        "2만원 이하 모자 추천해줘",
        # # 2. 격식도 & 모자 원천 배제 검증 (모자 제외, 단정한 로퍼/슬랙스 매칭되는지)
        "친한 친구 결혼식 깔끔한 하객룩 추천해줘",

        # # 3. 계절 및 소재 검증 (울·니트·기모 탈락, 시원한 여름 소재 통과되는지)
        # "베트남 나트랑 여행 갈 때 입을 시원한 바캉스룩",

        # # 4. 색상 및 톤 조화 검증 (무채색 베이스의 안정적인 톤온톤/모노톤 구성인지)
        # "블랙과 화이트 중심의 깔끔한 모노톤 셋업 추천해줘",

        # # 5. 단품 검색 분기 검증 (상황 수식어가 있어도 신발 단품만 3개 추출되는지)
        "여름 휴양지에서 편하게 신을 신발 추천해줘",
        # "야구장 갈 때 쓸 모자 추천해줘"
    ]

    for q in test_queries:
        print("\n" + "="*80)
        print(f"사용자 질의: \"{q}\"")
        print("="*80)

        output = coordinator.run(q)

        print(f">> [TPO 분석]: {output['tpo']}")
        print(f">> [AI 스타일리스트 코멘트]:\n{output['comment']}\n")

        if output["type"] == "single":
            print(">> [추천 단품 목록]")
            for idx, it in enumerate(output["results"], 1):
                print(f" {idx}. [유사도: {it['similarity_score']}] [ID: {it['product_id']}] [{it['brand_name']}] {it['product_name']}")
                print(f"    - 가격: {it['price']:,}원 | 색상: {it['colors']} | 링크: {it['product_url']}")
        # else:
        #     print(">> [추천 코디 세트]")
        #     for c_idx, outfit in enumerate(output["outfits"], 1):
        #         print(f"\n  ★ [코디 셋 #{c_idx}] 조화도: {outfit['harmony_score']}점 | 총액: {outfit['total_price']:,}원")
        #         print(f"     - 조합 총평: {outfit['verdict']}")
        #         print(f"     - 스타일링 팁: {outfit['styling_tip']}")
        #         for it in outfit["items"]:
        #             print(f"     • [{it['category'].upper()}] [ID: {it['product_id']}] {it['brand_name']} - {it['product_name']}")
        #             print(f"       - 가격: {it['price']:,}원 | 색상: {it['colors']}")
        #             print(f"       - 링크: {it['product_url']}")
        else:
            print(">> [추천 코디 세트]")
            if not output.get("outfits"):
                print("  ⚠️ 규칙 및 TPO 조건을 만족하는 적합한 코디 조합이 없습니다.")
            else:
                for c_idx, outfit in enumerate(output["outfits"], 1):
                    print(f"\n  ★ [코디 셋 #{c_idx}] 조화도: {outfit['harmony_score']}점 | 총액: {outfit['total_price']:,}원")
                    print(f"     - 조합 총평: {outfit['verdict']}")
                    print(f"     - 스타일링 팁: {outfit['styling_tip']}")
                    for it in outfit["items"]:
                        print(f"     • [{it['category'].upper()}] [ID: {it['product_id']}] {it['brand_name']} - {it['product_name']}")
                        print(f"       - 가격: {it['price']:,}원 | 색상: {it['colors']}")
                        print(f"       - 링크: {it['product_url']}")
                    
if __name__ == "__main__":
    main()