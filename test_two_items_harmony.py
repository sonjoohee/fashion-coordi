import os
from pathlib import Path
from core.harmonizer import OutfitHarmonizer
from core.search_engine import FashionSearchEngine

def run_direct_harmony_test():
    print(">> Harmonizer 및 Search Engine 초기화 중...")
    engine = FashionSearchEngine()
    harmonizer = OutfitHarmonizer(engine)

    # 테스트할 상·하의 조합
    sample_outfit = [
        {
            "product_id": "6765396",
            "product_name": "에센셜 썸홀 레이어드 티 (버건디)",
            "brand_name": "주앙옴므",
            "category": "top",
            "colors": ["red"],
            "price": 43200,
            "local_image_path": "outputs/cutouts_v2_4/6765396.png"
        },
        {
            "product_id": "6830540",
            "product_name": "타스 카펜터 치노 팬츠 (빈티지 퍼플)",
            "brand_name": "파브레가",
            "category": "bottom",
            "colors": ["purple"],
            "price": 97200,
            "local_image_path": "outputs/cutouts_v2_4/6830540.png"
        }
    ]

    # 이미지 경로 점검
    for item in sample_outfit:
        if not os.path.exists(item["local_image_path"]):
            print(f"⚠️ 경고: {item['local_image_path']} 파일이 없습니다.")

    # 1. 1차 시각 벡터 유사도 점수 계산
    try:
        vector_score = harmonizer.calculate_vector_harmony(sample_outfit)
    except Exception as e:
        vector_score = f"계산 실패 ({e})"

    # 2. 2차 VLM 순수 의류 조화도 검증 (TPO 질의 없이 호출)
    print("\n" + "=" * 80)
    print("🎯 상의 + 하의 의류 자체 조화도 정밀 검증")
    print(f"- 상의: [{sample_outfit[0]['brand_name']}] {sample_outfit[0]['product_name']} ({sample_outfit[0]['colors']})")
    print(f"- 하의: [{sample_outfit[1]['brand_name']}] {sample_outfit[1]['product_name']} ({sample_outfit[1]['colors']})")
    print(f"★ 1차 벡터 조화도 점수: {vector_score}점")
    print("=" * 80)

    try:
        # tpo_context 인자 없이 순수 옷 조화도만 평가
        verification = harmonizer.evaluate_with_vision(sample_outfit)
        
        status_badge = "✅ PASS (조화로움)" if verification.get("pass_status") else "❌ FAIL (부조화)"
        print(f" - 조화 여부: {status_badge}")
        print(f" - 부조화 원인: {verification.get('violated_rules', [])}")
        print(f" - VLM 총평  : {verification.get('feedback_summary')}")
    except Exception as e:
        print(f" ⚠️ 평가 도중 에러 발생: {e}")

    print("=" * 80)

if __name__ == "__main__":
    run_direct_harmony_test()