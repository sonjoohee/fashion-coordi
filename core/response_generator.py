import os
from openai import OpenAI
from dotenv import load_dotenv
from core.llm_settings import LLM_SEED

# ==============================================================================
# 4단계: 객관적 근거 기반 자연어 응답 생성기 (순위별 분리 브리핑 적용)
# ==============================================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResponseGenerator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def _format_single_items(self, items: list) -> str:
        lines = []
        for idx, it in enumerate(items, 1):
            brand = it.get("brand_name", "")
            name = it.get("product_name", "")
            price = it.get("price", 0)
            colors = it.get("colors") or []
            color_str = f" | 색상: {', '.join(colors)}" if colors else ""
            lines.append(f"{idx}. [{brand}] {name} ({price:,}원{color_str})")
        return "\n".join(lines)

    def _format_outfits(self, outfits: list) -> str:
        lines = []
        for idx, o in enumerate(outfits, 1):
            item_descs = [f"[{it.get('category')}] {it.get('brand_name')} {it.get('product_name')} ({it.get('price', 0):,}원)" for it in o.get("items", [])]
            lines.append(f"★ [코디 세트 {idx}] (총액: {o.get('total_price', 0):,}원 / 조화도: {o.get('harmony_score')}점)")
            lines.append(f" - 구성 아이템: {' + '.join(item_descs)}")
            lines.append(f" - 심사 피드백: {o.get('feedback_summary', '')}\n")
        return "\n".join(lines)

    def generate_single_response(self, query: str, tpo: str, items: list) -> str:
        if not items:
            return "요청하신 조건에 부합하는 상품을 찾지 못했습니다."

        formatted_items = self._format_single_items(items)

        system_prompt = (
            "당신은 패션 커머스 플랫폼의 전문 데이터 큐레이터입니다.\n"
            "감성적인 미사여구보다 사용자의 요청 조건, TPO 적합성, 가성비/가격 메리트 관점에서 "
            "객관적인 추천 근거를 간결하게 제시하세요.\n"
            "[절대 규칙]: 반드시 제공된 목록의 실제 [브랜드]와 상품명만 정확히 인용하세요."
        )

        user_prompt = f"""[사용자 질의]: "{query}"
[상황/TPO]: {tpo}

[실제 추천된 단품 목록]:
{formatted_items}

위 목록의 상품들이 사용자의 질의/상황에 왜 적합한지 객관적인 추천 근거(소재, 활용성, 가격 등)를 들어 2~3문장으로 명확히 안내하세요."""

        res = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            seed=LLM_SEED
        )
        return res.choices[0].message.content

    def generate_coordination_response(self, query: str, tpo: str, outfits: list) -> str:
        if not outfits:
            return "조건에 만족하는 코디 세트를 구성하지 못했습니다."

        formatted_outfits = self._format_outfits(outfits)
        num_outfits = len(outfits)

        system_prompt = (
            "당신은 패션 스타일링 검증 디렉터입니다.\n"
            "추천 코디 세트의 색상 조화, 실루엣, TPO/계절 적합성을 바탕으로 객관적인 추천 이유를 작성하세요.\n"
            "[작성 형식 규칙 - 필수]:\n"
            "1. 추천 세트가 2개인 경우, 반드시 아래 구조로 명확히 나누어 작성하세요:\n"
            "   - **1순위 추천 (메인 코디)**: 1번 세트의 조화도, 완성도, 메인 스타일링 포인트 및 추천 이유\n"
            "   - **2순위 추천 (대안 코디)**: 2번 세트만의 차별점(예: 슈즈 제외 기본 구성, 가성비, 캐주얼한 무드 등)과 추천 이유\n"
            "2. 추천 세트가 1개인 경우, 해당 코디가 선정된 결정적 이유(TPO 적합성, 실루엣, 컬러 매칭)를 집중적으로 설명하세요.\n"
            "3. 제공된 코디 세트의 실제 아이템 정보에만 근거하여 사실적으로 작성하세요."
        )

        user_prompt = f"""[사용자 질의]: "{query}"
[상황/TPO]: {tpo}
[추천 세트 개수]: {num_outfits}개

[검증 완료된 코디 세트 정보]:
{formatted_outfits}

위 정보를 바탕으로, 규칙에 맞추어 스타일리스트 브리핑을 작성하세요."""

        res = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            seed=LLM_SEED
        )
        return res.choices[0].message.content