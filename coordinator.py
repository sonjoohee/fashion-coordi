import os
import re
from core.query_parser import QueryParser
from core.search_engine import FashionSearchEngine
from core.harmonizer import OutfitHarmonizer
from core.response_generator import ResponseGenerator

# 전체 추천 파이프라인 중앙 제어기

class FashionPipelineCoordinator:
    def __init__(self):
        print(">> 패션 추천 파이프라인 컴포넌트 초기화 중...")
        self.parser = QueryParser()
        self.engine = FashionSearchEngine()
        self.harmonizer = OutfitHarmonizer(self.engine)
        self.responder = ResponseGenerator()
        print(">> 전체 파이프라인 준비 완료.")

    def _is_foreign_query(self, text: str) -> bool:
        """한글이 한 글자도 포함되어 있지 않고 영문 알파벳이 포함된 경우 외국어 질의로 판별"""
        has_hangul = bool(re.search(r'[가-힣]', text))
        has_alpha = bool(re.search(r'[a-zA-Z]', text))
        return not has_hangul and has_alpha

    def _safe_search(self, query_text: str, category: str, color: str = None, season: str = None, max_price: int = None, top_k: int = 3) -> list:
        """
        🎯 Fallback 탑재 안전 검색:
        1차: 지정 조건으로 검색
        2차: 모자 카테고리 명칭 호환 (headwear <-> hat)
        3차: 복합 색상으로 결과 0개 시 color 필터 해제 (시각 벡터로 검색)
        4차: season 필터 해제
        """
        target_cats = [category]
        if category in ["headwear", "hat"]:
            target_cats = ["headwear", "hat"]

        items = []
        for cat in target_cats:
            items = self.engine.search(
                query_text=query_text,
                category=cat,
                color=color,
                season=season,
                max_price=max_price,
                top_k=top_k
            )
            if items:
                break

        # [Fallback 1] color 조건 때문에 0개인 경우 (예: 복합색상) -> color 해제 후 재검색
        if not items and color:
            for cat in target_cats:
                items = self.engine.search(
                    query_text=f"{color} {query_text}",
                    category=cat,
                    color=None,
                    season=season,
                    max_price=max_price,
                    top_k=top_k
                )
                if items:
                    break

        # [Fallback 2] season 조건 때문에 0개인 경우 -> season 해제 후 재검색
        if not items and season:
            print(f"   ↳ [Fallback] {category}: season={season} 결과 0개 → 계절 필터 해제 후 재검색")
            for cat in target_cats:
                items = self.engine.search(
                    query_text=query_text,
                    category=cat,
                    color=None,
                    season=None,
                    max_price=max_price,
                    top_k=top_k
                )
                if items:
                    break

        return items

    def run(self, user_query: str) -> dict:
        # 🎯 [0단계: 외국어 질의 가드레일 (API 호출 전 즉시 차단)]
        if self._is_foreign_query(user_query):
            return {
                "type": "unsupported_language",
                "query": user_query,
                "tpo": "해당 없음",
                "comment": """죄송합니다. 현재 서비스는 한국어 질의만 지원하고 있습니다. 🇰🇷
한국어로 다시 질문해 주시면 최적의 스타일을 추천해 드릴게요!
(예: '50만원 이하 셔츠 추천해줘', '여름 린넨 셔츠 찾아줘')""",
                "results": [],
                "outfits": []
            }

        # [1단계] LLM 질의 분석
        intent = self.parser.parse(user_query)

        # 🎯 [가드레일 분기: 패션 무관 질의 조기 종료]
        if intent.search_type == "unrelated":
            return {
                "type": "unrelated",
                "query": user_query,
                "tpo": "해당 없음",
                "comment": """죄송하지만 저는 패션 및 의류 스타일링 추천 전문 AI 어시스턴트입니다. 👗·
상황(TPO), 날씨, 장소에 어울리는 옷이나 코디 추천을 질문해 주세요!
• 예시: '여름 휴양지 룩 추천해줘', '결혼식 하객룩 추천해줘', '캐주얼한 반팔 티셔츠 찾아줘'""",
                "results": [],
                "outfits": []
            }

        # LLM이 파싱한 계절값(SS/FW/ALL)을 그대로 사용 (ALL인 경우 시즌 필터 해제)
        detected_season = intent.season if getattr(intent, "season", None) not in ["ALL", None] else None

        full_ctx = f"{user_query} {intent.tpo_summary}".lower()

        # 포멀/격식 TPO 감지 (결혼식, 하객, 장례식, 면접, 출근, 비즈니스 등)
        is_formal = any(w in full_ctx for w in ["결혼", "하객", "장례", "조문", "면접", "출근", "비즈니스", "정장", "포멀"])

        # [단품 검색 분기]
        if intent.search_type == "single":
            slot = intent.slots[0] if intent.slots else None
            if not slot:
                return {"type": "single", "query": user_query, "tpo": intent.tpo_summary, "comment": "상품을 찾지 못했습니다.", "results": []}

            effective_price = slot.max_price or intent.total_budget
            items = self._safe_search(
                query_text=slot.clip_query_en,
                category=slot.category,
                color=slot.color,
                season=detected_season,
                max_price=effective_price,
                top_k=3
            )
            comment = self.responder.generate_single_response(user_query, intent.tpo_summary, items)
            return {
                "type": "single",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "results": items
            }

        # [코디 세트 검색 분기]
        else:
            slot_candidates = {}
            query_ctx = intent.tpo_summary or user_query

            # 1) 기본 의류 슬롯 (상의, 하의 등)
            for slot in intent.slots:
                items = self._safe_search(
                    query_text=slot.clip_query_en,
                    category=slot.category,
                    color=slot.color,
                    season=detected_season,
                    max_price=None,
                    top_k=3
                )
                if items:
                    slot_candidates[slot.slot_name] = items

            # 상의나 하의 슬롯이 LLM 파싱에서 누락된 경우 기본 슬롯 안전 보강
            if "top" not in slot_candidates:
                slot_candidates["top"] = self._safe_search(f"{query_ctx} top shirt", category="top", season=detected_season, top_k=3)
            if "bottom" not in slot_candidates:
                slot_candidates["bottom"] = self._safe_search(f"{query_ctx} pants shorts trousers", category="bottom", season=detected_season, top_k=3)

            # 2) 신발 검색 (격식 TPO: 로퍼/구두 / 캐주얼 TPO: 계절별 키워드 분기)
            # 'sandals'를 계절 무관하게 넣으면 겨울 질의에도 슬라이드·샌들이 1순위로 올라온다.
            if is_formal:
                shoes_query = f"{query_ctx} loafer derby dress shoes oxford"
            elif detected_season == "FW":
                shoes_query = f"{query_ctx} winter boots sneakers shoes"
            elif detected_season == "SS":
                shoes_query = f"{query_ctx} sandals slides sneakers shoes"
            else:
                shoes_query = f"{query_ctx} sneakers shoes"

            shoes_items = self._safe_search(
                query_text=shoes_query,
                category="shoes",
                season=detected_season,
                top_k=2
            )
            slot_candidates["shoes"] = [None] + shoes_items

            # 3) 모자 검색 (격식 TPO: 모자 원천 배제([None]), 캐주얼 TPO: 모자 후보 추가)
            if is_formal:
                slot_candidates["headwear"] = [None]
                headwear_count = 0
            else:
                # 모자도 계절별로 키워드를 나눈다. 'bucket hat'은 여름 모자라서
                # 겨울 질의에 넣으면 방한용이 아닌 햇(sun hat)이 올라온다.
                if detected_season == "FW":
                    headwear_query = f"{query_ctx} beanie knit cap earflap warm hat"
                elif detected_season == "SS":
                    headwear_query = f"{query_ctx} bucket hat ball cap"
                else:
                    headwear_query = f"{query_ctx} cap hat"

                headwear_items = self._safe_search(
                    query_text=headwear_query,
                    category="headwear",
                    season=detected_season,
                    top_k=2
                )
                slot_candidates["headwear"] = [None] + headwear_items
                headwear_count = len([h for h in headwear_items if h])

            print(f">> [후보군 수집] 상의: {len(slot_candidates.get('top', []))}개 | 하의: {len(slot_candidates.get('bottom', []))}개 | 신발: {len(shoes_items)}개 | 모자: {headwear_count}개 (포멀여부: {is_formal})")

            # [3단계] 조화도 채점 및 Vision Re-ranking (total_budget 반영)
            best_outfits = self.harmonizer.select_best_outfits(
                slot_candidates=slot_candidates,
                tpo_context=intent.tpo_summary,
                total_budget=intent.total_budget,
                top_k=2
            )

            # [4단계] 자연어 응답 생성
            if not best_outfits:
                comment = "요청하신 조건(TPO)과 규칙에 부합하는 적합한 상품 조합을 찾지 못했습니다. 다른 스타일이나 아이템으로 다시 검색해 보세요."
            else:
                comment = self.responder.generate_coordination_response(user_query, intent.tpo_summary, best_outfits)
                # 예산 차선책 안내 문구가 있으면 상단 첨부
                budget_notice = best_outfits[0].get("budget_note")
                if budget_notice:
                    comment = f"💡 [안내] {budget_notice}" + chr(10) + chr(10) + comment

            return {
                "type": "coordination",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "outfits": best_outfits
            }
