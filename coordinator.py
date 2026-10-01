import os
import re
from core.query_parser import QueryParser
from core.search_engine import FashionSearchEngine
from core.harmonizer import OutfitHarmonizer
from core.response_generator import ResponseGenerator

# 전체 추천 파이프라인 중앙 제어기

# 단품 검색은 한 슬롯만 보여주므로 후보를 넉넉히(최대 9개) 노출한다.
# 코디 세트 검색은 슬롯 조합 수가 곱으로 늘어나므로 슬롯별 3개를 유지한다.
SINGLE_SEARCH_TOP_K = 9


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

        # 야외 활동 여부는 질의 파서가 판단한다. 모자는 야외일 때만 후보로 올린다.
        # (키워드 목록으로 잡으면 '12월 캐나다 갈 건데'처럼 야외 단어가 없는 여행 질의를
        #  실내로 오판한다. 장소 분류는 파서가 질의 전체를 보고 하는 편이 정확하다.)
        is_outdoor = bool(getattr(intent, "is_outdoor", False))

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
                top_k=SINGLE_SEARCH_TOP_K
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

            # 쌀쌀한 계절(FW)이면 아우터를 반드시 채운다. 파서가 아우터 슬롯을 빼먹어도
            # 가을·겨울 질의에 상의 한 장만 추천되는 일을 막는다.
            # 검색어에 'padded'(패딩) 같은 한겨울 전용 단어를 박지 않고 query_ctx(TPO 요약)에
            # 맡긴다. FW가 가을과 겨울을 한 덩어리로 묶고 있어, 단어를 박으면 10월 질의에
            # 한겨울 패딩이 올라온다.
            if detected_season == "FW" and not slot_candidates.get("outer"):
                outer_query = f"{query_ctx} tailored blazer coat" if is_formal else f"{query_ctx} jacket coat outerwear"
                outer_items = self._safe_search(
                    query_text=outer_query,
                    category="outer",
                    season=detected_season,
                    top_k=3
                )
                if outer_items:
                    slot_candidates["outer"] = outer_items

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

            # 신발은 필수 착장이라 선택권을 주지 않는다. 예전에는 [None]을 앞에 붙여
            # '신발 없음'도 후보로 뒀지만, 이제 후보가 있으면 반드시 착용한다.
            # 후보가 0개면 슬롯 자체를 비워 두고 신발 없이 구성한다 (없으면 어쩔 수 없다).
            shoes_items = self._safe_search(
                query_text=shoes_query,
                category="shoes",
                season=detected_season,
                top_k=3
            )
            if shoes_items:
                slot_candidates["shoes"] = shoes_items

            # 3) 모자 검색 — 야외 상황에서만 후보로 올린다.
            #    격식 자리(실내 예식 등)는 is_outdoor가 false라 자연히 제외되지만,
            #    파서가 야외로 오판하는 경우를 막기 위해 is_formal도 함께 본다.
            if is_formal or not is_outdoor:
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
                if headwear_items:
                    slot_candidates["headwear"] = headwear_items
                headwear_count = len(headwear_items)

            print(f">> [후보군 수집] 아우터: {len(slot_candidates.get('outer', []))}개 | "
                  f"상의: {len(slot_candidates.get('top', []))}개 | "
                  f"하의: {len(slot_candidates.get('bottom', []))}개 | "
                  f"신발(필수): {len(shoes_items)}개 | 모자(선택): {headwear_count}개 "
                  f"(포멀: {is_formal} / 야외: {is_outdoor} / 계절: {detected_season or 'ALL'})")

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
