import os
import base64
import numpy as np
from itertools import combinations, product
from typing import List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ==============================================================================
# 3단계: 조화도 채점 & Vision LLM 규칙 검증기 (상의, 하의, 신발, 모자 통합 지원)
# ==============================================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 튜닝 상수 -----------------------------------------------------------------
FAIL_SCORE_CAP = 45.0       # 규칙 위반 세트의 점수 상한
NEUTRAL_COLORS = {"black", "white", "gray", "grey", "charcoal", "ivory"}

# 신발·모자는 모두 선택품(optional)이다.
# 카테고리가 다른 아이템 간 CLIP 유사도는 구조적으로 낮게 나와 절대 임계값으로는
# '어울림'과 '안 어울림'을 가를 수 없다. 그래서 잡화 착용 여부를 벡터 점수로 자르지 않고,
# 착용 조합별 변형(variant)을 만들어 명시적 규칙을 가진 VLM이 판단하게 한다.
ACCESSORY_VARIANT_CORES = 2   # 잡화 변형을 생성할 상위 코어 조합 수
MAX_VLM_EVALS = 10            # VLM 호출 수 상한 (비용 통제)

CORE_ORDER = ("outer", "top", "bottom")     # 코어 의류 (조합 뼈대)
ACC_ORDER = ("shoes", "headwear")           # 잡화 (모두 선택품)
SLOT_ALIASES = {"hat": "headwear"}          # 동일 부위 다른 명칭 통합
ITEM_ID_KEYS = ("product_id", "id", "goods_no")

class OutfitEvaluation(BaseModel):
    """Vision LLM 정밀 심사 결과 스키마.

    주의: Structured Outputs strict 모드는 숫자 범위 제약(ge/le)을 지원하지 않는다.
    harmony_score 범위는 프롬프트로 유도하고 최종 클램프는 코드에서 수행한다.
    """
    pass_status: bool = Field(
        description="규칙 위반 여부 판정 (위반 사항이 없거나 무난히 어울리면 True, 심각한 부적합 시 False)"
    )
    harmony_score: float = Field(
        description="코디 종합 조화도 점수 (0~100). 통과 시 60~100, 탈락 시 50 이하"
    )
    violated_rules: List[Literal["season", "color", "tpo", "fit", "formality"]] = Field(
        default_factory=list,
        description="위반된 규칙 목록 (season, color, tpo, fit, formality 중 해당 항목)"
    )
    feedback_summary: str = Field(
        description="판정 근거를 한 문장으로 요약 (아이템 간 조화, 색상/톤 충돌, TPO 중심)"
    )

# 구버전 이름으로 import하는 모듈 호환용 별칭
OutfitRuleVerification = OutfitEvaluation

class OutfitHarmonizer:
    def __init__(self, search_engine):
        self.engine = search_engine
        self._vector_cache = {}   # image_path -> np.ndarray (조합마다 재인코딩 방지)

    # ------------------------------------------------------------------
    # 입력 정규화
    # ------------------------------------------------------------------
    @staticmethod
    def _item_price(item: dict) -> int:
        """price가 None·문자열이어도 안전하게 정수 반환"""
        raw = (item or {}).get("price")
        if raw is None:
            return 0
        try:
            return int(float(str(raw).replace(",", "").strip()))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _item_key(item: dict) -> str:
        """중복 제거용 식별자"""
        for key in ITEM_ID_KEYS:
            value = (item or {}).get(key)
            if value:
                return str(value)
        return f"{(item or {}).get('brand_name', '')}|{(item or {}).get('product_name', '')}"

    def _normalize_slot(self, raw) -> List[dict]:
        """슬롯 값을 항상 list[dict]로 평탄화.

        None 센티넬('미착용' 표현), 단일 dict, 중첩 리스트, {"items": [...]} 래핑을 모두 흡수한다.
        잡화 미착용은 이 계층에서 센티넬이 아니라 '후보 없음'으로 표현되며,
        실제 착용 여부는 select_best_outfits가 만든 변형을 VLM이 심사해 결정한다.
        """
        if raw is None:
            return []
        if isinstance(raw, dict):
            if any(key in raw for key in ITEM_ID_KEYS + ("product_name",)):
                return [raw]
            for wrapper in ("items", "candidates", "results", "products"):
                if isinstance(raw.get(wrapper), list):
                    return self._normalize_slot(raw[wrapper])
            return []
        if isinstance(raw, (list, tuple, set)):
            flat = []
            for element in raw:
                flat.extend(self._normalize_slot(element))
            return flat
        return []   # 문자열·숫자 등 아이템이 아닌 값은 폐기

    def _core_key(self, core_items: List[dict]) -> tuple:
        """같은 코어 의류에서 파생된 변형들을 묶기 위한 키 (중복 추천 방지용)"""
        return tuple(sorted(self._item_key(item) for item in core_items))

    def _pick_best_accessory(self, base_items: List[dict], candidates: List[dict],
                             current_price: int, total_budget: Optional[int],
                             is_fallback_budget: bool):
        """후보 중 벡터 조화도가 가장 높은 잡화 1개를 대표로 선정한다.

        절대 임계값으로 탈락시키지 않는다. 아이템 수가 늘면 카테고리가 다른 쌍이
        평균을 끌어내려 어울리는 잡화도 점수가 떨어지기 때문에, 이 점수는
        '같은 슬롯 후보 간 상대 비교'에만 쓰고 착용 여부는 VLM이 판단한다.
        """
        best_item, best_score = None, -1.0
        for candidate in candidates:
            price = self._item_price(candidate)
            if total_budget and not is_fallback_budget and (current_price + price > total_budget):
                continue
            score = self.calculate_vector_harmony(base_items + [candidate])
            if score > best_score:
                best_score, best_item = score, candidate
        return best_item, best_score

    def _normalize_slot_candidates(self, slot_candidates: Optional[dict]) -> dict:
        """슬롯 키 별칭 통합(hat→headwear) + 값 정규화 + 슬롯 내 중복 아이템 제거"""
        merged = {}
        for raw_key, raw_value in (slot_candidates or {}).items():
            canonical = SLOT_ALIASES.get(raw_key, raw_key)
            bucket = merged.setdefault(canonical, [])
            seen = {self._item_key(existing) for existing in bucket}
            for item in self._normalize_slot(raw_value):
                key = self._item_key(item)
                if key in seen:
                    continue
                seen.add(key)
                bucket.append(item)
        return {key: value for key, value in merged.items() if value}

    # ------------------------------------------------------------------
    # 이미지 / 벡터
    # ------------------------------------------------------------------
    def _resolve_image_path(self, item: dict) -> Optional[str]:
        """outputs/cutouts_v2_4 경로 우선 탐색"""
        if not item:
            return None
        pid = str(item.get("product_id") or item.get("id") or item.get("goods_no") or "")
        if pid:
            v2_4_path = os.path.join("outputs", "cutouts_v2_4", f"{pid}.png")
            if os.path.exists(v2_4_path):
                return v2_4_path

        legacy_path = item.get("local_image_path")
        if legacy_path and os.path.exists(legacy_path):
            return legacy_path

        if pid:
            legacy_cutout = os.path.join("outputs", "cutouts", f"{pid}.png")
            if os.path.exists(legacy_cutout):
                return legacy_cutout

        return None

    def _encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_vector(self, item: dict) -> Optional[np.ndarray]:
        """이미지 임베딩을 경로 기준으로 캐싱하여 중복 인코딩 제거"""
        img_path = self._resolve_image_path(item)
        if not img_path:
            return None
        if img_path in self._vector_cache:
            return self._vector_cache[img_path]

        try:
            vector = np.asarray(self.engine.encode_image(img_path), dtype=np.float32).ravel()
        except Exception as exc:
            print(f"⚠️ [임베딩 실패] {img_path} → {type(exc).__name__}: {exc}")
            self._vector_cache[img_path] = None
            return None

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm          # 코사인 유사도를 내적으로 계산하기 위한 정규화
        self._vector_cache[img_path] = vector
        return vector

    @staticmethod
    def _collect_colors(items: List[dict]) -> List[str]:
        colors = []
        for item in items:
            raw = item.get("colors") or item.get("color_normalized") or []
            if isinstance(raw, str):
                raw = [raw]
            colors.extend(str(color).strip().lower() for color in raw if color)
        return colors

    def calculate_vector_harmony(self, items: list) -> float:
        """Fashion-CLIP 시각 벡터 유사도 + 색상 안정성 가산점 + 아이템 완성도 보정"""
        valid_items = [it for it in items if it]
        vectors = [vec for vec in (self._get_vector(it) for it in valid_items) if vec is not None]

        if len(vectors) < 2:
            return 70.0

        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(float(np.dot(vectors[i], vectors[j])))

        avg_sim = float(np.mean(sims))
        visual_score = np.clip((avg_sim - 0.15) / (0.60 - 0.15) * 100, 40, 95)

        # 1. 무채색 안정성 보너스
        color_bonus = 0
        all_colors = self._collect_colors(valid_items)
        if any(color in NEUTRAL_COLORS for color in all_colors):
            color_bonus += 5
        if 1 <= len(set(all_colors)) <= 3:
            color_bonus += 5

        # 2. 아이템 완성도 보너스 (신발/모자 추가 시 평균 유사도 하락분 상쇄)
        item_count_bonus = 0
        if len(valid_items) == 3:      # 상의 + 하의 + 신발 (또는 모자)
            item_count_bonus = 8.0
        elif len(valid_items) >= 4:    # 풀 착장 (상의 + 하의 + 신발 + 모자)
            item_count_bonus = 12.0

        return round(float(np.clip(visual_score + color_bonus + item_count_bonus, 50.0, 99.0)), 1)

    # ------------------------------------------------------------------
    # Vision LLM 정밀 심사
    # ------------------------------------------------------------------
    def evaluate_with_vision(self, items: list, tpo_context: Optional[str] = None) -> dict:
        """gpt-4o-mini로 정밀 규칙 검증 수행 (TPO/색상/핏/격식도 4개 기준 심사)"""
        valid_items = [it for it in items if it]

        image_contents = []
        for item in valid_items:
            path = self._resolve_image_path(item)
            if path:
                b64 = self._encode_image_b64(path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}
                })

        item_list_lines = []
        for it in valid_items:
            cat = it.get("category", "")
            brand = it.get("brand_name", "")
            pname = it.get("product_name", "")
            colors = it.get("colors") or []
            # 세부 품목(subcategory)을 함께 넘긴다. VLM이 상품명만으로 계절을 추측하지 않도록
            # 'slides', 'knit_sweater'처럼 계절이 확정되는 정보를 직접 제공한다.
            subcat = it.get("subcategory") or ""
            type_label = f"{cat}/{subcat}" if subcat else cat
            item_list_lines.append(f"- [{type_label}] {brand} {pname} (색상: {colors})")
        # 줄바꿈 문자. 이스케이프 표기를 쓰지 않아 복사·붙여넣기로 깨지지 않는다.
        line_break = chr(10)
        item_list_text = line_break.join(item_list_lines)

        # TPO 미지정이어도 프롬프트는 단일 템플릿을 유지하고 값만 '미지정'으로 치환한다.
        # (None이 그대로 렌더링되어 "착용 목적(None)과"가 되는 것을 방지)
        tpo_label = tpo_context.strip() if (tpo_context and tpo_context.strip()) else "미지정"

        # 1. 시스템 프롬프트: 역할 및 객관적 태도 지침 (충돌 원인 제거)
        system_prompt = """당신은 의류 조합의 완성도를 객관적 기준에 따라 엄격하게 검증하는 전문 패션 디렉터입니다.
무비판적인 칭찬이나 단순한 감상평을 배제하고, 유저 프롬프트에 제시된 세부 심사 기준(TPO, 색상 밸런스, 실루엣, 격식도)에 철저히 근거하여 판정하세요.
반드시 지정된 JSON 스키마로만 응답하세요."""

        # 2. 유저 프롬프트: 세부 채점표 및 조화도 점수 루브릭
        user_prompt_text = f"""[코디 세트 종합 조화도 검증 요청]
제공된 이미지들을 직접 눈으로 확인하고 다음 핵심 기준에 맞춰 엄격히 판정하세요.

1. TPO 및 계절/소재 적합성 ('tpo', 'season', 'formality')
   - 착용 목적({tpo_label})과 계절감에 부합하는지 확인하세요.
   - [격식 자리인 경우에만 적용] 착용 목적이 결혼식, 장례식, 면접, 비즈니스처럼 격식을 요구하는 자리라면 모자(볼캡/버킷햇/비니)나 샌들이 포함되는 즉시 탈락('formality', 'tpo')입니다.
   - [격식 자리가 아닌 경우] 착용 목적이 여행, 데일리, 캠퍼스, 휴양지, 운동처럼 캐주얼하다면 모자와 샌들은 정상적인 코디 아이템입니다. 이때 모자나 샌들이 포함된 것만을 근거로 'formality' 위반을 판정하는 것을 금지합니다.
   - [한여름·휴양지 목적에만 적용] 울, 기모, 가죽, 두꺼운 니트, 패딩, 플리스가 포함되면 탈락('season')입니다.
   - [한겨울·방한 목적에만 적용] 샌들, 슬리퍼, 슬라이드, 뮬, 쪼리, 반팔, 반바지, 여름용 햇(버킷햇/썬햇)이 포함되면 탈락('season')입니다.
   - [간절기 목적에는 위 두 규칙을 적용하지 마세요] 착용 목적이 봄, 가을, 환절기, 간절기라면 긴팔, 얇은 니트, 린넨 혼방, 스니커즈, 볼캡, 얇은 자켓은 모두 정상 아이템입니다.
   - 착용 목적에 계절이나 월이 명시되지 않았다면 계절을 임의로 가정하지 마세요. 이 경우 'season' 위반은 상·하의 간 두께감이 서로 명백히 충돌할 때만(예: 반팔 + 기모 바지) 판정합니다.
   - 계절 판정의 1차 근거는 각 아이템 대괄호 안의 세부 품목(슬래시 뒤 값)입니다. 이미지나 상품명으로 추측하지 말고 이 값을 먼저 확인하세요.
     * 여름 전용: short_sleeve_tshirt, shorts, slides, clogs, bucket_hat
     * 겨울 전용: heavy_puffer, lightweight_puffer, fleece, beanie, knit_sweater, leather_jacket, hiking_shoes
     * 사계절(계절 위반으로 판정 금지): long_sleeve_tshirt, shirt, cotton_pants, denim_pants, jogger_pants, slacks, baseball_cap, fashion_sneakers, running_shoes, loafers, derby_shoes, canvas_shoes, hoodie, sweatshirt, blazer, cardigan
   - 세부 품목이 비어 있을 때만 상품명으로 보조 판단하세요. ('슬라이드', '뮬', '샌들'은 여름 신발 / '이어플랩', '플리스 캡'은 겨울 모자)

2. 색상 및 톤 조화 ('color') — [★ 핵심 심사 항목]
   - 눈이 피로한 극단적 원색 충돌(예: 빨강+초록 신호등룩)이나 채도가 완전히 어긋나 겉도는 조합만 탈락(pass_status: false, violated_rules: ['color'])시키고, 그 외의 자연스러운 컬러 조합은 폭넓게 허용하세요.
   - 색상 값이 3개 이상 나열된 아이템은 한 벌에 그 색이 다 들어간 것이 아니라 '구매 가능한 색상 옵션 목록'입니다. 그 아이템은 어떤 색으로든 맞출 수 있는 아이템으로 간주하고, 나열된 색상들을 근거로 색상 충돌을 판정하는 것을 금지합니다.
   - 색상 판정은 이미지에서 실제로 보이는 색을 기준으로 하세요.

3. 실루엣 및 핏 균형 ('fit')
   - 상·하의 핏 밸런스가 조화로운지 확인하세요. ([오버핏+와이드], [슬림+와이드], [레귤러+레귤러] 허용)

4. 격식도 및 무드 통일감 ('formality')
   - 아이템들의 분위기가 한 착장 안에서 심각하게 겉돌지 않는지 확인하세요.

[점수 및 최종 판정 지침]
- harmony_score 채점 기준:
  * 모든 기준을 만족하고 조화로운 경우: 80점 ~ 100점 부여
  * 사소한 아쉬움이 있으나 착용 가능한 수준인 경우: 60점 ~ 79점 부여
  * 규칙 위반으로 탈락(pass_status: false)인 경우: 절대 고득점을 주지 말고 50점 이하로 감점 처리하세요.
- 위 2번 색상 규칙 위반 시 '개성 있는 연출이라 조화롭다'는 식의 자의적 판정을 금지하며, 반드시 pass_status: false 및 violated_rules: ['color']로 처리하세요.
- 판정을 시작하기 전에 [아이템 목록]의 상품명을 하나씩 훑어 계절 신호를 확인하세요. 착용 목적의 계절과 충돌하는 아이템이 단 하나라도 있으면 다른 항목이 아무리 좋아도 반드시 pass_status: false 및 violated_rules에 'season'을 포함하세요. 같은 아이템을 어떤 조합에서는 통과시키고 다른 조합에서는 탈락시키는 비일관 판정을 금지합니다.
- [아이템 목록]에 실제로 있는 아이템만 근거로 삼으세요. 목록에 없는 카테고리(예: 목록에 모자가 없는데 모자를 언급)를 판정 사유로 쓰거나 그것을 이유로 탈락시키는 것을 금지합니다. 판정 전에 목록의 카테고리를 먼저 확인하세요.
- violated_rules에는 실제로 확인한 위반만 담으세요. 확신이 없는 규칙을 관성적으로 함께 나열하지 마세요.
- harmony_score는 조합마다 다르게 매기세요. 여러 조합에 똑같은 점수(예: 전부 85점)를 반복하지 말고, 조합 간 우열이 드러나도록 1점 단위로 차이를 두세요.
- feedback_summary에는 구체적인 판정 사유(예: "자연스러운 톤 매치로 조화로움" 또는 "채도가 어긋나 겉도는 조합")를 1문장으로 명확히 작성하세요.

[아이템 목록]
{item_list_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt_text}, *image_contents]}
        ]

        try:
            res = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=OutfitEvaluation,
                temperature=0.2
            )
            parsed = res.choices[0].message.parsed
            if parsed is None:
                raise ValueError("VLM 응답 파싱 결과가 비어 있음 (refusal 또는 스키마 불일치)")

            # 🛡️ [점수 왜곡 방지 가드레일 — 단일 지점]
            # VLM이 탈락(False)을 주고도 90점대를 주는 경우를 코드로 강제 제어
            score = float(np.clip(parsed.harmony_score, 0.0, 100.0))
            if not parsed.pass_status:
                score = min(score, FAIL_SCORE_CAP)

            return {
                "harmony_score": round(score, 1),
                "pass_status": parsed.pass_status,
                "violated_rules": parsed.violated_rules,
                "feedback_summary": parsed.feedback_summary,
                "evaluated_by": "vision_llm"
            }
        except Exception as exc:
            # 예외를 조용히 삼키면 '모든 코디 무조건 통과' 버그가 눈에 보이지 않는다.
            print(f"⚠️ [VLM 심사 실패 → 벡터 점수로 대체] {type(exc).__name__}: {exc}")
            base_score = self.calculate_vector_harmony(items)
            return {
                "harmony_score": base_score,
                "pass_status": True,
                "violated_rules": [],
                "feedback_summary": "(자동 심사 불가) 시각 벡터 유사도 기준 잠정 조화 판정",
                "evaluated_by": "vector_fallback"
            }

    # ------------------------------------------------------------------
    # 최종 선별
    # ------------------------------------------------------------------
    def select_best_outfits(self,
                            slot_candidates: dict,
                            tpo_context: Optional[str] = None,
                            total_budget: Optional[int] = None,
                            top_k: int = 2) -> list:
        """
        [계층적 코디 선별 로직]
        1단계: 코어 의류(상의 + 하의 [+ 아우터])를 우선 조합하여 뼈대 구축 및 1차 검증
        2단계: 신발·모자는 모두 선택품이므로 착용 조합별 변형을 생성
               (잡화 없음 / 신발만 / 모자만 / 둘 다) — 착용 여부 판단을 VLM에 위임
        3단계: 변형들을 Vision LLM으로 심사한 뒤, 같은 코어에서 파생된 변형은 최고점만 남김
        """
        # 0. 입력 정규화: None 센티넬 제거, hat/headwear 통합, 빈 슬롯 정리
        slots = self._normalize_slot_candidates(slot_candidates)
        if not slots:
            print("⚠️ [후보 없음] 유효한 상품 후보가 없어 코디를 구성할 수 없습니다.")
            return []

        # 1. 슬롯을 코어 의류와 잡화(신발, 모자)로 분리
        core_slots = [key for key in CORE_ORDER if slots.get(key)]
        acc_slots = [key for key in ACC_ORDER if slots.get(key)]

        # 코어 슬롯이 없으면 전체 슬롯으로 fallback
        if not core_slots:
            core_slots = list(slots.keys())
            acc_slots = []

        # 2. [1단계] 코어 의류 조합 생성 및 1차 벡터 유사도 채점
        core_combos = list(product(*[slots[slot] for slot in core_slots]))
        ranked_core = []
        for combo in core_combos:
            items = [it for it in combo if it]
            if not items:
                continue

            ranked_core.append({
                "items": items,
                "vec_score": self.calculate_vector_harmony(items),
                "total_price": sum(self._item_price(it) for it in items)
            })

        if not ranked_core:
            return []

        # 🎯 [스마트 예산 필터링 & 근접 최저가 Fallback]
        budget_note = None
        is_fallback_budget = False

        if total_budget and total_budget > 0:
            in_budget_combos = [c for c in ranked_core if c["total_price"] <= total_budget]
            if in_budget_combos:
                candidate_pool = in_budget_combos
                print(f"💰 [예산 준수] 코어 의류 {len(in_budget_combos)}개 조합이 {total_budget:,}원 이하 기준을 만족합니다.")
            else:
                is_fallback_budget = True
                ranked_core.sort(key=lambda x: (x["total_price"], -x["vec_score"]))
                min_price = ranked_core[0]["total_price"]
                candidate_pool = ranked_core[:15]
                budget_note = f"요청하신 예산({total_budget:,}원 이하)에 딱 맞는 조합이 없어, 가장 근접한 최저가({min_price:,}원~) 세트를 추천합니다."
                print(f"💡 [예산 초과 차선책] {budget_note}")
        else:
            candidate_pool = ranked_core

        # 코어 조합 상위 후보 선별
        if is_fallback_budget:
            candidate_pool.sort(key=lambda x: (x["total_price"], -x["vec_score"]))
        else:
            candidate_pool.sort(key=lambda x: x["vec_score"], reverse=True)

        top_core_candidates = candidate_pool[:min(len(candidate_pool), top_k * 3)]

        # 3. [2단계] 잡화 착용 조합별 변형 생성 (신발·모자 모두 선택품)
        eval_pool = []
        for rank, cand in enumerate(top_core_candidates):
            core_key = self._core_key(cand["items"])

            # 슬롯별로 벡터 조화도가 가장 높은 잡화 1개씩만 대표로 뽑는다.
            # (임계값으로 자르지 않는다 — 착용 여부는 아래 변형 심사에서 VLM이 결정)
            picks = []
            if rank < ACCESSORY_VARIANT_CORES:
                for acc_slot in acc_slots:
                    item, score = self._pick_best_accessory(
                        cand["items"], slots[acc_slot], cand["total_price"],
                        total_budget, is_fallback_budget
                    )
                    if item:
                        picks.append((acc_slot, item))
                        print(f"   ↳ [{acc_slot} 후보] {item.get('product_name', '')} (벡터 {score}점)")

            # 부분집합 전개: 잡화 없음 → 한 종류만 → 전부
            for size in range(len(picks) + 1):
                for subset in combinations(picks, size):
                    variant_items = list(cand["items"]) + [item for _, item in subset]
                    variant_price = cand["total_price"] + sum(
                        self._item_price(item) for _, item in subset
                    )
                    if total_budget and not is_fallback_budget and variant_price > total_budget:
                        continue

                    worn = [slot for slot, _ in subset]
                    eval_pool.append({
                        "items": variant_items,
                        "total_price": variant_price,
                        "vec_score": cand["vec_score"],
                        "core_key": core_key,
                        "worn_accessories": worn
                    })

        # VLM 호출 수 상한 적용 (상위 코어의 변형이 앞쪽에 오도록 이미 정렬되어 있음)
        if len(eval_pool) > MAX_VLM_EVALS:
            print(f"   ↳ [호출 제한] 변형 {len(eval_pool)}개 중 상위 {MAX_VLM_EVALS}개만 심사합니다.")
            eval_pool = eval_pool[:MAX_VLM_EVALS]

        # 4. [3단계] 최종 선별된 조합들에 대해 Vision LLM 정밀 심사
        print()
        print(f">> [VLM 정밀 심사 시작] 총 {len(eval_pool)}개 조합 평가 중...")
        evaluated_outfits = []
        for idx, cand in enumerate(eval_pool, 1):
            eval_res = self.evaluate_with_vision(cand["items"], tpo_context=tpo_context)

            item_types = [it.get("category", "") for it in cand["items"]]
            worn = cand.get("worn_accessories") or []
            worn_label = f"잡화: {'+'.join(worn)}" if worn else "잡화 없음"
            status_badge = "✓ 통과" if eval_res["pass_status"] else f"✗ 탈락({eval_res['violated_rules']})"
            print(f"   [{idx}번 변형] ({' + '.join(item_types)} | {worn_label} | {cand['total_price']:,}원) "
                  f"{status_badge} | {eval_res['harmony_score']}점 | {eval_res['feedback_summary']}")

            status_text = "[규칙 통과]" if eval_res["pass_status"] else "[규칙 위반]"
            verdict_text = f"{status_text} {eval_res['feedback_summary']}"

            if eval_res["violated_rules"]:
                styling_tip_text = f"위반 규칙({', '.join(eval_res['violated_rules'])})을 고려하여 아이템을 재선택해 보세요."
            else:
                styling_tip_text = "TPO와 계절감, 실루엣이 균형 있게 어우러진 추천 코디입니다."

            evaluated_outfits.append({
                "pass_status": eval_res["pass_status"],
                "violated_rules": eval_res["violated_rules"],
                "feedback_summary": eval_res["feedback_summary"],
                "verdict": verdict_text,
                "styling_tip": styling_tip_text,
                # 점수 클램프는 evaluate_with_vision에서 이미 완료 (중복 가드레일 제거)
                "harmony_score": eval_res["harmony_score"],
                "evaluated_by": eval_res["evaluated_by"],
                "total_price": cand["total_price"],
                "budget_note": budget_note,
                "items": cand["items"],
                "worn_accessories": worn,
                "core_key": cand["core_key"]
            })

        passed_outfits = [o for o in evaluated_outfits if o["pass_status"]]

        if passed_outfits:
            return self._pick_distinct_outfits(passed_outfits, top_k, is_fallback_budget)

        print("⚠️ [주의] 완벽히 통과한 코디가 없어 차선책 조합을 안내합니다.")
        return self._pick_distinct_outfits(evaluated_outfits, top_k, is_fallback_budget)

    @staticmethod
    def _pick_distinct_outfits(outfits: List[dict], top_k: int, is_fallback_budget: bool) -> List[dict]:
        """점수 순으로 고르되, 같은 코어에서 파생된 변형은 최고점 하나만 채택한다.

        (상의+하의가 동일하고 모자 유무만 다른 코디가 나란히 추천되는 것을 방지)

        점수가 동점일 때는 잡화를 더 갖춘 변형을 우선한다. 변형 생성 순서가
        '잡화 없음'부터라서, 동점이면 안정 정렬 탓에 잡화 없는 쪽이 항상 이기는
        문제가 있었다. 같은 점수라면 착장이 완성된 쪽이 더 나은 추천이다.
        """
        if is_fallback_budget:
            # 예산 차선책 모드에서는 가격이 최우선이므로 잡화 가점을 적용하지 않는다.
            outfits.sort(key=lambda x: (x["total_price"], -x["harmony_score"]))
        else:
            outfits.sort(key=lambda x: (-x["harmony_score"],
                                        -len(x.get("worn_accessories") or [])))

        chosen, used_cores = [], set()
        for outfit in outfits:
            core_key = outfit.get("core_key")
            if core_key in used_cores:
                continue
            used_cores.add(core_key)
            outfit.pop("core_key", None)   # 내부 전용 키는 응답에서 제외
            chosen.append(outfit)
            if len(chosen) >= top_k:
                break
        return chosen
