import os
import base64
import numpy as np
from itertools import combinations, product
from typing import List, Literal, Optional, Tuple
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

# 카테고리가 다른 아이템 간 CLIP 유사도는 구조적으로 낮게 나와 절대 임계값으로는
# '어울림'과 '안 어울림'을 가를 수 없다. 그래서 모자 착용 여부를 벡터 점수로 자르지 않고,
# 착용 조합별 변형(variant)을 만들어 명시적 규칙을 가진 VLM이 판단하게 한다.
ACCESSORY_VARIANT_CORES = 2   # 모자 착용 변형을 생성할 상위 코어 조합 수
MAX_VLM_EVALS = 10            # VLM 호출 수 상한 (비용 통제)

CORE_ORDER = ("outer", "top", "bottom")     # 코어 의류 (조합 뼈대)
ACC_ORDER = ("shoes", "headwear")           # 코어 뒤에 붙는 슬롯

# 신발은 필수 착장이다. 후보가 있으면 모든 변형에 반드시 포함한다.
# (코어에 넣지 않는 이유: 코어에 넣으면 코어 식별키에 신발이 섞여, 최종 추천 2개가
#  옷은 똑같고 신발만 다른 세트로 채워질 수 있다. 옷 조합의 다양성을 지키려면
#  신발은 코어 뒤에 붙이되 선택권만 없애는 쪽이 맞다.)
MANDATORY_ACC_SLOTS = ("shoes",)
# 모자는 선택품이다. 야외 상황에서만 후보로 올라오고(coordinator가 판단),
# 룩에 어울리는지는 착용/미착용 변형을 비교해 VLM이 결정한다.
SLOT_ALIASES = {"hat": "headwear"}          # 동일 부위 다른 명칭 통합
ITEM_ID_KEYS = ("product_id", "id", "goods_no")

class OutfitEvaluation(BaseModel):
    """Vision LLM 심사 결과 스키마.

    VLM에게 '통과/탈락'이라는 종합 판정을 맡기지 않는다. 항목별 점수와 충돌 여부만
    받고, 종합 점수와 pass_status는 코드가 계산한다. VLM이 직접 harmony_score를
    매기게 했을 때 85점과 45점 두 값으로만 몰려 정렬이 무의미해졌기 때문이다.

    주의: Structured Outputs strict 모드는 숫자 범위 제약(ge/le)을 지원하지 않으므로
    범위는 프롬프트로 유도하고 클램프는 코드에서 수행한다.
    """
    color_score: float = Field(
        description="색상·톤 조화 점수 (0~100). 무채색 기반이나 톤이 이어지면 높고, 원색이 충돌하면 낮음"
    )
    fit_score: float = Field(
        description="실루엣·핏 균형 점수 (0~100). 상하의 볼륨 대비가 자연스러우면 높음"
    )
    mood_score: float = Field(
        description="무드 통일감 및 착용 목적 부합 점수 (0~100). 한 착장으로 읽히면 높음"
    )
    season_conflict: bool = Field(
        description="상·하의 두께감이 서로 명백히 충돌하는지 (예: 반팔 + 기모 바지). 개별 아이템의 계절 적합성은 판정 대상이 아님"
    )
    formality_conflict: bool = Field(
        description="유저 프롬프트의 격식 지침에 정확히 해당할 때만 true. "
                    "프롬프트가 '격식 수준은 심사 항목이 아니다'라고 안내한 경우에는 항상 false. "
                    "주관적 인상으로는 절대 true로 하지 말 것"
    )
    feedback_summary: str = Field(
        description="판정 근거를 한 문장으로 요약. 근거가 된 아이템의 상품명을 반드시 포함"
    )


# --- 종합 점수 산출 규칙 -------------------------------------------------------
# 색상을 가장 크게 본다. 실제 코디 품질에서 톤 충돌이 가장 눈에 띄기 때문이다.
SCORE_WEIGHTS = {"color": 0.5, "fit": 0.25, "mood": 0.25}
PASS_SCORE_THRESHOLD = 60.0   # 이 점수 미만이면 착용 가능 수준이 아니라고 본다
LOW_ITEM_SCORE = 40.0         # 항목 점수가 이 값 미만이면 해당 규칙 위반으로 기록

# 최종 조화도는 VLM 평가에 시각 벡터 유사도를 소량 섞어 산출한다.
# gpt-4o-mini가 항목 점수를 5의 배수로만 매기고 서로 다른 조합에 같은 세 값을
# 반복해서(예: 색85/핏80/무드90이 10변형 연속) 종합 점수가 동점으로 뭉쳤다.
# 벡터 유사도는 조합마다 다른 연속값이라 그 뭉침을 풀어준다.
VLM_WEIGHT = 0.85
VECTOR_WEIGHT = 0.15

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

        비교에는 상한이 없는 연속 점수(mix)를 쓴다. 표시용 점수로 비교하면 후보 여러 개가
        99.0에 함께 닿아 사실상 첫 번째 후보가 그냥 뽑히는 일이 생긴다.
        돌려주는 값은 로그 가독성을 위해 표시용 점수다.
        """
        best_item, best_display, best_mix = None, 70.0, -1e9
        for candidate in candidates:
            price = self._item_price(candidate)
            if total_budget and not is_fallback_budget and (current_price + price > total_budget):
                continue
            display_score, mix_score = self.vector_harmony_parts(base_items + [candidate])
            if mix_score > best_mix:
                best_mix, best_display, best_item = mix_score, display_score, candidate
        return best_item, best_display

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

    def vector_harmony_parts(self, items: list) -> Tuple[float, float]:
        """벡터 조화도를 (표시용 점수, 정렬·혼합용 연속 점수) 두 가지로 돌려준다.

        표시용 점수는 50~99 범위로 잘라 사람이 읽기 쉽게 만든 값이다.
        그런데 보너스(무채색 +10, 아이템 수 +8~12)가 더해지면 상한 99에 여러 조합이
        동시에 닿아버려 순위를 전혀 가리지 못한다. 실제로 겨울 질의에서 후보 8개의
        벡터 점수가 모두 99.0으로 붙었고, 그 결과 최종 조화도까지 83.9점으로 똑같아졌다.

        그래서 상·하한을 적용하지 않은 연속 점수를 따로 계산해 정렬과 점수 혼합에 쓴다.
        보너스 체계와 척도는 그대로 두었으므로 점수 수준은 기존과 거의 같고,
        상한에 눌려 사라졌던 조합 간 미세한 차이만 되살아난다.
        """
        valid_items = [it for it in items if it]
        vectors = [vec for vec in (self._get_vector(it) for it in valid_items) if vec is not None]

        if len(vectors) < 2:
            return 70.0, 70.0

        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(float(np.dot(vectors[i], vectors[j])))

        avg_sim = float(np.mean(sims))
        visual_raw = (avg_sim - 0.15) / (0.60 - 0.15) * 100
        visual_score = np.clip(visual_raw, 40, 95)

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

        display_score = round(float(np.clip(visual_score + color_bonus + item_count_bonus, 50.0, 99.0)), 1)
        mix_score = round(float(visual_raw + color_bonus + item_count_bonus), 2)
        return display_score, mix_score

    def calculate_vector_harmony(self, items: list) -> float:
        """표시용 벡터 조화도(50~99)만 필요한 호출부를 위한 래퍼"""
        return self.vector_harmony_parts(items)[0]

    # ------------------------------------------------------------------
    # Vision LLM 정밀 심사
    # ------------------------------------------------------------------
    def evaluate_with_vision(self, items: list, tpo_context: Optional[str] = None,
                             is_formal: bool = False) -> dict:
        """gpt-4o-mini로 정밀 규칙 검증 수행 (색상/핏/무드 채점 + 충돌 여부 표시)

        is_formal이 False면 격식 심사 지침을 프롬프트에서 아예 빼 버린다.
        '캐주얼하면 모자를 격식 위반으로 보지 말라'고 금지 문구를 넣어 두었어도
        gpt-4o-mini는 여행 질의에 "모자가 있어 격식 있는 자리에 부적합하다"는 판정을
        계속 내놓았다. 지키라고 말하는 것보다 판단 거리를 안 주는 쪽이 확실하다.
        """
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

        # 격식 심사 지침은 격식 자리일 때만 보낸다. 캐주얼 질의에는 '격식'이라는 단어
        # 자체를 노출하지 않아야 엉뚱한 격식 판정이 섞여 들어오지 않는다.
        if is_formal:
            formality_axis = ", 격식도"
            formality_rule = ("   - 착용 목적이 격식을 요구하는 자리입니다. 모자(볼캡/버킷햇/비니)나 "
                              "샌들/슬리퍼/슬라이드가 [아이템 목록]에 실제로 있으면 formality_conflict를 true로 하세요." + chr(10) +
                              "   - 해당 품목이 목록에 없으면 formality_conflict는 무조건 false입니다. "
                              "'격식이 부족해 보인다', '신발이 캐주얼하다' 같은 주관적 인상으로 true로 만드는 것을 금지합니다. "
                              "아쉬움은 mood_score를 낮추는 방식으로만 표현하세요." + chr(10) +
                              "   - 셔츠, 슬랙스, 로퍼, 더비슈즈, 옥스포드화는 격식 자리에 적합한 아이템입니다. "
                              "이들만으로 구성된 착장은 formality_conflict가 false입니다.")
        else:
            formality_axis = ""
            formality_rule = ("   - 이 착장은 캐주얼한 상황용입니다. 모자, 샌들, 슬라이드, 스니커즈는 모두 정상적인 "
                              "코디 아이템입니다." + chr(10) +
                              "   - 격식 수준은 이번 심사 항목이 아닙니다. formality_conflict는 false로 두고, "
                              "격식이나 예식장 적합성을 판정 사유나 feedback_summary에 쓰지 마세요.")

        # 1. 시스템 프롬프트: 역할 및 객관적 태도 지침 (충돌 원인 제거)
        system_prompt = f"""당신은 의류 조합의 완성도를 객관적 기준에 따라 엄격하게 검증하는 전문 패션 디렉터입니다.
무비판적인 칭찬이나 단순한 감상평을 배제하고, 유저 프롬프트에 제시된 세부 심사 기준(TPO, 색상 밸런스, 실루엣{formality_axis})에 철저히 근거하여 판정하세요.
반드시 지정된 JSON 스키마로만 응답하세요."""

        # 2. 유저 프롬프트: 세부 채점표 및 조화도 점수 루브릭
        user_prompt_text = f"""[코디 세트 종합 조화도 검증 요청]
제공된 이미지들을 직접 눈으로 확인하고 다음 핵심 기준에 맞춰 엄격히 판정하세요.

1. TPO 및 계절/소재 적합성 ('tpo', 'season')
   - 착용 목적({tpo_label})과 계절감에 부합하는지 확인하세요.
{formality_rule}
   - ★ 개별 아이템의 계절 적합성은 이미 검색 단계에서 보장되었습니다. 목록에 있는 아이템은 모두 이 계절에 착용 가능한 것으로 확정된 상태입니다. "이 아이템은 여름용/겨울용이라 부적합하다"는 식의 판정을 하지 마세요.
   - season_conflict는 상·하의의 두께감이 서로 명백히 충돌할 때만 true입니다. (예: 반팔 티셔츠 + 기모 바지) 그 외에는 false로 두세요.

2. 색상 및 톤 조화 ('color') — [★ 핵심 심사 항목]
   - 눈이 피로한 극단적 원색 충돌(예: 빨강+초록 신호등룩)이나 채도가 완전히 어긋나 겉도는 조합만 탈락(pass_status: false, violated_rules: ['color'])시키고, 그 외의 자연스러운 컬러 조합은 폭넓게 허용하세요.
   - 색상 값이 3개 이상 나열된 아이템은 한 벌에 그 색이 다 들어간 것이 아니라 '구매 가능한 색상 옵션 목록'입니다. 그 아이템은 어떤 색으로든 맞출 수 있는 아이템으로 간주하고, 나열된 색상들을 근거로 색상 충돌을 판정하는 것을 금지합니다.
   - 색상 판정은 이미지에서 실제로 보이는 색을 기준으로 하세요.

3. 실루엣 및 핏 균형 ('fit')
   - 상·하의 핏 밸런스가 조화로운지 확인하세요. ([오버핏+와이드], [슬림+와이드], [레귤러+레귤러] 허용)

4. 무드 통일감
   - 아이템들의 분위기가 한 착장 안에서 심각하게 겉돌지 않는지는 mood_score로 반영하세요.
   - 모자가 목록에 있다면, 그 모자가 착장에 보탬이 되는지를 mood_score에 반영하세요. 어색하다고 판단하면 그 판단을 글로만 쓰지 말고 반드시 mood_score를 낮추세요.

[채점 지침]
당신은 통과/탈락을 판정하지 않습니다. 아래 세 항목에 각각 0~100점을 매기고 충돌 여부만 표시하세요.
종합 점수와 최종 통과 여부는 시스템이 계산합니다.

- color_score / fit_score / mood_score 채점 기준:
  * 90~100 : 흠잡을 데 없이 잘 어우러짐
  * 70~89  : 자연스럽고 무난함
  * 50~69  : 아쉬운 점이 있으나 착용 가능
  * 30~49  : 눈에 거슬림
  * 0~29   : 심각하게 어긋남
- 세 점수를 모두 같은 값으로 매기지 마세요. 항목마다 실제 평가가 다르므로 점수도 달라야 합니다.
- 조합마다 다른 점수를 매기세요. 여러 조합에 똑같은 숫자(예: 전부 85점)를 반복하면 순위를 가릴 수 없습니다. 1점 단위로 차이를 두세요.
- [아이템 목록]에 실제로 있는 아이템만 근거로 삼으세요. 목록에 없는 카테고리(예: 목록에 모자가 없는데 모자를 언급)를 판정 사유로 쓰는 것을 금지합니다.
- feedback_summary에는 점수의 근거를 1문장으로 쓰고, 근거가 된 아이템의 상품명을 반드시 포함하세요. (예: "'피치드 레글런 롱슬리브'의 오트밀 톤이 데님과 부드럽게 이어짐")

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

            # 항목 점수는 VLM이, 종합 판정은 코드가 한다.
            # (VLM에게 harmony_score와 pass_status를 직접 맡겼을 때
            #  85점/45점 두 값으로만 몰려 정렬이 무의미해졌다)
            color = float(np.clip(parsed.color_score, 0.0, 100.0))
            fit = float(np.clip(parsed.fit_score, 0.0, 100.0))
            mood = float(np.clip(parsed.mood_score, 0.0, 100.0))

            score = (color * SCORE_WEIGHTS["color"]
                     + fit * SCORE_WEIGHTS["fit"]
                     + mood * SCORE_WEIGHTS["mood"])

            # 위반 목록과 통과 여부가 모순되지 않게, 위반이 하나라도 있으면 탈락시킨다.
            # (통과인데 '위반 규칙을 고려하세요'라는 안내가 붙는 상황을 막는다)
            violated_rules = []
            if parsed.season_conflict:
                violated_rules.append("season")
            if parsed.formality_conflict:
                violated_rules.append("formality")
            if color < LOW_ITEM_SCORE:
                violated_rules.append("color")
            if fit < LOW_ITEM_SCORE:
                violated_rules.append("fit")

            pass_status = (not violated_rules) and score >= PASS_SCORE_THRESHOLD
            if not pass_status:
                score = min(score, FAIL_SCORE_CAP)

            return {
                "harmony_score": round(score, 1),
                "pass_status": pass_status,
                "violated_rules": violated_rules,
                "feedback_summary": parsed.feedback_summary,
                "item_scores": {"color": round(color, 1), "fit": round(fit, 1), "mood": round(mood, 1)},
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
                "item_scores": None,
                "evaluated_by": "vector_fallback"
            }

    # ------------------------------------------------------------------
    # 최종 선별
    # ------------------------------------------------------------------
    def select_best_outfits(self,
                            slot_candidates: dict,
                            tpo_context: Optional[str] = None,
                            total_budget: Optional[int] = None,
                            top_k: int = 2,
                            is_formal: bool = False) -> list:
        """
        [계층적 코디 선별 로직]
        1단계: 코어 의류(상의 + 하의 [+ 아우터])를 우선 조합하여 뼈대 구축 및 1차 검증
        2단계: 신발은 필수이므로 모든 변형에 고정 포함하고, 모자만 착용/미착용 변형을
               만들어 어울리는지 판단을 VLM에 위임
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

            display_score, mix_score = self.vector_harmony_parts(items)
            ranked_core.append({
                "items": items,
                "vec_score": display_score,
                "vec_mix": mix_score,
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
                ranked_core.sort(key=lambda x: (x["total_price"], -x["vec_mix"]))
                min_price = ranked_core[0]["total_price"]
                candidate_pool = ranked_core[:15]
                budget_note = f"요청하신 예산({total_budget:,}원 이하)에 딱 맞는 조합이 없어, 가장 근접한 최저가({min_price:,}원~) 세트를 추천합니다."
                print(f"💡 [예산 초과 차선책] {budget_note}")
        else:
            candidate_pool = ranked_core

        # 코어 조합 상위 후보 선별
        # 정렬 기준은 상한이 없는 vec_mix다. vec_score로 정렬하면 상위 코어가 모두 99.0에
        # 묶여 사실상 조합 생성 순서대로 뽑히게 되고, 좋은 코어가 상위 후보에서 밀려난다.
        if is_fallback_budget:
            candidate_pool.sort(key=lambda x: (x["total_price"], -x["vec_mix"]))
        else:
            candidate_pool.sort(key=lambda x: x["vec_mix"], reverse=True)

        top_core_candidates = candidate_pool[:min(len(candidate_pool), top_k * 3)]

        # 3. [2단계] 변형 생성 — 신발은 전 변형에 고정, 모자만 착용/미착용으로 전개
        mandatory_slots = [s for s in acc_slots if s in MANDATORY_ACC_SLOTS]
        optional_slots = [s for s in acc_slots if s not in MANDATORY_ACC_SLOTS]

        eval_pool = []
        for rank, cand in enumerate(top_core_candidates):
            core_key = self._core_key(cand["items"])

            # 슬롯별로 벡터 조화도가 가장 높은 후보 1개씩만 대표로 뽑는다.
            # (임계값으로 자르지 않는다 — 모자 착용 여부는 아래 변형 심사에서 VLM이 결정)

            # 신발: 필수이므로 모든 코어에 대해 뽑고 고정으로 붙인다.
            fixed = []
            fixed_price = 0
            for acc_slot in mandatory_slots:
                item, score = self._pick_best_accessory(
                    cand["items"], slots[acc_slot], cand["total_price"] + fixed_price,
                    total_budget, is_fallback_budget
                )
                if item:
                    fixed.append((acc_slot, item))
                    fixed_price += self._item_price(item)
                    print(f"   ↳ [{acc_slot} 필수] {item.get('product_name', '')} (벡터 {score}점)")
                else:
                    # 후보가 없거나 예산에 걸리면 신발 없이 진행한다 (없으면 어쩔 수 없다).
                    print(f"   ↳ [{acc_slot} 필수] 조건에 맞는 후보가 없어 신발 없이 구성합니다.")

            # 모자: 선택품이라 상위 코어에서만 후보를 뽑아 착용/미착용을 비교한다.
            picks = []
            if rank < ACCESSORY_VARIANT_CORES:
                for acc_slot in optional_slots:
                    item, score = self._pick_best_accessory(
                        cand["items"], slots[acc_slot], cand["total_price"] + fixed_price,
                        total_budget, is_fallback_budget
                    )
                    if item:
                        picks.append((acc_slot, item))
                        print(f"   ↳ [{acc_slot} 선택] {item.get('product_name', '')} (벡터 {score}점)")

            # 부분집합 전개: 모자 없음 → 모자 착용
            for size in range(len(picks) + 1):
                for subset in combinations(picks, size):
                    worn_pairs = fixed + list(subset)
                    variant_items = list(cand["items"]) + [item for _, item in worn_pairs]
                    variant_price = cand["total_price"] + sum(
                        self._item_price(item) for _, item in worn_pairs
                    )
                    if total_budget and not is_fallback_budget and variant_price > total_budget:
                        continue

                    worn = [slot for slot, _ in worn_pairs]
                    eval_pool.append({
                        "items": variant_items,
                        "total_price": variant_price,
                        "vec_score": cand["vec_score"],
                        "vec_mix": cand["vec_mix"],
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
            eval_res = self.evaluate_with_vision(cand["items"], tpo_context=tpo_context,
                                                 is_formal=is_formal)

            # VLM 점수에 벡터 조화도를 섞어 동점을 해소한다.
            # 탈락 세트는 상한(FAIL_SCORE_CAP)을 유지해야 하므로 통과 세트만 혼합한다.
            # 혼합에는 표시용(50~99)이 아니라 상한 없는 vec_mix를 쓴다. 표시용 점수는
            # 상한에 눌려 후보 전체가 99.0으로 같아지는 일이 잦아 동점을 못 갈랐다.
            vlm_score = eval_res["harmony_score"]
            if eval_res["pass_status"]:
                blended = VLM_WEIGHT * vlm_score + VECTOR_WEIGHT * cand["vec_mix"]
                # 통과인데 표시 점수가 통과 기준 아래로 내려가는 모순을 막고,
                # vec_mix가 100을 넘을 수 있으므로 최종 점수는 100으로 묶는다.
                final_score = round(min(max(blended, PASS_SCORE_THRESHOLD), 100.0), 1)
            else:
                final_score = vlm_score

            item_types = [it.get("category", "") for it in cand["items"]]
            worn = cand.get("worn_accessories") or []
            worn_label = f"잡화: {'+'.join(worn)}" if worn else "잡화 없음"
            status_badge = "✓ 통과" if eval_res["pass_status"] else f"✗ 탈락({eval_res['violated_rules']})"
            scores = eval_res.get("item_scores")
            score_detail = (f" (색{scores['color']}/핏{scores['fit']}/무드{scores['mood']})"
                            if scores else "")
            print(f"   [{idx}번 변형] ({' + '.join(item_types)} | {worn_label} | {cand['total_price']:,}원) "
                  f"{status_badge} | {final_score}점{score_detail} "
                  f"[VLM {vlm_score} / 벡터 {cand['vec_score']} (혼합값 {cand['vec_mix']})] | {eval_res['feedback_summary']}")

            status_text = "[규칙 통과]" if eval_res["pass_status"] else "[규칙 위반]"
            verdict_text = f"{status_text} {eval_res['feedback_summary']}"

            # 팁 분기는 pass_status를 기준으로 한다. 점수 미달로 탈락했으나 특정 위반 항목이
            # 없는 경우(violated_rules가 빈 배열)에 칭찬 문구가 붙는 것을 막는다.
            if eval_res["pass_status"]:
                styling_tip_text = "TPO와 계절감, 실루엣이 균형 있게 어우러진 추천 코디입니다."
            elif eval_res["violated_rules"]:
                styling_tip_text = f"위반 규칙({', '.join(eval_res['violated_rules'])})을 고려하여 아이템을 재선택해 보세요."
            else:
                styling_tip_text = "종합 조화도가 기준에 미달했습니다. 색상 톤이나 실루엣을 조정해 보세요."

            evaluated_outfits.append({
                "pass_status": eval_res["pass_status"],
                "violated_rules": eval_res["violated_rules"],
                "feedback_summary": eval_res["feedback_summary"],
                "verdict": verdict_text,
                "styling_tip": styling_tip_text,
                # VLM 점수와 벡터 점수를 섞은 최종값. 원본 두 값도 함께 남겨 추적 가능하게 한다.
                "harmony_score": final_score,
                "vlm_score": vlm_score,
                "item_scores": eval_res.get("item_scores"),
                "evaluated_by": eval_res["evaluated_by"],
                "total_price": cand["total_price"],
                "budget_note": budget_note,
                "items": cand["items"],
                "worn_accessories": worn,
                "core_key": cand["core_key"],
                "vec_score": cand["vec_score"],
                "vec_mix": cand["vec_mix"]
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

        동점 처리 순서는 (1) 잡화를 더 갖춘 변형, (2) 벡터 조화도가 높은 변형이다.
        변형 생성 순서가 '잡화 없음'부터라서 동점이면 안정 정렬 탓에 잡화 없는 쪽이
        항상 이기는 문제가 있었고, VLM이 점수를 5의 배수로만 매겨 서로 다른 코어까지
        동점이 되는 경우가 많다. 벡터 조화도는 연속값이라 그 동점을 갈라준다.
        """
        if is_fallback_budget:
            # 예산 차선책 모드에서는 가격이 최우선이므로 잡화 가점을 적용하지 않는다.
            outfits.sort(key=lambda x: (x["total_price"], -x["harmony_score"]))
        else:
            outfits.sort(key=lambda x: (-x["harmony_score"],
                                        -len(x.get("worn_accessories") or []),
                                        -x.get("vec_mix", 0.0)))

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
