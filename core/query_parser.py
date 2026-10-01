import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
from core.llm_settings import LLM_SEED

# ==============================================================================
# 1단계: LLM 질의 분석기 & 가드레일 (의도 분류 및 슬롯 추출)
# ==============================================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 단품 슬롯 스키마 정의
class SlotQuery(BaseModel):
    slot_name: str = Field(description="슬롯명: outer, top, bottom, shoes, headwear 중 하나")
    category: Literal["outer", "top", "bottom", "shoes", "headwear"] = Field(
        description="무신사 카테고리 (outer, top, bottom, shoes, headwear)"
    )
    clip_query_en: str = Field(description="Fashion-CLIP 검색용 패션 영문 시각 묘사 (소재, 핏, 실루엣, 디테일)")
    color: Optional[str] = Field(None, description="색상 제약조건 (예: black, white, brown, blue, gray)")
    max_price: Optional[int] = Field(None, description="단품 개별 가격 상한선 (원 단위 정수, 예: '8만원 이하 티셔츠' -> 80000)")

# 2. 전체 의도 스키마 정의
class ParsedIntent(BaseModel):
    search_type: Literal["single", "coordination", "unrelated"] = Field(
        description="검색 유형: single(단품), coordination(세트 코디), unrelated(패션 무관 질문)"
    )
    tpo_summary: str = Field(
        description="TPO, 계절, 날씨, 무드 요약"
    )
    # 🎯 계절 추론을 LLM에게 완전히 위임!
    season: Optional[Literal["SS", "FW", "ALL"]] = Field(
        None,
        description="질의의 시기, 월(month), 기온, 여행지 기후를 종합 판단한 계절 (여름/봄/더위/열대여행지: SS, 가을/겨울/추위/한파: FW, 무관/사계절: ALL 또는 null)"
    )
    total_budget: Optional[int] = Field(None, description="코디 세트 전체 총 예산")
    # 🎯 모자 추천 여부를 가르는 판단. 모자는 야외 상황에서만 후보로 올린다.
    is_outdoor: bool = Field(
        False,
        description="주된 활동 장소가 야외인지 여부 (여행, 등산, 캠핑, 바다, 해변, 산책, 러닝, 축제, 야외 행사, 날씨에 맞춘 외출 질의: true / 실내 위주 상황(결혼식, 면접, 사무실, 집, 카페, 식당): false)"
    )
    slots: List[SlotQuery] = Field(default_factory=list, description="검색할 슬롯 리스트")

class QueryParser:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def parse(self, user_query: str) -> ParsedIntent:
        from datetime import datetime
        current_month = datetime.now().month  # 현재 실제 월 확인

        system_prompt = f"""
        당신은 패션 커머스 플랫폼의 수석 AI 스타일리스트이자 쿼리 분석기입니다.
        (참고: 현재 실제 날짜는 {current_month}월입니다. '오늘' 또는 '요즘' 질의 시 이를 반영하세요.)

        사용자의 한국어 질의를 분석하여 [패션 무관 질의], [단품 검색], [코디 세트 추천] 중 하나로 엄격하게 분류하세요.

        [핵심 분류 및 예산 할당 규칙]:

        1. single (단품 검색 - 최우선 규칙):
           - 특정 단일 품목(티셔츠, 셔츠, 니트, 맨투맨, 후드, 아우터, 자켓, 패딩, 바지, 슬랙스, 청바지, 신발, 모자 등) 1개만 언급된 경우
           - TPO 수식어나 가격 조건이 붙어 있어도 품목이 1개면 무조건 **"single"**입니다!
             * "8만원 이하 티셔츠 추천해줘" -> search_type: "single", total_budget: null, slot.max_price: 80000
             * "4만원 이하 맨투맨 추천해줘" -> search_type: "single", total_budget: null, slot.max_price: 40000
             * "5만원대 신발 찾아줘" -> search_type: "single", total_budget: null, slot.max_price: 50000
             * "여름 휴양지에서 편하게 신을 샌들" -> search_type: "single", total_budget: null, slot.max_price: null
           - slots 리스트에는 반드시 해당 1개의 품목 슬롯만 담으세요.
           - ★ 단품 검색일 때는 total_budget을 절대로 입력하지 말고 반드시 null(None)로 두세요!

        2. coordination (코디 세트 추천):
           - '룩', '코디', '세트', '셋업', '스타일링'을 요청하거나 상·하의 등 2개 이상 품목의 조합을 원할 때
           - ★ 특정 단일 품목 없이 **"8월에 입을 옷"**, **"오늘 날씨에 적합한 옷"**, **"옷 추천해줘"**, **"입을 것 추천해줘"**처럼 계절/날씨/상황에 어울리는 포괄적 의류를 요청한 경우에도 무조건 **"coordination"**으로 분류하세요!
             * "8월에 입고 다닐 옷 추천해줘" -> search_type: "coordination", tpo_summary: "8월 무더위 한여름 캐주얼 데일리룩", slots: [top, bottom]
             * "오늘 날씨에 적합한 옷 추천해줘" -> search_type: "coordination", tpo_summary: "현재 계절감에 어울리는 편안한 데일리룩", slots: [top, bottom]
             * "8만원 이하로 여름 코디 맞춰줘" -> search_type: "coordination", total_budget: 80000
             * "결혼식 하객룩 세트 추천" -> search_type: "coordination", total_budget: null
           - ★ [여행지 시기 미지정 규칙]:
             * "캐나다 갈 때 옷", "유럽 여행 룩"처럼 시기(월/계절)가 빠진 여행 질의의 경우, 임의로 한겨울 패딩을 잡지 말고 현재 실제 월({current_month}월)의 현지 기후를 고려하거나, 간절기 레이어드 룩(자켓/셔츠 등)을 추천하세요. (단, 동남아/나트랑 등 상열대 지역은 항상 SS 적용)
           - slots 리스트에는 최소 2개 이상(top, bottom 등)을 구성하세요.
           - 전체 착장 합산 금액 조건이 명시되었을 때만 total_budget에 정수를 입력하세요.

        3. unrelated (패션 무관):
           - "오늘 날씨 어때?", "비 와?", "점심 뭐 먹지?", "파이썬 코드 짜줘", "주식 추천"처럼 **패션/의류/착장/스타일링과 전혀 관련 없는 질문**만 해당합니다.
           - ★ 주의: "오늘 날씨에 입을 옷", "8월에 입을 옷", "비 올 때 입을 옷"처럼 날씨나 시기에 맞는 **'옷/착장'**을 묻는 질문은 100% 패션 질의이므로 절대 unrelated로 분류하지 마세요!
           - search_type: "unrelated", tpo_summary: "해당 없음", total_budget: null, slots: []

        4. color 추출 규칙:
           - "블랙과 화이트", "모노톤"처럼 복합 색상이 언급된 경우 color 필드는 null로 두고 clip_query_en에 묘사를 넣으세요.
           - 단일 색상일 때만 color에 단일 영문명을 지정하세요.

        5. category 매핑: 반드시 ["outer", "top", "bottom", "shoes", "headwear"] 중 하나로 지정하세요.

        5-1. is_outdoor 판단 규칙 (장소만 보고 사실 판단하세요. 옷이 어울리는지는 보지 마세요):
           - true: 여행/출국, 등산, 캠핑, 바다/해변/휴양지, 산책, 러닝/조깅, 자전거, 피크닉, 축제/페스티벌,
             놀이공원, 골프, 낚시, 스키/보드, 야외 행사, 그리고 "오늘 날씨에 맞는 옷"처럼 날씨를 근거로
             외출 복장을 묻는 질의
             * "12월 캐나다 갈 건데 옷 추천좀" -> true (해외 여행은 이동·관광으로 야외 시간이 길다)
             * "오늘 날씨에 적합한 옷 추천해줘" -> true (날씨를 묻는 것은 외출을 전제한다)
             * "8월에 입고 다닐 옷 추천해줘" -> true (일상 외출)
           - false: 결혼식/장례식 등 실내 예식, 면접, 사무실 출근, 소개팅, 집/홈웨어, 카페, 식당, 실내 데이트
             * "친한 친구 결혼식 깔끔한 하객룩 추천해줘" -> false (예식장 실내)
           - 단품 검색(single)이나 패션 무관(unrelated) 질의에서는 false로 두세요.

        6. clip_query_en (한국어 패션 어휘 영문 정밀 매핑 - ★ 중요):
           - Fashion-CLIP 시각 인코더가 혼동하지 않도록 품목별 핵심 시각 특징과 배제 조건을 명확히 기재하세요.
           * **맨투맨/스웨트셔츠**: "crewneck sweatshirt, ribbed collar cuffs and hem, heavy cotton terry pullover, no hood, not a thin t-shirt"
           * **후드티**: "hooded sweatshirt, hoodie, drawstring hood, front kangaroo pocket"
           * **긴팔티/롱슬리브**: "long sleeve t-shirt, thin single jersey cotton, lightweight basic top"
           * **반팔티**: "short sleeve t-shirt, crewneck tee, lightweight cotton"
           * **니트/스웨터**: "knitted sweater, wool knit fabric, rib knit pullover"
           * **셔츠**: "button-up collared dress shirt, cotton oxford shirt"
           * **슬랙스**: "formal tailored dress slacks, straight fit trousers, pressed crease"
           * **청바지/데님**: "denim jeans, washed denim pants, five pocket pants"
           * **기타 품목**: 핏(오버핏, 와이드핏 등)과 원단 소재 디테일을 구체적 영문 형용사로 작성하세요.
        """

        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format=ParsedIntent,
            temperature=0.0,
            # 같은 질의가 같은 검색어를 만들어야 추천 결과를 비교할 수 있다.
            seed=LLM_SEED
        )
        return completion.choices[0].message.parsed

        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format=ParsedIntent,
            temperature=0.0
        )
        return completion.choices[0].message.parsed