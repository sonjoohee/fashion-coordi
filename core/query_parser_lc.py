from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.llm_settings import LLM_SEED
from core.query_parser import ParsedIntent, build_system_prompt

# ==============================================================================
# 1단계 (LangChain 버전): LLM 질의 분석기
# ------------------------------------------------------------------------------
# core/query_parser.py(OpenAI SDK 직접 호출)와 같은 일을 LangChain으로 한다.
# - 프롬프트: build_system_prompt()를 그대로 써서 두 버전의 문장을 똑같이 맞춘다.
# - 출력 스키마: 같은 ParsedIntent(Pydantic)를 with_structured_output에 넘긴다.
# - 모델 설정: gpt-4o-mini, temperature=0, seed=LLM_SEED로 기존 버전과 같게 둔다.
# 이렇게 조건을 맞춰야 compare_parsers.py에서 차이가 나왔을 때
# 프레임워크 차이인지 프롬프트 차이인지 구분할 수 있다.
#
# 프롬프트를 ChatPromptTemplate으로 만들지 않고 SystemMessage에 바로 넣은 이유:
# 프롬프트 안에 중괄호가 들어가면 템플릿 변수로 해석되기 때문이다.
# 월(month)은 build_system_prompt()에서 이미 채워진 상태로 넘어온다.
# ==============================================================================

load_dotenv()


class LangChainQueryParser:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        llm = ChatOpenAI(model=model_name, temperature=0.0, seed=LLM_SEED)
        # method="json_schema": 기존 버전의 client.beta.chat.completions.parse와 같은
        # OpenAI Structured Outputs 방식을 쓴다. 응답이 스키마를 벗어나면 예외가 난다.
        self.chain = llm.with_structured_output(ParsedIntent, method="json_schema")

    def parse(self, user_query: str) -> ParsedIntent:
        current_month = datetime.now().month  # 현재 실제 월 확인
        messages = [
            SystemMessage(content=build_system_prompt(current_month)),
            HumanMessage(content=user_query),
        ]
        return self.chain.invoke(messages)
