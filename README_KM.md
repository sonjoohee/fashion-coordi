# FITzza: Qdrant 상품 데이터 인덱싱 가이드 (`index_to_qdrant.py`)

무신사 크롤링 상품 정제 데이터(`JSONL`)와 로컬 이미지(누끼/정면컷/썸네일)를 기반으로 **Fashion-CLIP 512차원 시각 임베딩 벡터**를 추출하고, **Qdrant Vector DB**에 배치 적재(Upsert)하는 스크립트입니다.

---

## 1. 사전 필수 요구사항 (Prerequisites)

- **Python:** 3.10 이상 권장
- **Docker / Docker Desktop:** Qdrant 서버 컨테이너 구동용
- **하드웨어 환경:** CPU 또는 CUDA 지원 GPU (자동 감지)

---

## 2. 가상환경 구성 및 패키지 설치

```bash

# 1. 필수 라이브러리 설치
pip install torch torchvision
pip install transformers pillow tqdm numpy qdrant-client

# 2. Qdrant Docker 컨테이너 실행
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant

# Qdrant 웹 대시보드 확인: 브라우저에서 http://localhost:6333/dashboard 접속

# 상품 임베딩 실행
python index_to_qdrant.py

# 테스트 코드 실행
python test_search.py

## 📁 프로젝트 디렉터리 구조 (Directory Structure)

```text
fitzza_poc/
├── core/
│   ├── harmonizer.py          # 1차 벡터 조화도 스크리닝 및 2차 Vision LLM 검증
│   ├── query_parser.py        # 사용자 자연어 질의 TPO 의도 분석 및 슬롯 파싱
│   ├── response_generator.py  # 추천 코디 최종 사용자 응답 및 스타일링 팁 생성
│   └── search_engine.py       # Fashion-CLIP 임베딩 및 Qdrant HNSW 코사인 검색 엔진
├── .env                       # OpenAI API Key 및 환경 변수 설정 파일
├── coordinator.py             # 전체 추천 파이프라인 오케스트레이션 메인 모듈
├── index_to_qdrant.py         # 500개 상품 이미지 Fashion-CLIP 임베딩 및 Qdrant 적재
├── test_search.py             # 단품 및 조합 검색 기능/지연 시간 검증 단위 테스트
└── README.md                  # 프로젝트 안내 및 가이드 문서

---

## 질의 분석기 LangChain 버전 (`core/query_parser_lc.py`)

1단계 질의 분석기를 LangChain(`langchain-openai`)으로 옮긴 버전입니다. 기존 OpenAI SDK 버전(`core/query_parser.py`)은 그대로 두고, 환경 변수로 골라 씁니다.

| | OpenAI SDK 버전 | LangChain 버전 |
| --- | --- | --- |
| 파일 | `core/query_parser.py` | `core/query_parser_lc.py` |
| 호출 | `client.beta.chat.completions.parse` | `ChatOpenAI(...).with_structured_output(ParsedIntent, method="json_schema")` |
| 프롬프트 | `build_system_prompt()` | `build_system_prompt()` (같은 함수) |
| 출력 스키마 | `ParsedIntent` | `ParsedIntent` (같은 클래스) |
| 모델 설정 | gpt-4o-mini, temperature 0, seed 고정 | 같음 |

프롬프트와 스키마, 모델 설정을 똑같이 맞춰 두었기 때문에 결과가 다르면 프레임워크 차이로 볼 수 있습니다.

```bash
pip install langchain-openai

# 두 분석기를 같은 질의 9종으로 비교 (Qdrant 없이 실행 가능)
python compare_parsers.py

# 전체 파이프라인을 LangChain 분석기로 실행
QUERY_PARSER=langchain python test_search.py
```

### 비교 결과 (2026.11.03, gpt-4o-mini)

같은 질의 9종에서 검색 조건이 **8/9 일치**했습니다.

| 질의 | 결과 |
| --- | --- |
| 12월 캐나다 / 캐나다 / 7월 발리 / 8월 / 오늘 날씨 | 일치 (검색 유형·계절·슬롯·영문 검색어까지 같음) |
| 2만원 이하 모자 / 여름 휴양지 신발 | 일치 (가격 상한 20000 등 슬롯 조건 같음) |
| 강남역 카페 (패션 무관) | 일치 (두 버전 모두 unrelated로 차단) |
| 친한 친구 결혼식 하객룩 | **season만 다름** (SDK: null, LangChain: FW) |

결혼식 질의는 시기 단서가 없어 계절을 비워 둘지 채울지가 모델 판단에 맡겨지는 경우입니다. 프레임워크가 바꾼 것은 호출 방식뿐이고, 차이는 프롬프트 규칙이 비어 있는 필드에서만 나왔습니다. 다음 단계로 "시기 단서가 없으면 season은 null"을 프롬프트 규칙에 명시할 수 있습니다.

