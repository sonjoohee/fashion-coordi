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
