import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==============================================================================
# 1. Fashion-CLIP 래퍼 클래스 정의
# ==============================================================================
class FashionCLIPRunner:
    """
    Farfetch 80만 개 의류 데이터셋으로 학습된 도메인 특화 Fashion-CLIP 모델 래퍼
    - Output: 512차원 정규화 임베딩 벡터
    - Transformers 4.45+ 버전의 BaseModelOutputWithPooling 객체 호환 처리
    """
    def __init__(self, model_name="patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">> Fashion-CLIP 모델 로딩 중... (Device: {self.device})")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, image_paths, batch_size=16):
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                # README_V2_2 패치: ModelOutput 호환 처리
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                
                # Cosine Similarity 연산을 위한 L2 단위 벡터 정규화
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                embeddings.extend(feats.cpu().numpy())
                
        return np.array(embeddings)


# ==============================================================================
# 2. Qdrant 클라이언트 및 컬렉션 초기화
# ==============================================================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "musinsa_products"

print(f">> Qdrant 서버 연결 시도: {QDRANT_URL}")
qdrant = QdrantClient(url=QDRANT_URL)

# 기존 테스트 데이터가 있을 경우 초기화 후 재구성 (500개 온전 적재 보장)
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)
    print(f">> 기존 컬렉션 '{COLLECTION_NAME}' 삭제 및 초기화 완료.")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)
print(f">> 신규 컬렉션 '{COLLECTION_NAME}' 생성 완료 (차원: 512, 거리: Cosine)")


# ==============================================================================
# 3. 500개 정제 데이터셋 로드
# ==============================================================================
DATASET_PATH = "outputs/products_dataset_completed_v2.jsonl"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")

products = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            products.append(json.loads(line))

print(f">> 데이터셋 파일 로드 완료: 총 {len(products)}개 상품")


# ==============================================================================
# 4. 이미지 Fallback 매핑 (누끼 -> 정면컷 -> 썸네일 캐시)
# ==============================================================================
valid_records = []
image_paths_to_embed = []

for p in products:
    p_id = str(p.get("product_id"))
    target_img_path = None

    # [1순위] 누끼 이미지 (v2.4 누끼컷 우선, 없을 시 v2 누끼컷 탐색)
    cand_cutout_v2_4 = os.path.join("outputs", "cutouts_v2_4", f"{p_id}.png")
    cand_cutout_v2 = p.get("transparent_cutout_path")
    if not cand_cutout_v2 or not os.path.exists(cand_cutout_v2):
        cand_cutout_v2 = os.path.join("outputs", "cutouts", f"{p_id}.png")

    if os.path.exists(cand_cutout_v2_4):
        target_img_path = cand_cutout_v2_4
    elif os.path.exists(cand_cutout_v2):
        target_img_path = cand_cutout_v2
    else:
        # [2순위] 누끼 실패 상품 Fallback: AI 분류 정면 이미지
        front_info = (p.get("selected_images") or {}).get("front") or {}
        local_front = front_info.get("local_path")
        if local_front and os.path.exists(local_front):
            target_img_path = local_front
        else:
            # [3순위] 로컬 캐시 첫 번째 썸네일/상품 이미지
            candidates = p.get("image_classification_candidates") or []
            for cand in candidates:
                cand_path = cand.get("local_path")
                if cand_path and os.path.exists(cand_path):
                    target_img_path = cand_path
                    break

    if target_img_path and os.path.exists(target_img_path):
        valid_records.append((p, target_img_path))
        image_paths_to_embed.append(target_img_path)
    else:
        print(f"!! 경고: 상품 {p_id}에 연결할 수 있는 로컬 이미지가 없습니다.")

print(f">> 임베딩 대상 유효 이미지 매핑 완료: {len(image_paths_to_embed)} / {len(products)} 건")


# ==============================================================================
# 5. Fashion-CLIP 이미지 임베딩 추출 (Batch 연산)
# ==============================================================================
fclip = FashionCLIPRunner()

BATCH_SIZE = 16
all_embeddings = []
for i in tqdm(range(0, len(image_paths_to_embed), BATCH_SIZE), desc="Fashion-CLIP 500건 임베딩 추출"):
    batch_paths = image_paths_to_embed[i:i + BATCH_SIZE]
    batch_vecs = fclip.encode_images(batch_paths, batch_size=len(batch_paths))
    all_embeddings.extend(batch_vecs)


# ==============================================================================
# 6. Qdrant Payload 구축 및 업서트 (Upsert)
# ==============================================================================
points = []
for idx, (p, img_path) in enumerate(valid_records):
    price_val = int(p.get("price") or p.get("sale_price") or 0)
    
    # 검색 필터링 및 서빙에 필수적인 메타데이터 구성
    payload = {
        "product_id": int(p["product_id"]),
        "product_name": p.get("product_name", ""),
        "brand_name": p.get("brand_name", ""),
        "category": p.get("category") or p.get("source_category", ""),
        "subcategory": p.get("subcategory", ""),
        "price": price_val,
        "color_normalized": p.get("color_normalized", []),
        "fit_normalized": p.get("fit_normalized", []),
        "pattern_normalized": p.get("pattern_normalized", []),
        "material_normalized": p.get("material_normalized", []),
        "image_url": (p.get("selected_images", {}).get("front") or {}).get("image_url", "") or ((p.get("images") or [{}])[0].get("image_url", "")),
        "local_image_path": img_path,
        "is_cutout": "cutouts" in img_path,
        "product_url": p.get("product_url", "")
    }

    points.append(
        models.PointStruct(
            id=int(p["product_id"]),
            vector=all_embeddings[idx].tolist(),
            payload=payload
        )
    )

# Qdrant에 100건씩 배치 Upsert
print(">> Qdrant DB에 포인트 배치 업서트 시작...")
UPSERT_BATCH_SIZE = 100
for i in range(0, len(points), UPSERT_BATCH_SIZE):
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points[i:i + UPSERT_BATCH_SIZE]
    )

print(f"\n========================================================")
print(f">> 인덱싱 성공! Qdrant 컬렉션 '{COLLECTION_NAME}'에 총 {len(points)}건 적재 완료.")
print(f"========================================================")