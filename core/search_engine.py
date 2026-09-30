import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 2단계: Fashion-CLIP + Qdrant 탐색기

# --- 계절별 배제 서브카테고리 -------------------------------------------------
# payload에 season 필드가 존재하지 않는다(388건 전수 확인). 그래서 계절 적합성은
# 모든 상품에 채워져 있는 subcategory로 판정한다.
#
# '포함 목록'이 아니라 '배제 목록'을 쓴다. 포함 목록으로 좁히면 사계절 아이템
# (long_sleeve_tshirt, denim_pants, baseball_cap, fashion_sneakers 등)이 후보에서
# 통째로 사라져 추천할 상품이 남지 않는다.
SUMMER_ONLY_SUBCATEGORIES = [
    "short_sleeve_tshirt", "shorts", "slides", "clogs", "bucket_hat",
]
WINTER_ONLY_SUBCATEGORIES = [
    "heavy_puffer", "lightweight_puffer", "fleece", "beanie",
    "knit_sweater", "leather_jacket", "hiking_shoes",
]

class FashionSearchEngine:
    def __init__(self, qdrant_url="http://localhost:6333", collection_name="musinsa_products"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        model_name = "patrickjohncyh/fashion-clip"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

    def encode_text(self, text: str) -> list:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            feats = self._normalize(feats)
        return feats.cpu().numpy()[0].tolist()

    def encode_image(self, image_path: str) -> list:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            feats = self._normalize(feats)
        return feats.cpu().numpy()[0].tolist()

    def search(self,
               query_text: str = None,
               query_image_path: str = None,
               category: str = None,
               color: str = None,
               season: str = None,       # 'SS' 또는 'FW' (subcategory 배제 방식으로 적용)
               max_price: int = None,
               top_k: int = 5) -> list:

        if query_text:
            query_vector = self.encode_text(query_text)
        elif query_image_path and os.path.exists(query_image_path):
            query_vector = self.encode_image(query_image_path)
        else:
            raise ValueError("query_text 또는 유효한 query_image_path가 필요합니다.")

        must_conditions = []
        if category:
            must_conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
        if color:
            must_conditions.append(models.FieldCondition(key="color_normalized", match=models.MatchAny(any=[color])))
        if max_price is not None:
            must_conditions.append(models.FieldCondition(key="price", range=models.Range(lte=max_price)))

        # 🎯 계절 필터: season 필드가 없으므로 subcategory 배제로 처리한다.
        # (기존의 key="season" MatchValue는 항상 0건이라 계절 필터가 무의미했고,
        #  그 탓에 coordinator의 Fallback이 매번 발동해 color 필터까지 버려졌다)
        must_not_conditions = []
        if season == "FW":
            must_not_conditions.append(
                models.FieldCondition(key="subcategory", match=models.MatchAny(any=SUMMER_ONLY_SUBCATEGORIES))
            )
        elif season == "SS":
            must_not_conditions.append(
                models.FieldCondition(key="subcategory", match=models.MatchAny(any=WINTER_ONLY_SUBCATEGORIES))
            )

        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = models.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        results = []
        for hit in response.points:
            p = hit.payload
            results.append({
                "similarity_score": round(hit.score, 4),
                "product_id": p.get("product_id"),
                "product_name": p.get("product_name"),
                "brand_name": p.get("brand_name"),
                "category": p.get("category"),
                "subcategory": p.get("subcategory"),
                "price": p.get("price"),
                "colors": p.get("color_normalized"),
                "local_image_path": p.get("local_image_path"),
                "product_url": p.get("product_url")
            })
        return results
