import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 2단계: Fashion-CLIP + Qdrant 탐색기


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
               season: str = None,       # 'SS' 또는 'FW' (사계절 'ALL' 상품은 항상 포함)
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

        # 🎯 계절 필터: 요청 계절과 사계절('ALL') 상품을 함께 허용한다.
        # MatchValue(season)만 쓰면 사계절 상품이 전부 배제되어 후보가 말라버린다.
        if season:
            must_conditions.append(
                models.FieldCondition(key="season", match=models.MatchAny(any=[season, "ALL"]))
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

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
                "season": p.get("season"),
                "price": p.get("price"),
                "colors": p.get("color_normalized"),
                "local_image_path": p.get("local_image_path"),
                "product_url": p.get("product_url")
            })
        return results
