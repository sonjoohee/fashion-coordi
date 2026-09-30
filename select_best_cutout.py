import os
import shutil
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# 1. 설정 및 경로
SRC_ROOT = Path("fashion_clip_images")
DST_DIR = Path("outputs/cutouts_v2_4")
DST_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 2. CLIP 필터링 프롬프트
POS_PROMPTS = [
    "a full product photo showing the entire clothing item laid flat alone",
    "an isolated single fashion garment displayed alone with no person",
    "front view of the whole clothing product"
]

NEG_PROMPTS = [
    "a person or model wearing clothes with shoes and body",
    "a close-up macro detail of fabric, zipper, stitch, tag, or texture",
    "an extreme close-up cropped view of a garment"
]

print(f">> Device: {DEVICE}")
print(">> CLIP 모델 로딩 중...")
model_name = "patrickjohncyh/fashion-clip"
clip = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained(model_name)

# 텍스트 임베딩 사전 계산
with torch.inference_mode():
    pos_inp = proc(text=POS_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    neg_inp = proc(text=NEG_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    pos_vec = clip.get_text_features(**pos_inp).mean(dim=0, keepdim=True)
    neg_vec = clip.get_text_features(**neg_inp).mean(dim=0, keepdim=True)
    pos_vec = pos_vec / pos_vec.norm(dim=-1, keepdim=True)
    neg_vec = neg_vec / neg_vec.norm(dim=-1, keepdim=True)

# 3. 상품별 최적 이미지 선별 루프
product_dirs = [p for p in SRC_ROOT.glob("*/*") if p.is_dir()]
print(f">> 총 {len(product_dirs)}개 상품 최적 컷 선별 시작")

success = 0

for pdir in tqdm(product_dirs, desc="Selecting Best Cutout"):
    pid = pdir.name
    
    # 1순위: goods_images 우선 탐색, 없으면 thumbnail_images
    candidates = list((pdir / "goods_images").glob("*.png"))
    if not candidates:
        candidates = list((pdir / "thumbnail_images").glob("*.png"))
        
    if not candidates:
        continue

    best_score = -float("inf")
    best_img_path = None

    for cpath in candidates:
        try:
            with Image.open(cpath) as img:
                # [필터링 1] 극단적인 디테일 확대컷 사전 배제 (마스크 알파 채널 비율)
                # 화면의 88% 이상이 불투명 옷감으로 꽉 차 있다면 100% 넥라인/원단 확대 컷
                if img.mode == "RGBA":
                    alpha = np.array(img.split()[-1])
                    coverage = (alpha > 0).mean()
                    if coverage > 0.88 or coverage < 0.03:
                        continue
                
                # RGB 변환 후 CLIP 입력
                rgb_img = img.convert("RGB")
                inp = proc(images=rgb_img, return_tensors="pt").to(DEVICE)
                
                with torch.inference_mode():
                    feat = clip.get_image_features(**inp)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    
                    pos_sim = torch.cosine_similarity(feat, pos_vec).item()
                    neg_sim = torch.cosine_similarity(feat, neg_vec).item()
                    
                    # 단품 전체 컷 점수 가산 (+), 모델 착용 및 디테일 확대 컷 감산 (-)
                    final_score = pos_sim - neg_sim

                if final_score > best_score:
                    best_score = final_score
                    best_img_path = cpath

        except Exception:
            continue

    # 선별된 최적의 단품 컷 복사
    if best_img_path:
        shutil.copy2(best_img_path, DST_DIR / f"{pid}.png")
        success += 1

print("\n" + "=" * 60)
print(f"🎉 선별 완료: 총 {success}개 상품의 최적 단품 정면 컷을 {DST_DIR} 에 저장했습니다.")
print("=" * 60)