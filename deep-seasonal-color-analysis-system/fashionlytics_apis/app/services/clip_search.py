import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Tuple

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text: str):
    inputs = processor([text], return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)[0]

def get_image_embeddings(image_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except:
            continue
    inputs = processor(images=images, return_tensors="pt")
    return model.get_image_features(**inputs), valid_paths

def rank_images(text_emb, image_embs, paths):
    cosine_sim = torch.nn.CosineSimilarity(dim=1)
    repeated_text = text_emb.unsqueeze(0).expand(image_embs.shape[0], -1)
    scores = cosine_sim(repeated_text, image_embs).detach().cpu().numpy()
    idx_sorted = scores.argsort()[::-1]
    return [{"image_path": paths[i], "score": float(scores[i])} for i in idx_sorted[:7]]
