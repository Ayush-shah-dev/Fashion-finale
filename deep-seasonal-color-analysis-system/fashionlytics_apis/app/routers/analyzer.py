from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from app.services.clip_search import get_text_embedding, get_image_embeddings, rank_images
from app.services.face_season import detect_face_palette
from app.services.file_ops import save_temp_image, save_multiple_images
from app.models.types import AnalyzeResponse

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    face_image: UploadFile = File(...),
    body_type: str = Form(...),
    colors: List[str] = Form(...),
    product_images: List[UploadFile] = File(...)
):
    face_path = save_temp_image(face_image)
    product_paths = save_multiple_images(product_images)

    result = detect_face_palette(face_path)
    prompt = f"cloth for {body_type} of color: {', '.join(colors)}"
    text_emb = get_text_embedding(prompt)
    image_embs, valid_paths = get_image_embeddings(product_paths)
    ranked = rank_images(text_emb, image_embs, valid_paths)

    #extract face

    
    return {
        "season": result["season"],
        "hex_colors": result["hex_colors"],
        "prompt": prompt,
        "top_matches": ranked
    }