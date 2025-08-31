import torch
from PIL import Image
from face_body_segeration import process_image
from face_season_detection import detect_face_palette

def suggest_color_palette(user_image_path: str) -> dict:
    """
    Given a path to a user image, extract the face, classify the color season,
    and return the suggested palette.

    Args:
        user_image_path (str): Path to the user's full-body image.

    Returns:
        dict: {
            "season": str,
            "rgb_colors": list[tuple[int, int, int]],
            "hex_colors": list[str],
            "tile_plot": BytesIO (optional visual buffer)
        }
    """
    # Segment face and body
    crop_result = process_image(user_image_path)

    if not crop_result or "face_image" not in crop_result:
        raise ValueError("Face image could not be extracted.")

    # Get cropped face as a PIL image
    face_image_pil = crop_result["face_image"]

    # Run season classification pipeline
    suggested_palette = detect_face_palette(face_image_pil)

    return suggested_palette
