import torch
from PIL import Image
from face_body_segeration import process_image
from face_season_detection import detect_face_palette


def suggest_color_palette(user_image_path):
    # Your segmentation, filters, clustering, and color season logic here
    # Return dict with keys: season, hex_colors (list), tile_plot (optional path)
    cropped_img=process_image(input_path=user_image_path)
    face_pil=cropped_img['face_image']
    
    suggested_pallet= detect_face_palette(face_pil)

   
    return suggested_pallet