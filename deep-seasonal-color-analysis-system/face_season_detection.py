# face_season_detection.py

import torch
from PIL import Image
import io
from face_body_segeration import process_image
from pipeline import pipeline, segmentation_filter, user_palette_classification_filter
from utils import segmentation_labels, utils
import matplotlib.pyplot as plt
from palette_classification import color_processing, palette
import glob
import json


def detect_face_palette(input_face_pil: Image.Image):
    """
    Extracts the face from an image, classifies it into a seasonal color palette,
    and returns the palette name, RGB & HEX codes, and a tile plot image buffer.

    Args:
        input_image_path (str): Full path to the input image

    Returns:
        dict: {
            "season": str,
            "rgb_colors": list[tuple[int, int, int]],
            "hex_colors": list[str],
            "tile_plot": BytesIO buffer (for Streamlit)
        }
    """

    # Step 2: Prepare pipeline
    device = "cpu"
    palettes_path = "palette_classification/palettes/"

    palette_filenames = glob.glob(palettes_path + "*.csv")
    reference_palettes = [
        palette.PaletteRGB().load(fname.replace("\\", "/"), header=True)
        for fname in palette_filenames
    ]

    # Build pipeline
    pl = pipeline.Pipeline()
    pl.add_filter(segmentation_filter.SegmentationFilter("local"))
    pl.add_filter(user_palette_classification_filter.UserPaletteClassificationFilter(reference_palettes))

    # Run the pipeline
    input_image =image = input_face_pil.convert("RGB")

    user_palette = pl.execute(input_image, device, verbose=False)

    # Step 3: Get colors
    colors_tensor = user_palette.colors()  # [3, 1, N]
    colors_array = colors_tensor.squeeze(1).permute(1, 0).int().numpy()
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(*rgb) for rgb in colors_array]
    rgb_colors = [tuple(rgb) for rgb in colors_array]

    # Step 4: Generate tile plot and save to memory
    plt.figure(figsize=(5, 1))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(utils.from_DHW_to_HWD(colors_tensor).numpy())
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    return {
        "season": user_palette.description(),
        "rgb_colors": rgb_colors,
        "hex_colors": hex_colors,
        "tile_plot": buf
    }
