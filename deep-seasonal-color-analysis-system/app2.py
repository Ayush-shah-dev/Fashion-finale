import streamlit as st
st.set_page_config(page_title="Fashionlytics CLIP Search", layout="wide")

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
from transformers import CLIPProcessor, CLIPModel
from face_season_detection import detect_face_palette
from tqdm import tqdm

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

clip_model, clip_processor = load_clip()


st.title("AI-Powered Fashion Finder with CLIP")

# ---------- STEP 1: Upload product images ----------
st.header("Upload Folder of Product Images")
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
product_images = []

if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.read())
            product_images.append(tmp.name)
    st.success(f"{len(product_images)} images uploaded successfully!")

# ---------- STEP 2: Upload face image for seasonal analysis ----------
st.header("Upload Your Face Image for Color Season Analysis")
face_file = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"], key="face")

hex_colors = []
default_colors = []

if face_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(face_file.read())
        face_image_path = tmp_file.name
        
    st.subheader("Your Uploaded Face Image")
    # Fixed image display with smaller width
    st.image(face_file, caption="Uploaded Face", width=300)

    with st.spinner("Detecting your season palette..."):
        result = detect_face_palette(face_image_path)

    st.markdown(f"**Detected Season:** `{result['season']}`")
    # Fixed palette image display with smaller width
    st.image(result["tile_plot"], caption="Suggested Color Palette", width=400)

    hex_colors = result["hex_colors"]
    default_colors = hex_colors[:4]

# ---------- STEP 3: User selects colors from palette ----------
st.header("Select Preferred Colors (AI suggested or customize)")

if hex_colors:
    selected_colors = st.multiselect(
        "Choose colors for product search",
        options=hex_colors,
        default=default_colors,
        format_func=lambda x: f"{x.upper()}"
    )
else:
    selected_colors = st.multiselect("Choose colors (no face uploaded)", options=[
        "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ffffff", "#000000"
    ])

# ---------- STEP 4: Select body shape ----------
st.header("Select Your Body Shape")
body_types = ["slim", "curvy", "athletic", "plus-size"]
selected_body = st.selectbox("Choose body type", body_types)

# ---------- STEP 5: Generate Prompt and Run CLIP Search ----------
if product_images and selected_colors:
    st.header("Top Matches")
    prompt = f"cloth for {selected_body} of color: {', '.join(selected_colors)}"
    st.markdown(f"**Generated Prompt:** `{prompt}`")

    def get_text_embedding(text):
        try:
            inputs = clip_processor([text], return_tensors="pt", padding=True)
            text_features = clip_model.get_text_features(**inputs)
            
            if text_features.shape[0] == 0:
                st.error("Failed to generate text embeddings")
                return None
                
            return text_features[0]
        except Exception as e:
            st.error(f"Error generating text embedding: {e}")
            return None

    def get_image_embeddings(image_paths):
        try:
            images = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    st.warning(f"Could not load image {path}: {e}")
                    continue
            
            if not images:
                st.error("No valid images found")
                return None, []
            
            inputs = clip_processor(images=images, return_tensors="pt")
            image_features = clip_model.get_image_features(**inputs)
            return image_features, valid_paths
        except Exception as e:
            st.error(f"Error generating image embeddings: {e}")
            return None, []

    text_emb = get_text_embedding(prompt)
    if text_emb is None:
        st.error("Could not generate text embedding. Please try again.")
        st.stop()
        
    image_embs, valid_image_paths = get_image_embeddings(product_images)
    if image_embs is None:
        st.error("Could not generate image embeddings. Please check your images.")
        st.stop()

    # Ranking
    cosine_sim = torch.nn.CosineSimilarity(dim=1)
    repeated_text_emb = text_emb.unsqueeze(0).expand(image_embs.shape[0], -1)
    scores = cosine_sim(repeated_text_emb, image_embs).detach().cpu().numpy()


    # Show top 5–7 images
    idx_sorted = np.argsort(scores)[::-1]
    top_indices = idx_sorted[:min(7, len(scores))]

    cols = st.columns(3)
    for i, idx in enumerate(top_indices):
        with cols[i % 3]:
            try:
                # Fixed image display with smaller width
                st.image(valid_image_paths[idx], caption=f"Score: {scores[idx]:.2f}", width=250)
            except Exception as e:
                st.error(f"Error displaying image {idx}: {e}")

else:
    st.info("Upload product images and select at least one color to view results.")