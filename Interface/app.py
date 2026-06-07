import streamlit as st
import requests
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Config
API_URL = "http://localhost:8000/predict"
IMG_DIR = Path("../Data/Images/val/input")
MASK_DIR = Path("../Data/Images/val/mask")

st.title("🚗 Segmentation d'images - Véhicule autonome")

# Liste des images disponibles
images = sorted(IMG_DIR.glob("*.png"))
image_ids = [img.stem for img in images]

selected_id = st.selectbox("Sélectionne une image", image_ids)

if st.button("Lancer la prédiction"):
    # Chemins
    img_path = IMG_DIR / f"{selected_id}.png"
    
    # Trouver le mask réel correspondant
    mask_files = list(MASK_DIR.glob(f"*{selected_id.replace('leftImg8bit', '')}*"))
    
    # Appel API
    with open(img_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
    
    # Image réelle
    img = Image.open(img_path)
    
    # Mask prédit
    mask_predit = Image.open(io.BytesIO(response.content))
    
    # Affichage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Image réelle")
        st.image(img, use_column_width=True)
    
    with col2:
        st.subheader("Mask réel")
        if mask_files:
            mask_reel = Image.open(mask_files[0])
            st.image(mask_reel, use_column_width=True)
        else:
            st.write("Mask réel non trouvé")
    
    with col3:
        st.subheader("Mask prédit")
        st.image(mask_predit, use_column_width=True)
