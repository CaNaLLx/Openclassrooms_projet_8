import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Charger le modèle une seule fois au démarrage
model = tf.keras.models.load_model("Models/vgg16_unet.keras")

IMG_SIZE = (256, 512)

COLORS = [
    [128, 64, 128],   # flat
    [220, 20, 60],    # human
    [0, 0, 142],      # vehicle
    [70, 70, 70],     # construction
    [153, 153, 153],  # object
    [107, 142, 35],   # nature
    [70, 130, 180],   # sky
    [0, 0, 0],        # void
]

def predict_mask(image_bytes: bytes) -> bytes:
    # Charger et redimensionner l'image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prédiction
    pred = model.predict(arr)
    pred_classes = np.argmax(pred[0], axis=-1)

    # Coloriser le mask
    h, w = pred_classes.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(COLORS):
        color_mask[pred_classes == class_id] = color

    # Retourner en bytes PNG
    out_img = Image.fromarray(color_mask)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
