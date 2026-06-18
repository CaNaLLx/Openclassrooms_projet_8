import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Chemin du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Models", "vgg16_unet_quantized.tflite")

# Charge le modèle une fois
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Couleurs des 8 classes
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [153, 153, 153], [107, 142, 35], [70, 130, 180], [220, 20, 60]
], dtype=np.uint8)


def predict_mask(image_bytes):
    # Charge l'image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    original_size = img.size

    # Taille attendue par le modèle
    _, h, w, _ = input_details[0]['shape']

    # Prépare l'image
    img_array = np.array(img.resize((w, h)), dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prédiction
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Mask colorisé
    mask = np.argmax(output[0], axis=-1).astype(np.uint8)
    mask_color = PALETTE[mask]

    # Retourne en PNG
    mask_img = Image.fromarray(mask_color).resize(original_size, Image.NEAREST)
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return buffer.getvalue()
