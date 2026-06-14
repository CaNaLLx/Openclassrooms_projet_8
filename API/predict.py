import numpy as np
import tensorflow as tf
from PIL import Image
import io

# ✅ Charger le modèle TFLite light
interpreter = tf.lite.Interpreter(
    model_path="vgg16_unet_quantized.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

def predict_mask(image_bytes: bytes) -> dict:
    """
    Prédire avec le modèle TFLite light (rapide & léger).
    """
    try:
        # 1️⃣ Charger et redimensionner
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # 2️⃣ Prédiction avec TFLite
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        
        pred_classes = np.argmax(pred[0], axis=-1)

        # 3️⃣ Coloriser
        h, w = pred_classes.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in enumerate(COLORS):
            color_mask[pred_classes == class_id] = color

        # 4️⃣ Retourner en bytes PNG
        out_img = Image.fromarray(color_mask)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)
        
        return {
            "mask_bytes": buf.read(),
            "success": True,
            "classes": pred_classes.tolist()
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }
