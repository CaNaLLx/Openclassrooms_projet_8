from pathlib import Path
import numpy as np
import cv2

IMG_DIR = "Data/Images"
IMG_INPUT = "input"
IMG_MASK = "mask"

NUM_CLASSES = 8

LABEL_TO_CATEGORY = np.array([
    0, 0, 0, 0, 0, 0, 0,   # 0-6  : void
    1, 1, 1, 1,            # 7-10 : flat
    2, 2, 2, 2, 2, 2,      # 11-16: construction
    3, 3, 3, 3,            # 17-20: object
    4, 4,                  # 21-22: nature
    5,                     # 23   : sky
    6, 6,                  # 24-25: human
    7, 7, 7, 7, 7, 7, 7, 7 # 26-33: vehicle
], dtype=np.uint8)

CATEGORY_NAMES = ["void", "flat", "construction", "object",
                  "nature", "sky", "human", "vehicle"]

#FONCTION POUR CHARGER UNE PAIRE D'IMAGES (IMAGE + MASQUE)
def load_pair(img_path, mask_path, img_size=(256, 512)):
    """Charge une image + son masque remappé en 8 classes."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = LABEL_TO_CATEGORY[mask]  # 33 → 8 classes

    return img, mask


def load_split(split="train", img_size=(256, 512), max_samples=None):
    """
    Charge un split (train/val/test).
    Suppose que input/<name>.png correspond à mask/<même_name>.png
    (ou mapping leftImg8bit → gtFine_labelIds).
    """
    input_dir = Path(IMG_DIR) / split / IMG_INPUT
    mask_dir  = Path(IMG_DIR) / split / IMG_MASK

    input_files = sorted(input_dir.glob("*.png"))
    if max_samples:
        input_files = input_files[:max_samples]

    X, y = [], []
    for img_path in input_files:
        # Trouver le masque correspondant
        stem = img_path.stem.replace("_leftImg8bit", "")
        # Essaie plusieurs noms possibles
        candidates = [
            mask_dir / f"{stem}_gtFine_labelIds.png",
            mask_dir / f"{img_path.stem}.png",
            mask_dir / f"{stem}.png",
        ]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            print(f"Pas de masque pour {img_path.name}")
            continue

        img, mask = load_pair(img_path, mask_path, img_size)
        X.append(img)
        y.append(mask)

    print(f"[{split}] {len(X)} paires chargées")
    return np.array(X), np.array(y)