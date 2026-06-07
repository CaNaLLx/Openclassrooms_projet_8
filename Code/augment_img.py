# Code/augment_img.py
import numpy as np
import albumentations as A
from Code.load_img import load_split


def get_augmentation():
    """Pipeline d'augmentation (image + masque synchronisés)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomGamma(p=0.2),
    ])


def augment_split(split="train", img_size=(256, 512), max_samples=None, n_aug=1):
    """
    Charge un split et applique de la data augmentation.

    Args:
        split       : "train", "val" ou "test"
        img_size    : taille des images
        max_samples : limite éventuelle
        n_aug       : nombre de versions augmentées par image

    Returns:
        X_aug, y_aug : images et masques augmentés (concaténés avec l'original)
    """
    X, y = load_split(split=split, img_size=img_size, max_samples=max_samples)
    aug = get_augmentation()

    X_aug, y_aug = [X], [y]

    for _ in range(n_aug):
        X_new, y_new = [], []
        for img, mask in zip(X, y):
            out = aug(image=img, mask=mask)
            X_new.append(out["image"])
            y_new.append(out["mask"])
        X_aug.append(np.array(X_new))
        y_aug.append(np.array(y_new))

    X_final = np.concatenate(X_aug, axis=0)
    y_final = np.concatenate(y_aug, axis=0)

    print(f"[{split}] {len(X)} originales + {n_aug}×{len(X)} augmentées = {len(X_final)} total")
    return X_final, y_final
