"""
SIFT + BoVW + RandomForest — baseline simple pour segmentation par patchs.
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


class SIFTSegmenter:
    def __init__(self, patch_size=16, n_clusters=50):
        self.patch_size = patch_size
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def _patch_feature(self, patch):
        """Histogramme BoVW d'un patch."""
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        _, desc = self.sift.detectAndCompute(gray, None)
        hist = np.zeros(self.n_clusters, dtype=np.float32)
        if desc is not None:
            for w in self.kmeans.predict(desc):
                hist[w] += 1
            if hist.sum() > 0:
                hist /= hist.sum()
        return hist

    def _iter_patches(self, img):
        ps = self.patch_size
        h, w = img.shape[:2]
        for i in range(0, h - ps + 1, ps):
            for j in range(0, w - ps + 1, ps):
                yield i, j, img[i:i + ps, j:j + ps]

    def fit(self, X, y):
        # 1. Vocabulaire visuel (KMeans sur tous les descripteurs)
        print("[SIFT] Construction du vocabulaire...")
        all_desc = []
        for img in tqdm(X):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, desc = self.sift.detectAndCompute(gray, None)
            if desc is not None:
                all_desc.append(desc)
        self.kmeans.fit(np.vstack(all_desc))

        # 2. Features + labels par patch
        print("[SIFT] Extraction features patchs...")
        feats, labels = [], []
        for img, mask in tqdm(list(zip(X, y))):
            for i, j, patch in self._iter_patches(img):
                feats.append(self._patch_feature(patch))
                p_mask = mask[i:i + self.patch_size, j:j + self.patch_size]
                vals, counts = np.unique(p_mask, return_counts=True)
                labels.append(vals[np.argmax(counts)])

        # 3. RandomForest
        print("[SIFT] Entraînement RandomForest...")
        self.clf.fit(np.array(feats), np.array(labels))
        print("[SIFT] ✅ Terminé.")
        return self

    def predict(self, X):
        masks = []
        ps = self.patch_size
        for img in tqdm(X, desc="[SIFT] Prédiction"):
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for i, j, patch in self._iter_patches(img):
                pred = self.clf.predict(self._patch_feature(patch).reshape(1, -1))[0]
                mask[i:i + ps, j:j + ps] = pred
            masks.append(mask)
        return np.array(masks)

    def score(self, X, y):
        """Pixel accuracy moyenne."""
        y_pred = self.predict(X)
        return (y_pred == y).mean()
