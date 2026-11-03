"""Extrait les 50 premières images du jeu de test CIFAR-10 dans data/cifar10_test_50.npz.

Exécuté une seule fois ; le fichier produit (150 Ko) est versionné pour que les 4 pipelines
lisent exactement les mêmes pixels sans dépendre d'un téléchargement.
Source : miroir Hugging Face du jeu officiel (même ordre que le test_batch d'origine,
étiquettes vérifiées : 3, 8, 8, 0, 6, 6, 1, 6, 3, 1, ...).
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"

df = pd.read_parquet(URL).iloc[:50]
images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df["img"]])
labels = df["label"].to_numpy(np.int64)
assert images.shape == (50, 32, 32, 3) and images.dtype == np.uint8
assert labels[:10].tolist() == [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
os.makedirs("data", exist_ok=True)
np.savez_compressed("data/cifar10_test_50.npz", images=images, labels=labels)
print("data/cifar10_test_50.npz :", images.shape, np.bincount(labels, minlength=10))
