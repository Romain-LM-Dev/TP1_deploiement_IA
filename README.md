# TP1 Déploiement IA : 4 pipelines d'inférence ResNet-18 sur CPU et GPU

Comparaison de quatre chaînes de déploiement d'un même modèle (ResNet-18 pré-entraîné ImageNet) sur les mêmes 50 images CIFAR-10, exécutées sur Google Colab : débit, latence, précision Top-1, cohérence des prédictions et énergie, avec 10 répétitions par mesure.

| # | Plateforme | Pipeline | Notebook |
|---|---|---|---|
| 1 | CPU | PyTorch → ONNX → **OpenVINO Runtime** → oneDNN | [01_cpu_openvino](notebooks/01_cpu_openvino.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romain-LM-Dev/TP1_deploiement_IA/blob/main/notebooks/01_cpu_openvino.ipynb) |
| 2 | CPU | PyTorch → **TorchScript / torch.compile** (Inductor) | [02_cpu_torchcompile](notebooks/02_cpu_torchcompile.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romain-LM-Dev/TP1_deploiement_IA/blob/main/notebooks/02_cpu_torchcompile.ipynb) |
| 3 | GPU | PyTorch → ONNX → **TensorRT** (FP32, FP16) → CUDA/cuDNN | [03_gpu_tensorrt](notebooks/03_gpu_tensorrt.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romain-LM-Dev/TP1_deploiement_IA/blob/main/notebooks/03_gpu_tensorrt.ipynb) |
| 4 | GPU | PyTorch → **TorchScript / torch.compile** (Inductor, Triton) → CUDA | [04_gpu_torchcompile](notebooks/04_gpu_torchcompile.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romain-LM-Dev/TP1_deploiement_IA/blob/main/notebooks/04_gpu_torchcompile.ipynb) |
| – | – | Agrégation des résultats et graphiques comparatifs | [05_comparaison](notebooks/05_comparaison.ipynb) |

Chaque notebook mesure aussi **PyTorch eager sur la même machine** : cette référence sert à calculer le gain de chaque optimisation et à vérifier qu'elle ne change pas les prédictions.

## Organisation du dépôt

```
common.py                 code partagé : données, modèle, mapping des classes, protocole de mesure, énergie, graphiques
prepare_data.py           extraction (unique) des 50 images CIFAR-10 utilisées
data/cifar10_test_50.npz  les 50 premières images du jeu de test CIFAR-10 (uint8, 32x32) + étiquettes
notebooks/                un notebook par pipeline + notebook de comparaison
results/                  résultats bruts (JSON) et tableau de synthèse (CSV)
figures/                  graphiques générés
rapport/                  rapport technique (PDF)
```

## Protocole expérimental (commun aux 4 pipelines, défini dans `common.py`)

| Paramètre | Valeur |
|---|---|
| Modèle | `torchvision` ResNet-18, poids `IMAGENET1K_V1`, mode `eval` |
| Données | 50 premières images du test CIFAR-10, versionnées dans `data/` (aucun téléchargement, mêmes pixels partout) |
| Prétraitement | 32×32 → 224×224 bilinéaire, normalisation ImageNet mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)`, sans recadrage central (il couperait l'objet) |
| Classes | les 1000 classes ImageNet sont regroupées en 10 classes CIFAR (ex. les 118 races de chiens → *dog*) ; la classe prédite est celle du meilleur logit. *deer* n'existe pas dans ImageNet : on utilise les antilopes (ibex, impala, gazelle, bubale) |
| Batch | 10 images (5 batchs), batch fixe dans les exports ONNX |
| Répétitions | 1 premier appel chronométré à part (compilation JIT), 3 passes de chauffe, puis **10 exécutions mesurées** |
| Chronométrage | `time.perf_counter`, données déjà sur le périphérique, `torch.cuda.synchronize()` après chaque batch sur GPU |
| Statistiques | moyenne, écart-type, min, max et coefficient de variation pour le débit, le temps total, la latence par batch et l'énergie |
| Énergie | compteur matériel NVML (GPU) ou RAPL (CPU, s'il est accessible), intégré sur ≥ 3 s d'inférence après chaque exécution |
| Précision | FP32 partout pour la comparaison principale ; TensorRT FP16 en variante supplémentaire |

## Exécution

**Avec Google Colab dans le navigateur** : cliquer sur le badge Colab d'un notebook, choisir le type d'exécution (**CPU** pour 01-02, **GPU T4** pour 03-04), puis *Exécution > Tout exécuter*. Le notebook clone ce dépôt puis importe `common.py`. Le JSON de résultats est téléchargé à la fin : le déposer dans `results/`. Enregistrer le notebook exécuté dans le dépôt (*Fichier > Enregistrer une copie dans GitHub*).

**Avec l'extension Colab de VS Code** : ouvrir le notebook, *Select Kernel > Colab* puis choisir le runtime (CPU ou T4), et *Run All*. Il suffit ensuite d'enregistrer le notebook : les résultats sont intégrés à ses sorties, et `05_comparaison.ipynb` les y retrouve s'il n'y a pas de fichier dans `results/`.

Ensuite, exécuter `05_comparaison.ipynb` (n'importe quel environnement, sans GPU) pour produire `figures/` et `results/synthese.csv`.

Exécution locale (CPU) : `pip install torch torchvision openvino onnx pandas matplotlib`, puis lancer les notebooks depuis `notebooks/`. Sous Windows, `torch.compile` exige le compilateur MSVC.

## Principales conclusions

*À compléter avec les mesures Colab (voir `rapport/`).*

Tests préliminaires en local (i7-11800H, hors protocole Colab) : OpenVINO FP32 traite environ **1,8× plus d'images/s** que PyTorch eager, et TorchScript environ 1,08×. Les prédictions sont identiques (écart max des logits ≈ 2·10⁻⁵) et la précision Top-1 est de **64 %** sur les 50 images, une limite qui vient du modèle ImageNet utilisé sans réentraînement et non des optimisations.
