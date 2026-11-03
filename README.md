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
rapport/rapport.pdf       rapport technique
rapport/build_report.py   génère le rapport et ses figures (rapport/figures/) à partir des résultats
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

Le rapport PDF se régénère à partir des résultats avec `python rapport/build_report.py` (dépendances : `reportlab`, `matplotlib`, `pandas`).

## Principales conclusions

Mesures Colab : CPU Intel Xeon 2,2 GHz (2 vCPU, 1 thread PyTorch) et GPU Tesla T4. Moyenne ± écart-type sur 10 exécutions des 50 images, en batchs de 10. Le détail est dans le [rapport](rapport/rapport.pdf) et dans [results/synthese.csv](results/synthese.csv).

| Pipeline | Variante | Débit (img/s) | Temps 50 images (ms) | Gain vs eager | Énergie GPU / image |
|---|---|---|---|---|---|
| CPU n°1 | PyTorch eager (réf.) | 13,1 ± 1,3 | 3865 ± 405 | – | n/d |
| | OpenVINO FP32 | 17,8 ± 2,0 | 2842 ± 362 | ×1,37 | n/d |
| CPU n°2 | PyTorch eager (réf.) | 11,6 ± 0,9 | 4352 ± 341 | – | n/d |
| | TorchScript (freeze) | 14,4 ± 2,3 | 3555 ± 649 | ×1,25 | n/d |
| | torch.compile (Inductor) | 15,6 ± 1,5 | 3243 ± 327 | ×1,35 | n/d |
| GPU n°1 | PyTorch eager (réf.) | 803 ± 33 | 62,4 ± 2,8 | – | 84 mJ |
| | TensorRT FP32 | 1212 ± 31 | 41,3 ± 1,0 | ×1,51 | 56 mJ |
| | TensorRT FP16 | 4095 ± 480 | 12,4 ± 2,0 | ×5,10 | 15 mJ |
| GPU n°2 | PyTorch eager (réf.) | 830 ± 7 | 60,3 ± 0,5 | – | 84 mJ |
| | TorchScript (freeze) | 916 ± 14 | 54,6 ± 0,8 | ×1,10 | 75 mJ |
| | torch.compile (Inductor) | 539 ± 8 | 92,9 ± 1,5 | ×0,65 | 126 mJ |

- **Le matériel prime** : à code égal, le T4 est environ 72 fois plus rapide que le CPU Colab.
- **TensorRT est la chaîne la plus performante** : ×1,5 en FP32 avec des prédictions identiques, et ×5,1 en FP16 grâce aux Tensor Cores, pour 82 % d'énergie en moins par image. Le GPU tourne à pleine puissance (67-69 W) dans toutes les variantes, donc l'énergie par image suit directement le débit.
- **Sur CPU, les trois optimisations se valent** (×1,25 à ×1,37) : leurs écarts sont noyés dans la variabilité de la VM partagée (CV de 8 à 16 %). OpenVINO compile en 0,5 s, contre 25 s pour torch.compile.
- **torch.compile est plus lent qu'eager sur le T4 en FP32** (×0,65) : une compilation doit toujours être mesurée face à la référence.
- **Aucune optimisation ne change une prédiction** : précision Top-1 de 64 % partout et accord de 100 % entre les 11 variantes. L'écart sur les logits est d'environ 3·10⁻⁵ en FP32 et de 6·10⁻² en FP16. Le score de 64 % vient du modèle ImageNet appliqué sans réentraînement à des images 32×32 agrandies, et du regroupement des classes (11 des 18 erreurs sont prédites *dog*, un groupe de 118 classes).
- **Méthode** : les mesures CPU varient de 13 % d'une session Colab à l'autre, d'où une référence eager par notebook. Sur GPU, la première exécution mesurée est parfois lente, signe que la chauffe est trop courte.
