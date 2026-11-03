"""Code partagé par les 4 pipelines : mêmes données, même modèle, même protocole de mesure.

Chaque notebook importe ce module, ce qui garantit que les 4 pipelines utilisent
exactement le même sous-ensemble d'images, le même prétraitement, la même taille
de batch et la même méthode de chronométrage.
"""
import json
import os
import platform
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------- protocole
SEED = 0
N_IMAGES = 50          # 50 premières images du jeu de test CIFAR-10
BATCH_SIZE = 10        # 5 batchs de 10 images, identique pour tous les pipelines
INPUT_SIZE = 224       # résolution d'entrée de ResNet-18
N_WARMUP = 3           # passes complètes de chauffe, non mesurées
N_RUNS = 10            # exécutions mesurées (>= 3 demandées)
ENERGY_MIN_S = 3.0     # durée mini d'une mesure d'énergie (compteurs matériels peu résolus)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # normalisation ImageNet

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# Indices ImageNet-1k regroupés par classe CIFAR-10. La prédiction CIFAR est la
# classe dont le meilleur logit ImageNet est le plus élevé.
# "deer" n'existe pas dans ImageNet : on utilise les antilopes les plus proches.
IMAGENET_GROUPS = [
    [404, 895, 908],                                             # airplane: airliner, warplane, wing
    [436, 468, 511, 609, 627, 656, 661, 717, 751, 817],          # automobile: wagon, cab, convertible, jeep...
    list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),  # bird
    list(range(281, 286)),                                       # cat: tabby, tiger cat, persian, siamese, egyptian
    [350, 351, 352, 353],                                        # deer -> ibex, hartebeest, impala, gazelle
    list(range(151, 269)),                                       # dog: toutes les races
    [30, 31, 32],                                                # frog
    [339, 603],                                                  # horse: sorrel, horse cart
    [403, 472, 484, 510, 554, 576, 625, 628, 724, 780, 814, 833, 871, 913, 914],  # ship
    [407, 555, 569, 675, 734, 864, 867],                         # truck: ambulance, fire engine, garbage...
]


# ---------------------------------------------------------------- données / modèle
def load_data(path="data/cifar10_test_50.npz"):
    """50 premières images CIFAR-10 test (voir prepare_data.py), 32x32 -> 224x224 + normalisation ImageNet."""
    tf = T.Compose([
        T.ToTensor(),  # uint8 HWC -> float CHW dans [0, 1]
        # Pas de CenterCrop : sur une image 32x32 il couperait l'objet.
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.Normalize(MEAN, STD),
    ])
    d = np.load(path)
    x = torch.stack([tf(img) for img in d["images"][:N_IMAGES]]).contiguous()
    return x, d["labels"][:N_IMAGES]


def load_model():
    torch.manual_seed(SEED)
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    return torchvision.models.resnet18(weights=w).eval()


def make_batches(x):
    """Découpe en batchs de BATCH_SIZE (marche pour un tenseur torch ou un tableau numpy)."""
    return [x[i:i + BATCH_SIZE] for i in range(0, N_IMAGES, BATCH_SIZE)]


def export_onnx(model, path="resnet18.onnx"):
    """Export ONNX à batch fixe (BATCH_SIZE), commun à OpenVINO et TensorRT. Modèle sur CPU."""
    dummy = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    return path


def to_cifar(logits):
    logits = np.asarray(logits)
    return np.stack([logits[:, g].max(1) for g in IMAGENET_GROUPS], 1).argmax(1)


def _to_numpy(o):
    return o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o, dtype=np.float32)


# ---------------------------------------------------------------- énergie
class EnergyMeter:
    """Compteur d'énergie matériel cumulé (joules) : NVML pour le GPU, RAPL pour le CPU.

    read() renvoie None si aucun compteur n'est accessible (cas fréquent du CPU sur Colab).
    """

    def __init__(self, device):
        self.source, self.read = None, lambda: None
        if device == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # mJ depuis le chargement du driver
                self.read = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1e3
                self.source = "NVML (GPU seul)"
            except Exception as e:
                print("Pas de mesure d'énergie GPU :", e)
        else:
            f = "/sys/class/powercap/intel-rapl:0/energy_uj"
            try:
                with open(f) as fh:
                    int(fh.read())
                self.read = lambda: int(open(f).read()) / 1e6
                self.source = "RAPL (package CPU)"
            except (OSError, ValueError) as e:
                print("Pas de mesure d'énergie CPU (RAPL inaccessible) :", e)


def _measure_energy(infer, batches, sync, meter):
    """Rejoue les 50 images en boucle pendant >= ENERGY_MIN_S et intègre le compteur."""
    if meter is None or meter.read() is None:
        return None, None
    n, e0, t0 = 0, meter.read(), time.perf_counter()
    while time.perf_counter() - t0 < ENERGY_MIN_S:
        for b in batches:
            infer(b)
        n += N_IMAGES
    sync()
    dt, de = time.perf_counter() - t0, meter.read() - e0
    if de <= 0:  # compteur RAPL qui a bouclé
        return None, None
    return de / n, de / dt


# ---------------------------------------------------------------- benchmark
def benchmark(pipeline, variant, infer, batches, y, device="cpu", ref=None,
              sync=lambda: None, meter=None):
    """Mesure un moteur d'inférence.

    infer(batch) -> logits (1000 classes). Les batchs sont déjà sur le bon device :
    on mesure l'inférence seule, sans transfert hôte -> GPU ni prétraitement.
    ref : logits de référence (PyTorch eager) pour vérifier que l'optimisation ne
    dégrade pas les prédictions.
    """
    t0 = time.perf_counter()
    infer(batches[0])
    sync()
    first_call_s = time.perf_counter() - t0  # inclut la compilation JIT éventuelle

    for _ in range(N_WARMUP):
        for b in batches:
            infer(b)
    sync()

    runs = []
    for _ in range(N_RUNS):
        lat, outs = [], []
        t_run = time.perf_counter()
        for b in batches:
            t = time.perf_counter()
            outs.append(infer(b))
            sync()
            lat.append(time.perf_counter() - t)
        total = time.perf_counter() - t_run
        e_img, power = _measure_energy(infer, batches, sync, meter)
        runs.append({"total_ms": 1e3 * total, "throughput_ips": N_IMAGES / total,
                     "batch_latency_ms": [1e3 * v for v in lat],
                     "energy_j_per_img": e_img, "power_w": power})

    logits = np.concatenate([_to_numpy(o) for o in outs])
    pred = to_cifar(logits)
    r = {"pipeline": pipeline, "variant": variant, "device": device,
         "first_call_s": first_call_s, "runs": runs,
         "imagenet_top1": logits.argmax(1).tolist(), "cifar_pred": pred.tolist(),
         "accuracy": float((pred == y).mean()),
         "energy_source": meter.source if meter else None, "_logits": logits}
    if ref is not None:
        r["agreement_vs_eager"] = float((logits.argmax(1) == ref.argmax(1)).mean())
        r["max_abs_diff_vs_eager"] = float(np.abs(logits - ref).max())
    s = summarize([r]).iloc[0]
    print(f"{variant:28s} {s.throughput_mean:8.1f} ± {s.throughput_std:6.1f} img/s | "
          f"{s.total_ms_mean:8.1f} ± {s.total_ms_std:6.1f} ms / 50 img | acc {r['accuracy']:.0%}"
          + (f" | accord eager {r['agreement_vs_eager']:.0%}, max|Δlogit| {r['max_abs_diff_vs_eager']:.2e}"
             if ref is not None else ""))
    return r


# ---------------------------------------------------------------- résultats
def summarize(results):
    """Une ligne par variante : moyenne, écart-type, min, max et CV des indicateurs."""
    import pandas as pd
    rows = []
    for r in results:
        row = {"pipeline": r["pipeline"], "variant": r["variant"], "device": r["device"],
               "accuracy": r["accuracy"], "first_call_s": r["first_call_s"],
               "agreement_vs_eager": r.get("agreement_vs_eager", 1.0),
               "max_abs_diff_vs_eager": r.get("max_abs_diff_vs_eager", 0.0)}
        metrics = {
            "throughput": [u["throughput_ips"] for u in r["runs"]],
            "total_ms": [u["total_ms"] for u in r["runs"]],
            "batch_ms": [np.mean(u["batch_latency_ms"]) for u in r["runs"]],
            "energy_j_img": [u["energy_j_per_img"] for u in r["runs"] if u["energy_j_per_img"] is not None],
            "power_w": [u["power_w"] for u in r["runs"] if u["power_w"] is not None],
        }
        for k, v in metrics.items():
            v = np.asarray(v, float)
            ok = v.size > 0
            row[f"{k}_mean"] = v.mean() if ok else np.nan
            row[f"{k}_std"] = v.std(ddof=1) if v.size > 1 else np.nan
            row[f"{k}_min"] = v.min() if ok else np.nan
            row[f"{k}_max"] = v.max() if ok else np.nan
            row[f"{k}_cv_pct"] = 100 * row[f"{k}_std"] / row[f"{k}_mean"] if ok else np.nan
        all_batches = np.concatenate([u["batch_latency_ms"] for u in r["runs"]])
        row["batch_ms_p95"] = np.percentile(all_batches, 95)
        rows.append(row)
    return pd.DataFrame(rows)


# Colonnes affichées dans les tableaux de synthèse
TABLE_COLS = ["pipeline", "variant", "throughput_mean", "throughput_std", "throughput_min", "throughput_max",
              "throughput_cv_pct", "total_ms_mean", "total_ms_std", "total_ms_min", "total_ms_max",
              "batch_ms_mean", "batch_ms_p95", "energy_j_img_mean", "energy_j_img_std", "power_w_mean",
              "accuracy", "agreement_vs_eager", "max_abs_diff_vs_eager", "first_call_s"]


def env_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cpu": platform.processor() or platform.machine(),
            "cpu_threads": torch.get_num_threads(), "os_cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def save_results(name, results, extra=None, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    meta = {"env": env_info(), "protocol": {"n_images": N_IMAGES, "batch_size": BATCH_SIZE,
            "input_size": INPUT_SIZE, "n_warmup": N_WARMUP, "n_runs": N_RUNS,
            "mean": MEAN, "std": STD}, **(extra or {})}
    payload = {"meta": meta, "results": clean}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print("Résultats écrits dans", path)
    # Copie dans les sorties du notebook : avec l'extension Colab de VS Code le fichier reste
    # sur la VM distante, mais le .ipynb enregistré en local garde ces données.
    from IPython.display import display
    display({"application/json": {"tp1_results": name, **json.loads(json.dumps(payload))},
             "text/plain": f"[résultats {name} intégrés au notebook]"}, raw=True)
    return path


def load_results(out_dir="results", nb_dir="notebooks"):
    """Lit results/*.json ; à défaut, extrait les résultats des sorties des notebooks 01 à 04."""
    found = {}
    for fn in sorted(os.listdir(nb_dir)):
        if fn[:2] in ("01", "02", "03", "04") and fn.endswith(".ipynb"):
            with open(os.path.join(nb_dir, fn), encoding="utf-8") as f:
                for cell in json.load(f)["cells"]:
                    for o in cell.get("outputs", []):
                        d = o.get("data", {}).get("application/json")
                        if isinstance(d, dict) and "tp1_results" in d:
                            found[d["tp1_results"]] = d
    if os.path.isdir(out_dir):  # les fichiers JSON sont prioritaires
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(out_dir, fn)) as f:
                    found[fn[:-5]] = json.load(f)
    results, metas = [], {}
    for name in sorted(found):
        results += found[name]["results"]
        metas[name] = found[name]["meta"]
    print("Résultats chargés :", list(metas))
    return results, metas


# ---------------------------------------------------------------- graphiques
def plot_summary(results, title="", path=None):
    """Débit, latence, précision, énergie (barres = moyenne, erreur = écart-type) + boîtes de variabilité."""
    import matplotlib.pyplot as plt
    df = summarize(results)
    labels = [f"{p}\n{v}" for p, v in zip(df.pipeline, df.variant)]
    colors = ["#4C72B0" if d == "cpu" else "#55A868" for d in df.device]
    x = np.arange(len(df))
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    panels = [("throughput", "Débit (images/s)", True), ("total_ms", "Temps d'inférence 50 images (ms)", True),
              ("batch_ms", f"Latence par batch de {BATCH_SIZE} (ms)", True),
              ("energy_j_img", "Énergie par image (J)", False), ("power_w", "Puissance moyenne (W)", False)]
    for a, (k, name, log) in zip(ax.flat, panels):
        a.bar(x, df[f"{k}_mean"].fillna(0), yerr=df[f"{k}_std"].fillna(0), color=colors, capsize=4)
        for i, (m, s) in enumerate(zip(df[f"{k}_mean"], df[f"{k}_std"])):
            a.text(i, 0 if np.isnan(m) else m + np.nan_to_num(s), "n/d" if np.isnan(m) else f"{m:.3g}", ha="center", va="bottom", fontsize=8)
        a.set_title(name)
        if log and (df[f"{k}_mean"].max() / df[f"{k}_mean"].min() > 20):
            a.set_yscale("log")
    a = ax.flat[5]
    a.bar(x, 100 * df.accuracy, color=colors)
    for i, v in enumerate(df.accuracy):
        a.text(i, 100 * v, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    a.set_ylim(0, 100)
    a.set_title("Précision Top-1 CIFAR-10 (%)")
    for a in ax.flat:
        a.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        a.grid(axis="y", alpha=.3)
    fig.suptitle(title + "  (bleu = CPU, vert = GPU ; barres d'erreur = écart-type sur "
                 f"{N_RUNS} exécutions)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()

    # Variabilité : distribution du débit sur les exécutions, normalisée par la médiane
    fig, a = plt.subplots(figsize=(max(8, 1.2 * len(df)), 4))
    data = [np.array([u["throughput_ips"] for u in r["runs"]]) for r in results]
    a.boxplot([d / np.median(d) for d in data], tick_labels=labels)
    a.axhline(1, color="grey", lw=.8)
    a.set_ylabel("Débit / médiane")
    a.set_title("Variabilité du débit entre exécutions")
    a.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path.replace(".png", "_variabilite.png"), dpi=130, bbox_inches="tight")
    plt.show()
    return df
