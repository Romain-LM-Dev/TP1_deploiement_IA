"""Génère rapport/rapport.pdf et ses figures à partir des résultats des notebooks 01 à 04.

Usage (depuis la racine du dépôt) : python rapport/build_report.py
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, os.getcwd())
import common as C  # noqa: E402

OUT = "rapport"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ données
results, metas = C.load_results()
df = C.summarize(results)
df["speedup"] = df.throughput_mean / df.groupby("pipeline").throughput_mean.transform("first")
SHORT_V = {"PyTorch eager (réf.)": "eager", "OpenVINO FP32": "OpenVINO", "TorchScript (freeze)": "TorchScript",
           "torch.compile (inductor)": "torch.compile", "TensorRT FP32": "TRT FP32", "TensorRT FP16": "TRT FP16"}
df["label"] = [f"{p.split()[0]}\n{SHORT_V.get(v, v).replace('torch.compile', 'compile')}"
               for p, v in zip(df.pipeline, df.variant)]
df["is_ref"] = df.variant.str.contains("eager")
row = {(p.split()[0], SHORT_V[v]): r for (p, v), r in zip(zip(df.pipeline, df.variant), df.itertuples())}
compile_s = {(r["pipeline"].split()[0], SHORT_V[r["variant"]]): r.get("compile_s") or r["first_call_s"]
             for r in results}
y = np.load("data/cifar10_test_50.npz")["labels"]
pred = np.array(results[0]["cifar_pred"])

# ------------------------------------------------------------------ figures
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
CPU, GPU, REF, INK = "#2a78d6", "#eb6834", "#b9b8b1", "#0b0b0b"
bar_colors = [REF if ref else (CPU if d == "cpu" else GPU) for ref, d in zip(df.is_ref, df.device)]


def legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=REF, label="PyTorch eager (référence)"), Patch(color=CPU, label="optimisé CPU"),
                       Patch(color=GPU, label="optimisé GPU")], frameon=False, fontsize=7, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3)


def bars(metric, ylabel, name, log=True):
    fig, ax = plt.subplots(figsize=(7, 2.9))
    x = np.arange(len(df))
    ax.bar(x, df[f"{metric}_mean"], yerr=df[f"{metric}_std"], color=bar_colors, width=.7,
           edgecolor="white", linewidth=1, capsize=3, error_kw={"elinewidth": .8, "ecolor": "#52514e"})
    for i, (m, s) in enumerate(zip(df[f"{metric}_mean"], df[f"{metric}_std"])):
        ax.annotate(f"{m:.0f}" if m >= 100 else f"{m:.1f}", (i, m + s), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x, df.label, fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)
    ax.axvline(4.5, color="#8a8984", lw=.6, ls=":")
    legend(ax)
    fig.tight_layout()
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


f_thr = bars("throughput", "images / s (échelle log)", "debit.png")
f_lat = bars("total_ms", "ms pour 50 images (échelle log)", "latence.png")

# accélération
fig, ax = plt.subplots(figsize=(7, 2.6))
opt = df[~df.is_ref]
ax.bar(np.arange(len(opt)), opt.speedup, color=[CPU if d == "cpu" else GPU for d in opt.device], width=.6,
       edgecolor="white")
ax.axhline(1, color="#52514e", lw=.8)
for i, v in enumerate(opt.speedup):
    ax.annotate(f"×{v:.2f}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
ax.set_xticks(np.arange(len(opt)), opt.label, fontsize=7)
ax.set_ylabel("débit / débit eager")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
f_speed = os.path.join(FIG, "acceleration.png")
fig.savefig(f_speed, dpi=200)
plt.close(fig)

# variabilité
fig, ax = plt.subplots(figsize=(7, 2.8))
data = [np.array([u["throughput_ips"] for u in r["runs"]]) for r in results]
bp = ax.boxplot([d / np.median(d) for d in data], tick_labels=list(df.label), patch_artist=True, widths=.55,
                medianprops={"color": INK}, flierprops={"markersize": 3})
for patch, c in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(c)
    patch.set_edgecolor("#52514e")
top = max((d / np.median(d)).max() for d in data) + .1
ax.set_ylim(top=top)
for i, cv in enumerate(df.throughput_cv_pct):
    ax.annotate(f"CV\n{cv:.1f} %", (i + 1, top), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=6)
ax.axhline(1, color="#8a8984", lw=.6)
ax.set_ylabel("débit / médiane (10 exécutions)")
ax.tick_params(axis="x", labelsize=6.5)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
f_var = os.path.join(FIG, "variabilite.png")
fig.savefig(f_var, dpi=200)
plt.close(fig)

# énergie (GPU uniquement)
g = df[df.device == "cuda"]
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.barh(np.arange(len(g)), 1e3 * g.energy_j_img_mean, xerr=1e3 * g.energy_j_img_std,
        color=[REF if r else GPU for r in g.is_ref], height=.6, edgecolor="white", capsize=2)
for i, (e, es, p) in enumerate(zip(g.energy_j_img_mean, g.energy_j_img_std, g.power_w_mean)):
    ax.annotate(f"{1e3 * e:.1f} mJ  ({p:.0f} W)", (1e3 * (e + es), i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7)
ax.set_yticks(np.arange(len(g)), [lbl.replace("\n", " · ") for lbl in g.label])
ax.invert_yaxis()
ax.set_xlabel("énergie GPU par image (mJ), puissance moyenne entre parenthèses")
ax.set_xlim(0, 1e3 * g.energy_j_img_mean.max() * 1.35)
ax.grid(axis="x", alpha=.25)
fig.tight_layout()
f_energy = os.path.join(FIG, "energie.png")
fig.savefig(f_energy, dpi=200)
plt.close(fig)

# précision : par variante + par classe
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
a1.bar(np.arange(len(df)), 100 * df.accuracy, color=bar_colors, width=.7, edgecolor="white")
a1.set_xticks(np.arange(len(df)), [lbl.replace("\n", " ") for lbl in df.label], fontsize=6, rotation=90)
a1.set_ylim(0, 100)
a1.set_ylabel("Top-1 CIFAR-10 (%)")
a1.set_title("Précision par variante")
a1.annotate(f"{100 * df.accuracy.iloc[0]:.0f} % pour les 11 variantes", (5, 100 * df.accuracy.iloc[0]),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
correct = np.bincount(y, weights=pred == y, minlength=10)
total = np.bincount(y, minlength=10)
a2.barh(np.arange(10), total, color="#e4e3dd", height=.65)
a2.barh(np.arange(10), correct, color="#2a78d6", height=.65)
for i, (c, t) in enumerate(zip(correct, total)):
    a2.annotate(f"{c:.0f}/{t}", (t, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5)
a2.set_yticks(np.arange(10), C.CIFAR_CLASSES, fontsize=7)
a2.invert_yaxis()
a2.set_xlim(0, total.max() + 2)
a2.set_title("Bonnes prédictions par classe")
fig.tight_layout()
f_acc = os.path.join(FIG, "precision.png")
fig.savefig(f_acc, dpi=200)
plt.close(fig)

# écart numérique
fig, ax = plt.subplots(figsize=(7, 2.3))
ax.barh(np.arange(len(opt)), opt.max_abs_diff_vs_eager,
        color=["#e34948" if "FP16" in v else (CPU if d == "cpu" else GPU) for v, d in zip(opt.variant, opt.device)],
        height=.6)
for i, v in enumerate(opt.max_abs_diff_vs_eager):
    ax.annotate(f"{v:.1e}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
ax.set_xscale("log")
ax.set_yticks(np.arange(len(opt)), [lbl.replace("\n", " · ") for lbl in opt.label])
ax.invert_yaxis()
ax.set_xlabel("écart maximal des logits par rapport à PyTorch eager (échelle log)")
ax.grid(axis="x", alpha=.25)
fig.tight_layout()
f_diff = os.path.join(FIG, "ecart_logits.png")
fig.savefig(f_diff, dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ PDF
for style, file in [("", "DejaVuSans"), ("-Bold", "DejaVuSans-Bold"), ("-Oblique", "DejaVuSans-Oblique"),
                    ("-BoldOblique", "DejaVuSans-BoldOblique")]:
    pdfmetrics.registerFont(TTFont(f"DV{style}", os.path.join(FONT_DIR, f"{file}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV-Oblique", boldItalic="DV-BoldOblique")
pdfmetrics.registerFont(TTFont("DVMono", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))

ss = getSampleStyleSheet()
body = ParagraphStyle("body", ss["Normal"], fontName="DV", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                      spaceAfter=5)
bullet = ParagraphStyle("bullet", body, leftIndent=12, bulletIndent=2, spaceAfter=2)
h1 = ParagraphStyle("h1", body, fontName="DV-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1d3d6e"), alignment=0, keepWithNext=1)
h2 = ParagraphStyle("h2", h1, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
caption = ParagraphStyle("cap", body, fontName="DV-Oblique", fontSize=8, leading=10.5, alignment=TA_CENTER,
                         textColor=colors.HexColor("#52514e"), spaceAfter=8)
cell = ParagraphStyle("cell", body, fontSize=7.5, leading=9.5, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", cell, fontName="DV-Bold")
title = ParagraphStyle("title", h1, fontSize=20, leading=26, alignment=TA_CENTER, spaceBefore=0)
sub = ParagraphStyle("sub", body, alignment=TA_CENTER, fontSize=11, leading=15)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))  # noqa: E731
B = lambda t: story.append(Paragraph(t, bullet, bulletText="•"))  # noqa: E731


def fig(path, cap, w=17):
    img = Image(path)
    img.drawWidth, img.drawHeight = w * cm, w * cm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(cap, caption)]))


def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#c3c2b7")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")))
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f5")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


v = lambda p, k: row[p]  # noqa: E731
r_ov, r_cpu1 = row[("CPU-1", "OpenVINO")], row[("CPU-1", "eager")]
r_ts, r_tc, r_cpu2 = row[("CPU-2", "TorchScript")], row[("CPU-2", "torch.compile")], row[("CPU-2", "eager")]
r_t32, r_t16, r_gpu1 = row[("GPU-1", "TRT FP32")], row[("GPU-1", "TRT FP16")], row[("GPU-1", "eager")]
r_gts, r_gtc, r_gpu2 = row[("GPU-2", "TorchScript")], row[("GPU-2", "torch.compile")], row[("GPU-2", "eager")]
env_cpu, env_gpu = metas["01_cpu_openvino"]["env"], metas["03_gpu_tensorrt"]["env"]
gpu_cpu_ratio = r_gpu2.throughput_mean / r_cpu2.throughput_mean
best_cpu = max([r_ov, r_ts, r_tc], key=lambda r: r.throughput_mean)

# ---- page de titre
story += [Spacer(1, 5 * cm)]
P("Déploiement d'un modèle d'IA :<br/>comparaison de quatre pipelines d'inférence CPU et GPU", title)
story.append(Spacer(1, .6 * cm))
P("TP1 — Déploiement IA — EIGSI", sub)
P("ResNet-18 pré-entraîné sur ImageNet · 50 images CIFAR-10 · Google Colab", sub)
story.append(Spacer(1, 1.5 * cm))
P("Dépôt : https://github.com/Romain-LM-Dev/TP1_deploiement_IA", sub)
story.append(Spacer(1, 3 * cm))
table([["Pipeline", "Chaîne", "Débit moyen (img/s)", "Accélération vs eager"],
       ["CPU n°1", "PyTorch → ONNX → OpenVINO → oneDNN", f"{r_ov.throughput_mean:.1f}", f"×{r_ov.speedup:.2f}"],
       ["CPU n°2", "PyTorch → TorchScript", f"{r_ts.throughput_mean:.1f}", f"×{r_ts.speedup:.2f}"],
       ["", "PyTorch → torch.compile (Inductor)", f"{r_tc.throughput_mean:.1f}", f"×{r_tc.speedup:.2f}"],
       ["GPU n°1", "PyTorch → ONNX → TensorRT FP32 → CUDA/cuDNN", f"{r_t32.throughput_mean:.0f}",
        f"×{r_t32.speedup:.2f}"],
       ["", "PyTorch → ONNX → TensorRT FP16", f"{r_t16.throughput_mean:.0f}", f"×{r_t16.speedup:.2f}"],
       ["GPU n°2", "PyTorch → TorchScript → CUDA", f"{r_gts.throughput_mean:.0f}", f"×{r_gts.speedup:.2f}"],
       ["", "PyTorch → torch.compile (Inductor) → CUDA", f"{r_gtc.throughput_mean:.0f}", f"×{r_gtc.speedup:.2f}"]],
      [2, 8.2, 3.2, 3.6])
P("<i>Synthèse : débit moyen sur 10 exécutions et accélération par rapport à PyTorch eager mesuré dans la même "
  "session. Précision Top-1 identique (64 %) pour toutes les variantes.</i>", caption)
story.append(PageBreak())

# ---- 1. introduction
P("1. Objectif", h1)
P("Un même modèle peut être exécuté par des chaînes logicielles très différentes : le framework d'entraînement "
  "lui-même (PyTorch), un compilateur intégré au framework (TorchScript, torch.compile) ou un moteur d'inférence "
  "spécialisé alimenté par un format d'échange (ONNX vers OpenVINO ou TensorRT). Ce TP compare quatre de ces "
  "chaînes, deux sur CPU et deux sur GPU, sur le même modèle et les mêmes données. Il évalue le débit, le "
  "temps d'inférence, la précision, la cohérence des prédictions, l'énergie et la stabilité des mesures.")

P("2. Les quatre pipelines", h1)
table([["Niveau", "CPU n°1", "CPU n°2", "GPU n°1", "GPU n°2"],
       ["Haut niveau", "PyTorch", "PyTorch", "PyTorch", "PyTorch"],
       ["Format", "ONNX (opset 17)", "TorchScript ; graphe FX (TorchDynamo)", "ONNX (opset 17), FP32 et FP16",
        "TorchScript ; graphe FX (TorchDynamo)"],
       ["Middle level", "OpenVINO Runtime : fusions, choix des kernels", "JIT freeze + optimize_for_inference ; "
        "TorchInductor", "TensorRT : fusion de couches, sélection de tactiques", "JIT freeze ; TorchInductor "
        "(kernels Triton)"],
       ["Bas niveau", "oneDNN (brgconv AVX2), MLAS", "oneDNN (MKLDNN), C++/OpenMP généré", "cuDNN, kernels xmma "
        "(Tensor Cores en FP16)", "cuDNN, Triton"],
       ["Matériel", "CPU Colab", "CPU Colab", "GPU T4 Colab", "GPU T4 Colab"]],
      [2.4, 3.6, 3.7, 3.7, 3.6])
P("<b>CPU n°1 — OpenVINO.</b> L'export ONNX produit 49 nœuds : les BatchNorm sont déjà repliées dans les "
  "convolutions par le repliement de constantes de l'exporteur. OpenVINO importe ce graphe (305 opérations après "
  "décomposition), puis la compilation le réduit à 112 opérations exécutées. Les 20 convolutions sont fusionnées "
  "avec leur ReLU et leur addition résiduelle, et exécutées par des kernels oneDNN <font face='DVMono'>brgconv_avx2"
  "</font> (multiplications de matrices par blocs vectorisées AVX2). La compilation ne prend que "
  f"{compile_s[('CPU-1', 'OpenVINO')]:.2f} s.")
P("<b>CPU n°2 — TorchScript et torch.compile.</b> TorchScript trace le modèle, gèle les poids et applique "
  "<font face='DVMono'>optimize_for_inference</font> : les 20 <font face='DVMono'>batch_norm</font> et 20 "
  "convolutions du graphe tracé disparaissent au profit de convolutions MKLDNN pré-empaquetées, avec les BN repliées. "
  "torch.compile capture le modèle en un seul graphe (aucune rupture, 69 opérations). Inductor fusionne ensuite "
  "les opérations élémentaires en boucles C++/OpenMP et délègue les convolutions à oneDNN.")
P("<b>GPU n°1 — TensorRT.</b> Les 49 nœuds ONNX deviennent 25 couches dans le moteur FP32. Par exemple, "
  "<font face='DVMono'>conv2 + Add + ReLU</font> d'un bloc résiduel forment un seul kernel. Pour chaque couche, "
  "TensorRT chronomètre plusieurs implémentations et garde la plus rapide : Winograd cuDNN en FP32, kernels "
  "<font face='DVMono'>sm75_xmma_…_tensor16x8x8</font> sur Tensor Cores en FP16. TensorRT 11 a supprimé le drapeau "
  "FP16 : la précision est désormais portée par le graphe ONNX lui-même (réseau « strongly typed »). La variante FP16 "
  "est donc un ONNX exporté avec des poids FP16, avec des entrées/sorties FP32 pour garder le même code d'exécution.")
P("<b>GPU n°2 — TorchScript et torch.compile.</b> Sur GPU, le gel TorchScript fusionne les convolutions en "
  "<font face='DVMono'>cudnn_convolution_relu</font> (×9) et <font face='DVMono'>cudnn_convolution_add_relu</font> "
  "(×8). torch.compile génère des kernels Triton pour les opérations élémentaires et appelle cuDNN pour les "
  "convolutions.")

# ---- 3. protocole
P("3. Protocole expérimental", h1)
P("Pour une comparaison équitable, tout le protocole est défini une seule fois dans <font face='DVMono'>common.py"
  "</font> et importé par les quatre notebooks :")
table([["Élément", "Choix"],
       ["Modèle", "torchvision ResNet-18, poids IMAGENET1K_V1, mode eval"],
       ["Données", "50 premières images du jeu de test CIFAR-10, versionnées dans le dépôt (mêmes pixels partout) ; "
        "répartition : " + ", ".join(f"{c} {n}" for c, n in zip(C.CIFAR_CLASSES, np.bincount(y, minlength=10)))],
       ["Prétraitement", "32×32 → 224×224 (bilinéaire), normalisation ImageNet mean (0,485 ; 0,456 ; 0,406), std "
        "(0,229 ; 0,224 ; 0,225). Pas de recadrage central, qui couperait l'objet sur une image 32×32"],
       ["10 classes", "Les 1000 classes ImageNet sont regroupées en 10 (ex. 118 races de chiens → dog, 59 espèces "
        "d'oiseaux → bird). La classe prédite est celle dont le meilleur logit est le plus élevé. « deer » n'existe pas "
        "dans ImageNet : les antilopes (ibex, bubale, impala, gazelle) servent d'approximation"],
       ["Batch", f"{C.BATCH_SIZE} images (5 batchs), batch fixe dans les exports ONNX"],
       ["Répétitions", f"1 premier appel chronométré à part (compilation), {C.N_WARMUP} passes de chauffe, puis "
        f"{C.N_RUNS} exécutions mesurées des 50 images"],
       ["Chronométrage", "time.perf_counter ; données déjà sur le périphérique (pas de transfert hôte → GPU) ; "
        "torch.cuda.synchronize() après chaque batch sur GPU"],
       ["Statistiques", "moyenne, écart-type (n−1), minimum, maximum, coefficient de variation (CV = σ / moyenne)"],
       ["Énergie", f"compteur matériel cumulé NVML (GPU), intégré sur ≥ {C.ENERGY_MIN_S:.0f} s d'inférence en boucle "
        "après chaque exécution. RAPL (CPU) n'est pas exposé sur les VM Colab"],
       ["Référence", "PyTorch eager mesuré dans chaque notebook, sur la même machine et dans la même session"]],
      [2.6, 14.4])
P("Environnement Colab mesuré :", body)
table([["", "Notebooks CPU (01, 02)", "Notebooks GPU (03, 04)"],
       ["Processeur", f"{env_cpu['cpu']}, {env_cpu['os_cpu_count']} vCPU, {env_cpu['cpu_threads']} thread PyTorch",
        f"{env_gpu['cpu']}, {env_gpu['os_cpu_count']} vCPU"],
       ["Accélérateur", "—", f"NVIDIA {env_gpu['gpu']} (16 Go, 70 W), CUDA {env_gpu['cuda']}, cuDNN "
        f"{env_gpu['cudnn']}"],
       ["Logiciels", f"Python {env_cpu['python']}, PyTorch {env_cpu['torch']}, OpenVINO "
        f"{metas['01_cpu_openvino']['openvino'].split('-')[0]}",
        f"Python {env_gpu['python']}, PyTorch {env_gpu['torch']}, TensorRT {metas['03_gpu_tensorrt']['tensorrt']}"]],
      [2.6, 7.2, 7.2])
P("Chaque notebook tourne dans sa propre session Colab, donc potentiellement sur une VM différente. Les deux "
  f"références eager CPU diffèrent de {100 * abs(1 - r_cpu1.throughput_mean / r_cpu2.throughput_mean):.0f} % "
  f"({r_cpu1.throughput_mean:.1f} contre {r_cpu2.throughput_mean:.1f} img/s) pour un code strictement identique. "
  "C'est pourquoi les gains sont toujours calculés par rapport à la référence de la même session.")
story.append(PageBreak())

# ---- 4. résultats
P("4. Résultats", h1)
P("4.1 Débit et temps d'inférence", h2)
fig(f_thr, "Figure 1 — Débit (images/s) des 11 variantes, moyenne ± écart-type sur 10 exécutions, échelle "
    "logarithmique. Gris : référence eager de chaque notebook.")
fig(f_lat, "Figure 2 — Temps d'inférence des 50 images (ms), moyenne ± écart-type, échelle logarithmique.")
rows = [["Variante", "Débit (img/s)<br/>moy. ± σ", "min – max", "CV", "Temps 50 img (ms)<br/>moy. ± σ",
         "min – max", "Latence batch p95 (ms)"]]
for r in df.itertuples():
    rows.append([f"{r.pipeline.split()[0]} · {SHORT_V[r.variant]}",
                 f"{r.throughput_mean:.1f} ± {r.throughput_std:.1f}", f"{r.throughput_min:.1f} – {r.throughput_max:.1f}",
                 f"{r.throughput_cv_pct:.1f} %", f"{r.total_ms_mean:.1f} ± {r.total_ms_std:.1f}",
                 f"{r.total_ms_min:.1f} – {r.total_ms_max:.1f}", f"{r.batch_ms_p95:.1f}"])
table(rows, [3.4, 2.6, 2.6, 1.4, 2.8, 2.6, 1.9])
P(f"<b>L'écart CPU/GPU domine tout le reste.</b> À code égal (eager), le T4 traite {gpu_cpu_ratio:.0f} fois plus "
  f"d'images par seconde que le CPU Colab ({r_gpu2.throughput_mean:.0f} contre {r_cpu2.throughput_mean:.1f} "
  f"img/s) : 50 images prennent environ {r_gpu2.total_ms_mean:.0f} ms au lieu de "
  f"{r_cpu2.total_ms_mean / 1000:.1f} s. ResNet-18 demande environ 1,8 milliard de multiplications-additions par image, un calcul très parallèle "
  "que les 2560 cœurs CUDA du T4 absorbent. Le CPU, lui, n'expose qu'un seul thread de calcul à PyTorch. "
  f"Même la meilleure variante CPU ({SHORT_V[best_cpu.variant]}, {best_cpu.throughput_mean:.1f} img/s) reste "
  f"{r_gpu1.throughput_mean / best_cpu.throughput_mean:.0f} fois plus lente que le GPU sans optimisation.")

P("4.2 Gain apporté par chaque optimisation", h2)
fig(f_speed, "Figure 3 — Accélération par rapport à PyTorch eager mesuré dans la même session (débit / débit "
    "eager). Ce rapport neutralise les écarts de matériel entre sessions Colab.")
P(f"<b>Sur CPU</b>, les trois optimisations apportent un gain modeste et du même ordre : OpenVINO ×{r_ov.speedup:.2f}, "
  f"torch.compile ×{r_tc.speedup:.2f}, TorchScript ×{r_ts.speedup:.2f}. Les trois appliquent les mêmes "
  "transformations de fond : repliement des BatchNorm, fusion convolution + ReLU (+ addition résiduelle) et "
  "kernels oneDNN choisis pour le jeu d'instructions (AVX2 ici). Avec un seul cœur, le temps est dominé par les "
  "convolutions elles-mêmes, que ni la fusion ni la génération de code n'accélèrent beaucoup. Les intervalles "
  "min–max des trois variantes se recouvrent largement : vu la variabilité (§ 4.3), on ne peut pas les départager "
  "de façon fiable. Elles sont en revanche toutes nettement au-dessus de la référence eager.")
P(f"<b>Sur GPU</b>, les écarts sont nets et très reproductibles. TensorRT FP32 gagne ×{r_t32.speedup:.2f} : la fusion "
  "réduit le nombre de kernels lancés et d'allers-retours en mémoire. Le temps par batch est court (≈ "
  f"{r_gpu1.batch_ms_mean:.0f} ms en eager), donc ces surcoûts pèsent lourd. TensorRT FP16 atteint "
  f"×{r_t16.speedup:.2f} ({r_t16.throughput_mean:.0f} img/s, {r_t16.total_ms_mean:.1f} ms pour 50 images). En FP32 "
  "le T4 n'utilise que ses cœurs CUDA (≈ 8 TFLOPS) ; le FP16 active les Tensor Cores (≈ 65 TFLOPS) et divise par "
  f"deux le trafic mémoire. TorchScript gagne ×{r_gts.speedup:.2f} grâce aux convolutions cuDNN fusionnées. "
  f"<b>torch.compile est en revanche plus lent qu'eager</b> (×{r_gtc.speedup:.2f}, {r_gtc.batch_ms_mean:.1f} ms par "
  f"batch contre {r_gpu2.batch_ms_mean:.1f}). Notre hypothèse, non vérifiée par un profilage : avec "
  "<font face='DVMono'>freezing=True</font>, Inductor convertit les convolutions au format mémoire "
  "<i>channels-last</i>, et les noyaux cuDNN FP32 du T4 (architecture Turing, sans TF32) y sont moins efficaces. "
  "Les conversions de format ajoutent aussi des kernels. Le mode par défaut n'active ni CUDA Graphs ni l'auto-réglage "
  "(<font face='DVMono'>max-autotune</font>, d'ailleurs bridé sur le T4 : « Not enough SMs »). Ce résultat montre "
  "qu'une compilation n'est pas un gain garanti : il faut toujours la mesurer face à la référence.")
story.append(PageBreak())

P("4.3 Stabilité et reproductibilité des mesures", h2)
fig(f_var, "Figure 4 — Distribution du débit sur les 10 exécutions, normalisé par la médiane de chaque variante, "
    "et coefficient de variation (CV).")
cv_cpu = df[df.device == "cpu"].throughput_cv_pct
cv_gpu = df[(df.device == "cuda") & ~df.variant.str.contains("FP16")].throughput_cv_pct
norm = [np.array([u["throughput_ips"] for u in r["runs"]]) for r in results]
norm = [t / np.median(t) for t in norm]
worst_cpu = min(t.min() for t, d in zip(norm, df.device) if d == "cpu")
first_t16 = norm[df.index[(df.pipeline.str.startswith("GPU-1")) & df.variant.str.contains("FP16")][0]][0]
rest_t16 = norm[df.index[(df.pipeline.str.startswith("GPU-1")) & df.variant.str.contains("FP16")][0]][1:]
first_gpu1 = norm[df.index[(df.pipeline.str.startswith("GPU-1")) & df.is_ref][0]][0]
P(f"<b>Les mesures GPU sont stables</b> : CV de {cv_gpu.min():.1f} à {cv_gpu.max():.1f} % (hors FP16), et des écarts "
  "entre variantes bien supérieurs à leur dispersion, donc des conclusions reproductibles. <b>Les mesures CPU le "
  f"sont beaucoup moins</b> : CV de {cv_cpu.min():.1f} à {cv_cpu.max():.1f} %, avec des exécutions isolées jusqu'à "
  f"{100 * (1 - worst_cpu):.0f} % plus lentes que la médiane. La VM Colab partage en général son cœur physique (2 vCPU = 1 cœur hyperthreadé) avec d'autres "
  "processus : la moindre activité concurrente (Jupyter, le système, d'autres locataires de l'hôte) se répercute "
  "directement sur le temps. L'écart d'une session à l'autre (§ 3, "
  f"{100 * abs(1 - r_cpu1.throughput_mean / r_cpu2.throughput_mean):.0f} % entre les deux références eager) est "
  "d'ailleurs du même ordre que les gains mesurés.")
P(f"TensorRT FP16 a un CV de {r_t16.throughput_cv_pct:.1f} % alors que les 9 autres exécutions restent entre "
  f"{rest_t16.min():.2f} et {rest_t16.max():.2f} × la médiane : c'est la <b>première</b> exécution mesurée qui est lente ({r_t16.throughput_min:.0f} img/s, soit "
  f"{first_t16:.2f} × la médiane). La référence eager du même notebook présente le même défaut ({first_gpu1:.2f} × la "
  "médiane à la première exécution). C'est la trace d'une chauffe insuffisante : 3 passes de 50 images ne durent que "
  "quelques dizaines de millisecondes sur GPU, trop peu pour que le T4 quitte son état de repos (P8, 9 W au "
  "lancement) et atteigne sa fréquence maximale. Avec une passe de 12 ms, ce démarrage pèse lourd en relatif. "
  "Une chauffe calibrée en durée (≈ 1 s) plutôt qu'en nombre de passes corrigerait ce biais. En attendant, "
  "la médiane est plus représentative que la moyenne pour cette variante.")
P("<b>Conséquence méthodologique</b> : trois exécutions (le minimum demandé) n'auraient pas suffi sur CPU. "
  "Avec un CV de 10 %, l'erreur-type de la moyenne de 3 mesures vaut ≈ 6 %, du même ordre que l'écart entre "
  "OpenVINO, TorchScript et torch.compile. Nos 10 exécutions la ramènent à ≈ 3 %, et la référence eager mesurée "
  "dans la même session supprime l'effet machine.")

P("4.4 Qualité des prédictions", h2)
fig(f_acc, "Figure 5 — Précision Top-1 CIFAR-10 de chaque variante (gauche) et détail par classe pour la référence "
    "(droite ; la barre grise est le nombre d'images de la classe).")
P(f"Les 11 variantes obtiennent exactement la même précision, {100 * df.accuracy.iloc[0]:.0f} % "
  f"({int((pred == y).sum())}/50). Ce niveau s'explique par le modèle et non par les optimisations : le réseau "
  "ImageNet n'a jamais vu d'images CIFAR, qui sont des images 32×32 agrandies 7 fois, donc floues. Souvent, sa "
  "classe ImageNet préférée n'appartient à aucun groupe (« sweatshirt », « safety pin », « marmoset »). "
  f"L'analyse des {int((pred != y).sum())} erreurs montre deux biais du regroupement de classes. "
  f"{int(((pred != y) & (pred == 5)).sum())} erreurs sont prédites <i>dog</i>. Ce groupe compte 118 classes "
  "ImageNet, et le maximum de 118 logits est mécaniquement plus élevé que celui des groupes de 2 à 5 classes "
  "(<i>horse</i>, <i>cat</i>, <i>deer</i>). Ensuite, les 3 voitures sont reconnues comme « moving van », classé "
  "<i>truck</i>. Les classes bien couvertes sont parfaitement reconnues : <i>dog</i> 7/7, <i>truck</i> 8/8, "
  "<i>ship</i> 4/4. Un réentraînement de la dernière couche sur CIFAR-10 relèverait fortement ce score, mais "
  "sortirait du cadre « modèle pré-entraîné » commun aux groupes.")
fig(f_diff, "Figure 6 — Écart maximal des 1000 logits par rapport à PyTorch eager (même session), sur les 50 images. "
    "Rouge : FP16.")
P("La comparaison des prédictions confirme qu'<b>aucune optimisation ne dégrade les résultats</b>. La classe "
  "ImageNet prédite est identique pour les 50 images dans les 11 variantes (accord 100 % deux à deux, matrice "
  "dans le notebook 05), et donc aussi la classe CIFAR. Toutes les variantes FP32 restent à moins de "
  f"{df[~df.variant.str.contains('FP16')].max_abs_diff_vs_eager.max():.0e} des logits eager : c'est la différence "
  "attendue quand on change l'ordre des sommes flottantes (fusion, Winograd, autres kernels). TensorRT FP16 s'écarte "
  f"de {r_t16.max_abs_diff_vs_eager:.2f}, environ 2000 fois plus, du fait de la mantisse 10 bits du demi-flottant. "
  "Cet écart reste très inférieur aux marges entre la première et la deuxième classe, d'où aucun changement de "
  "prédiction. Sur un jeu plus grand, le FP16 pourrait toutefois inverser quelques cas limites : à vérifier avant "
  "un déploiement.")
story.append(PageBreak())

P("4.5 Consommation énergétique", h2)
fig(f_energy, "Figure 7 — Énergie consommée par le GPU par image (mJ, moyenne ± écart-type) et puissance moyenne, "
    "mesurées avec le compteur NVML.")
P(f"Pendant l'inférence en boucle, le T4 tourne à pleine charge dans toutes les variantes : puissance moyenne de "
  f"{df[df.device == 'cuda'].power_w_mean.min():.0f} à {df[df.device == 'cuda'].power_w_mean.max():.0f} W pour une "
  "limite de 70 W. La puissance étant quasi constante, l'énergie par image est simplement inversement "
  f"proportionnelle au débit. TensorRT FP32 consomme {1e3 * r_t32.energy_j_img_mean:.0f} mJ par image contre "
  f"{1e3 * r_gpu1.energy_j_img_mean:.0f} en eager "
  f"(−{100 * (1 - r_t32.energy_j_img_mean / r_gpu1.energy_j_img_mean):.0f} %), et TensorRT FP16 "
  f"{1e3 * r_t16.energy_j_img_mean:.0f} mJ (−{100 * (1 - r_t16.energy_j_img_mean / r_gpu1.energy_j_img_mean):.0f} %). "
  f"À l'inverse, torch.compile en consomme {100 * (r_gtc.energy_j_img_mean / r_gpu2.energy_j_img_mean - 1):.0f} % "
  "de plus. Optimiser le temps d'inférence revient donc ici directement à économiser l'énergie.")
P("Limites de la mesure : NVML ne compte que la carte graphique (ni le CPU hôte, ni la mémoire, ni le refroidissement). "
  "Sur CPU, les VM Colab n'exposent pas les compteurs RAPL (<font face='DVMono'>/sys/class/powercap</font> absent), "
  "si bien qu'aucune mesure n'est disponible. Une estimation à partir du TDP n'aurait mesuré que des hypothèses ; "
  "nous ne l'avons pas faite.")

P("4.6 Coût de préparation", h2)
table([["Variante", "Préparation (s)", "Nature"],
       ["CPU · OpenVINO", f"{compile_s[('CPU-1', 'OpenVINO')]:.2f}", "lecture ONNX + compilation"],
       ["CPU · TorchScript", f"{compile_s[('CPU-2', 'TorchScript')]:.2f}", "trace + freeze + optimize_for_inference"],
       ["CPU · torch.compile", f"{compile_s[('CPU-2', 'torch.compile')]:.1f}", "premier appel (capture + génération C++)"],
       ["GPU · TensorRT FP32", f"{compile_s[('GPU-1', 'TRT FP32')]:.1f}", "construction du moteur (sélection de tactiques)"],
       ["GPU · TensorRT FP16", f"{compile_s[('GPU-1', 'TRT FP16')]:.1f}", "idem, davantage de tactiques candidates"],
       ["GPU · TorchScript", f"{compile_s[('GPU-2', 'TorchScript')]:.2f}", "trace + freeze"],
       ["GPU · torch.compile", f"{compile_s[('GPU-2', 'torch.compile')]:.1f}", "premier appel (capture + kernels Triton)"]],
      [4, 3, 10])
P("Les moteurs TensorRT et les kernels torch.compile coûtent 6 à 25 s de préparation. C'est négligeable pour un "
  "service qui tourne en continu, où le moteur TensorRT est sérialisé une fois puis rechargé. C'est en revanche "
  "prohibitif pour un traitement ponctuel de 50 images : sur CPU, la compilation de torch.compile dure autant que "
  f"{compile_s[('CPU-2', 'torch.compile')] / (r_tc.total_ms_mean / 1000):.0f} traitements complets des 50 images.")

# ---- 5. conclusion
P("5. Conclusion", h1)
B(f"<b>Le choix du matériel prime</b> : à code égal, le GPU T4 est ≈ {gpu_cpu_ratio:.0f} fois plus rapide que le "
  "CPU Colab, bien plus que ce qu'apporte n'importe quelle optimisation logicielle sur CPU.")
B(f"<b>TensorRT est la chaîne la plus performante</b> : ×{r_t32.speedup:.1f} en FP32 sans aucun écart de "
  f"prédiction, et ×{r_t16.speedup:.1f} en FP16 grâce aux Tensor Cores, pour "
  f"{100 * (1 - r_t16.energy_j_img_mean / r_gpu1.energy_j_img_mean):.0f} % d'énergie en moins par image.")
B(f"<b>Sur CPU, OpenVINO, TorchScript et torch.compile gagnent tous ×{min(r_ov.speedup, r_ts.speedup, r_tc.speedup):.2f} "
  f"à ×{max(r_ov.speedup, r_ts.speedup, r_tc.speedup):.2f}</b>, des écarts entre eux noyés dans la variabilité "
  "d'une VM partagée. OpenVINO a l'avantage d'une compilation quasi instantanée.")
B("<b>Une compilation n'est pas un gain garanti</b> : torch.compile est plus lent qu'eager sur le T4 en FP32. Il faut "
  "toujours mesurer chaque variante face à une référence dans les mêmes conditions.")
B("<b>Aucune optimisation n'a modifié une prédiction</b> (précision 64 % partout, accord 100 %). L'écart numérique "
  "reste de l'ordre de 10<super>-5</super> en FP32 et 10<super>-2</super> en FP16.")
B(f"<b>Méthode</b> : les mesures CPU sur Colab ont un CV de {cv_cpu.min():.0f} à {cv_cpu.max():.0f} % d'une "
  f"exécution à l'autre et varient de {100 * abs(1 - r_cpu1.throughput_mean / r_cpu2.throughput_mean):.0f} % d'une "
  "session à l'autre. Sur GPU, la première exécution révèle une chauffe trop courte. Répéter les mesures (10 ici), donner la dispersion et normaliser par une référence de même "
  "session sont indispensables pour conclure.")
P("Pistes : quantification INT8 (OpenVINO NNCF, TensorRT), torch.compile en FP16 ou avec "
  "<font face='DVMono'>mode=\"max-autotune\"</font> sur un GPU plus récent, variation de la taille de batch "
  "(débit contre latence), et transfert hôte → GPU inclus dans la mesure pour un scénario de bout en bout.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor("#8a8984"))
    canvas.drawString(2 * cm, 1.2 * cm, "TP1 Déploiement IA — pipelines d'inférence ResNet-18")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(os.path.join(OUT, "rapport.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="TP1 Déploiement IA — pipelines d'inférence",
                        author="Romain")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("rapport/rapport.pdf écrit")
