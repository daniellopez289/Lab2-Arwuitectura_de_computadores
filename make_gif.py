import glob
import io
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Crear dataset visual (con la misma distribucion de large_input.csv)
rng = np.random.default_rng(42)
centers = [(-4.0, 2.0), (8.0, -3.0), (14.0, 15.0)]
points = np.vstack([
    rng.normal(loc=c, scale=3.0, size=(500, 2)) for c in centers
])
np.savetxt("gif_input.csv", points, fmt="%.4f", delimiter=",")

# Ejecutar GPU avanzada exportando cada iteracion
os.makedirs("gif_frames", exist_ok=True)
prefix = "gif_frames/run"
res = subprocess.run(
    ["./kmeans_advanced", "gif_input.csv", "32", "0", prefix],
    capture_output=True, text=True, check=True
)
print(res.stdout)

# Leer historial de centroides y archivos de iteraciones
df_cent = pd.read_csv(f"{prefix}_centroids.csv")
iter_files = sorted(
    glob.glob(f"{prefix}_iter_*.csv"),
    key=lambda p: int(re.search(r"iter_(\d+)\.csv$", p).group(1))
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
xmin, xmax = points[:, 0].min() - 2, points[:, 0].max() + 2
ymin, ymax = points[:, 1].min() - 2, points[:, 1].max() + 2

pil_frames = []

def render_frame(df_pts, current_iter, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)

    if df_pts is None:
        ax.scatter(points[:, 0], points[:, 1], s=12, c="gray", alpha=0.45, label="Sin asignar")
    else:
        for c in range(3):
            sub = df_pts[df_pts["label"] == c]
            ax.scatter(sub["x"], sub["y"], s=12, c=colors[c], alpha=0.55, label=f"Cluster {c}")

    # Dibujar trayectoria y posicion actual de cada centroide
    hist = df_cent[df_cent["iter"] <= current_iter]
    for c in range(3):
        hc = hist[hist["cluster"] == c].sort_values("iter")
        ax.plot(hc["x"], hc["y"], "--", color="black", linewidth=1.2, alpha=0.7)
        cur = hc.iloc[-1]
        ax.scatter(cur["x"], cur["y"], s=180, c=colors[c], marker="X",
                   edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# Cuadro inicial: semillas
pil_frames.append(render_frame(None, -1, "Inicializacion (Semillas iniciales)"))

# Un cuadro por iteracion de GPU
for path in iter_files:
    it = int(re.search(r"iter_(\d+)\.csv$", path).group(1))
    df_pts = pd.read_csv(path)
    pil_frames.append(render_frame(df_pts, it, f"K-means GPU Avanzado — Iteracion {it}"))

# Guardar GIF animado
durations = [900] * (len(pil_frames) - 1) + [1800]
pil_frames[0].save(
    "kmeans_evolution.gif",
    save_all=True,
    append_images=pil_frames[1:],
    duration=durations,
    loop=0
)
print(f"GIF generado: kmeans_evolution.gif ({len(pil_frames)} cuadros)")
