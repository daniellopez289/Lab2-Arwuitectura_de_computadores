import csv
import glob
import os
import re
import subprocess

import numpy as np
import pandas as pd

DATASET = "gif_input.csv"
K_VALUES = list(range(2, 9))
OUT_DIR = "k_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_wcss(points, labels, centroids):
    assigned = centroids[labels]
    diff = points - assigned
    return float(np.sum(diff * diff))


def compute_silhouette(points, labels, K):
    # Matriz de distancias euclidianas N x N
    diff = points[:, None, :] - points[None, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=2))

    N = len(points)
    scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        c_i = labels[i]
        same_mask = (labels == c_i)
        same_count = int(np.sum(same_mask))
        if same_count <= 1:
            scores[i] = 0.0
            continue

        # a(i): distancia media a los demas puntos del mismo cluster
        a_i = float(np.sum(dist_mat[i, same_mask])) / (same_count - 1)

        # b(i): minima distancia media a otro cluster no vacio
        b_i = np.inf
        for c in range(K):
            if c == c_i:
                continue
            other_mask = (labels == c)
            if np.any(other_mask):
                mean_d = float(np.mean(dist_mat[i, other_mask]))
                if mean_d < b_i:
                    b_i = mean_d

        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0.0 else 0.0

    return float(np.mean(scores))


rows = []
print(f"Evaluando K = 2..8 en GPU sobre {DATASET}...")

for K in K_VALUES:
    prefix = os.path.join(OUT_DIR, f"k{K}")
    # Limpiar archivos previos de ese K
    for old_f in glob.glob(f"{prefix}_*.csv"):
        os.remove(old_f)

    res = subprocess.run(
        ["./kmeans_advanced", DATASET, "32", "0", prefix, str(K)],
        capture_output=True, text=True, check=True
    )
    out = res.stdout
    verified = "Verificacion contra CPU: CORRECTO" in out

    iter_files = sorted(
        glob.glob(f"{prefix}_iter_*.csv"),
        key=lambda p: int(re.search(r"iter_(\d+)\.csv$", p).group(1))
    )
    last_iter_file = iter_files[-1]
    last_iter = int(re.search(r"iter_(\d+)\.csv$", last_iter_file).group(1))

    df_pts = pd.read_csv(last_iter_file)
    points = df_pts[["x", "y"]].to_numpy(dtype=np.float64)
    labels = df_pts["label"].to_numpy(dtype=np.int64)

    df_cent = pd.read_csv(f"{prefix}_centroids.csv")
    df_last_c = df_cent[df_cent["iter"] == last_iter].sort_values("cluster")
    centroids = df_last_c[["x", "y"]].to_numpy(dtype=np.float64)

    wcss = compute_wcss(points, labels, centroids)
    sil = compute_silhouette(points, labels, K)

    rows.append({
        "K": K,
        "iterations": last_iter + 1,
        "wcss": round(wcss, 2),
        "silhouette": round(sil, 4),
        "verified": verified,
    })

    print(f"K={K} | iters={last_iter + 1:2d} | WCSS={wcss:10.2f} | Silhouette={sil:.4f} | OK={verified}")

with open("k_analysis_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

best = max(rows, key=lambda r: r["silhouette"])
print(f"\nMejor K segun Silhouette Score: K={best['K']} (Silhouette={best['silhouette']:.4f}, WCSS={best['wcss']:.2f})")
print("Guardado en: k_analysis_summary.csv")
