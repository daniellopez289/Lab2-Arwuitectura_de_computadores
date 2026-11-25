import csv
import re
import statistics
import subprocess

DATASET = "large_input.csv"
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
REPEATS = 5
VERSIONS = [
    ("basic", "./kmeans_basic"),
    ("advanced", "./kmeans_advanced"),
]

def run_once(exe, block_size):
    cmd = [exe, DATASET, str(block_size)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Fallo {cmd}:\n{res.stdout}\n{res.stderr}")

    out = res.stdout
    ok = "Verificacion contra CPU: CORRECTO" in out
    m_blocks = re.search(r"numBlocks\s*=\s*(\d+)", out)
    m_total = re.search(r"Tiempo total ciclo K-means.*?:\s*([\d.]+)\s*ms", out)
    m_kernel = re.search(r"Tiempo acumulado solo kernel.*?:\s*([\d.]+)\s*ms", out)

    if not (m_blocks and m_total and m_kernel):
        raise RuntimeError(f"No se pudieron extraer metricas de {cmd}:\n{out}")

    return {
        "num_blocks": int(m_blocks.group(1)),
        "total_ms": float(m_total.group(1)),
        "kernel_ms": float(m_kernel.group(1)),
        "verified": ok,
    }

raw_rows = []
summary_rows = []

print(f"Iniciando benchmark sobre {DATASET} ({REPEATS} repeticiones por configuracion)...")

for version, exe in VERSIONS:
    for bs in BLOCK_SIZES:
        try:
            # 1 corrida de calentamiento (no se guarda)
            run_once(exe, bs)

            totals = []
            kernels = []
            num_blocks = 0
            all_ok = True

            for rep in range(1, REPEATS + 1):
                m = run_once(exe, bs)
                num_blocks = m["num_blocks"]
                all_ok = all_ok and m["verified"]
                totals.append(m["total_ms"])
                kernels.append(m["kernel_ms"])
                raw_rows.append({
                    "version": version,
                    "block_size": bs,
                    "num_blocks": num_blocks,
                    "run": rep,
                    "total_ms": m["total_ms"],
                    "kernel_ms": m["kernel_ms"],
                    "verified": m["verified"],
                })

            summary_rows.append({
                "version": version,
                "block_size": bs,
                "num_blocks": num_blocks,
                "total_mean_ms": round(statistics.mean(totals), 4),
                "total_std_ms": round(statistics.stdev(totals), 4),
                "kernel_mean_ms": round(statistics.mean(kernels), 4),
                "kernel_std_ms": round(statistics.stdev(kernels), 4),
                "all_verified": all_ok,
            })

            print(
                f"[{version:8s}] blockSize={bs:4d} | numBlocks={num_blocks:5d} | "
                f"total={statistics.mean(totals):7.4f} +/- {statistics.stdev(totals):.4f} ms | "
                f"kernel={statistics.mean(kernels):7.4f} +/- {statistics.stdev(kernels):.4f} ms | "
                f"OK={all_ok}"
            )
        except Exception as e:
            print(f"[{version:8s}] blockSize={bs:4d} -> ERROR: {e}")

with open("benchmark_raw.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
    writer.writeheader()
    writer.writerows(raw_rows)

with open("benchmark_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nArchivos guardados: benchmark_raw.csv y benchmark_summary.csv")