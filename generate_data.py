import numpy as np

rng = np.random.default_rng(42)
centers = [(-4.0, 2.0), (8.0, -3.0), (14.0, 15.0)]
n_per_cluster = 500_000  # Total: 1,500,000 puntos

points = np.vstack([
    rng.normal(loc=c, scale=3.0, size=(n_per_cluster, 2))
    for c in centers
])

np.savetxt("large_input.csv", points, fmt="%.4f", delimiter=",")
print(f"Generado large_input.csv con {len(points)} puntos.")