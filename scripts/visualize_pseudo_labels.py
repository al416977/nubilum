import numpy as np
import open3d as o3d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "ai" / "raw_annotations" / "SalaElectrica" / "SalaElectrica_labeled.npy"
OUTPUT_PLY = BASE_DIR / "ai" / "raw_annotations" / "SalaElectrica" / "SalaElectrica_labeled_colored.ply"

# Cargar datos
data = np.load(INPUT_PATH)

points = data[:, :3]
labels = data[:, 3].astype(int)

# Colores por clase
label_colors = {
    0: [0.5, 0.5, 0.5],  # other -> gris
    1: [1.0, 0.0, 0.0],  # wall -> rojo
    2: [0.0, 1.0, 0.0],  # floor -> verde
    3: [0.0, 0.0, 1.0],  # ceiling -> azul
}

colors = np.array([label_colors.get(label, [1.0, 1.0, 1.0]) for label in labels])

# Crear nube de puntos coloreada
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Guardar y visualizar
o3d.io.write_point_cloud(str(OUTPUT_PLY), pcd)

print("✔ Archivo coloreado guardado en:")
print(OUTPUT_PLY)

unique, counts = np.unique(labels, return_counts=True)
print("\nDistribución de etiquetas:")
for u, c in zip(unique, counts):
    print(f"Clase {u}: {c} puntos")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Pseudoetiquetas coloreadas",
    width=1000,
    height=700
)