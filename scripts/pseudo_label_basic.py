# ============================================================
# pseudo_label_basic.py
# Pseudoetiquetado geométrico básico para nubes de puntos
# ============================================================

import open3d as o3d
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Archivo de entrada
INPUT_PATH = BASE_DIR / "ai" / "raw_annotations" / "SalaElectrica" / "SalaElectrica_quick_view.ply"

# Archivo de salida con etiquetas numéricas
OUTPUT_PATH = BASE_DIR / "ai" / "raw_annotations" / "SalaElectrica" / "SalaElectrica_labeled.npy"

# Etiquetas
LABEL_OTHER = 0
LABEL_WALL = 1
LABEL_FLOOR = 2
LABEL_CEILING = 3


def load_cloud(input_path: Path):
    """
    Carga la nube de puntos desde disco.
    """
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise ValueError(f"La nube está vacía: {input_path}")

    return pcd, points


def detect_planes(
    pcd: o3d.geometry.PointCloud,
    max_planes: int = 8,
    min_points_remaining: int = 1000,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000
):
    """
    Detecta varios planos usando RANSAC.

    Devuelve una lista de tuplas:
    (plane_model, inliers_locales, indices_originales)
    """
    remaining_pcd = pcd
    remaining_indices = np.arange(len(pcd.points))
    planes = []

    for _ in range(max_planes):
        if len(remaining_pcd.points) < min_points_remaining:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # Guardamos los índices originales para poder etiquetar
        original_indices = remaining_indices[inliers]
        planes.append((plane_model, inliers, original_indices))

        # Eliminamos ese plano de la nube restante
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_indices = np.delete(remaining_indices, inliers)

    return planes


def estimate_floor_and_ceiling(points: np.ndarray, planes):
    """
    Analiza los planos horizontales para encontrar:
    - suelo = plano horizontal más bajo
    - techo = plano horizontal más alto
    """
    floor_z = None
    ceiling_z = None

    for plane_model, _, original_indices in planes:
        a, b, c, d = plane_model

        normal = np.array([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Si |z| es alto, el plano es horizontal
        verticality = abs(normal[2])

        if verticality > 0.9:
            z_values = points[original_indices][:, 2]
            avg_z = np.mean(z_values)

            if floor_z is None or avg_z < floor_z:
                floor_z = avg_z

            if ceiling_z is None or avg_z > ceiling_z:
                ceiling_z = avg_z

    return floor_z, ceiling_z


def assign_labels(points: np.ndarray, planes, floor_z, ceiling_z, tolerance: float = 0.2):
    """
    Asigna etiquetas a los puntos según la orientación y altura de cada plano.
    """
    labels = np.zeros(len(points), dtype=int)

    for plane_model, _, original_indices in planes:
        a, b, c, d = plane_model

        normal = np.array([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)

        verticality = abs(normal[2])

        # Planos horizontales -> suelo o techo
        if verticality > 0.9:
            z_values = points[original_indices][:, 2]
            avg_z = np.mean(z_values)

            if floor_z is not None and abs(avg_z - floor_z) < tolerance:
                labels[original_indices] = LABEL_FLOOR
            elif ceiling_z is not None and abs(avg_z - ceiling_z) < tolerance:
                labels[original_indices] = LABEL_CEILING
            else:
                labels[original_indices] = LABEL_OTHER

        # Planos no horizontales -> pared
        else:
            labels[original_indices] = LABEL_WALL

    return labels


def save_labels(points: np.ndarray, labels: np.ndarray, output_path: Path):
    """
    Guarda un array N x 4:
    x, y, z, label
    """
    data = np.hstack([points, labels.reshape(-1, 1)])
    np.save(output_path, data)


def main():
    """
    Flujo principal del pseudoetiquetado.
    """
    print("Cargando nube...")
    pcd, points = load_cloud(INPUT_PATH)

    print("Detectando planos...")
    planes = detect_planes(
        pcd=pcd,
        max_planes=8,
        min_points_remaining=1000,
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=1000
    )

    print(f"Planos detectados: {len(planes)}")

    print("Estimando suelo y techo...")
    floor_z, ceiling_z = estimate_floor_and_ceiling(points, planes)
    print("floor_z =", floor_z)
    print("ceiling_z =", ceiling_z)

    print("Asignando etiquetas...")
    labels = assign_labels(
        points=points,
        planes=planes,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        tolerance=0.2
    )

    print("Guardando etiquetas...")
    save_labels(points, labels, OUTPUT_PATH)

    print("✔ Pseudoetiquetado completado")
    print(f"Archivo guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()