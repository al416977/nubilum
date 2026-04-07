
from pathlib import Path
import json
import os
import numpy as np
import open3d as o3d

LABEL_OTHER = 0
LABEL_WALL = 1
LABEL_FLOOR = 2
LABEL_CEILING = 3

LABEL_NAMES = {
    LABEL_OTHER: "otros",
    LABEL_WALL: "pared",
    LABEL_FLOOR: "suelo",
    LABEL_CEILING: "techo",
}

LABEL_COLORS = {
    LABEL_OTHER: [0.5, 0.5, 0.5],
    LABEL_WALL: [1.0, 0.0, 0.0],
    LABEL_FLOOR: [0.0, 1.0, 0.0],
    LABEL_CEILING: [0.0, 0.0, 1.0],
}


def detect_planes(
    pcd: o3d.geometry.PointCloud,
    max_planes: int = 8,
    min_points_remaining: int = 1000,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
):
    remaining_pcd = pcd
    remaining_indices = np.arange(len(pcd.points))
    planes = []

    for _ in range(max_planes):
        if len(remaining_pcd.points) < min_points_remaining:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        original_indices = remaining_indices[inliers]
        planes.append((plane_model, original_indices))

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_indices = np.delete(remaining_indices, inliers)

    return planes


def estimate_floor_and_ceiling(points: np.ndarray, planes):
    floor_z = None
    ceiling_z = None

    for plane_model, original_indices in planes:
        a, b, c, _ = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)
        verticality = abs(normal[2])

        if verticality > 0.9:
            z_values = points[original_indices][:, 2]
            avg_z = float(np.mean(z_values))
            if floor_z is None or avg_z < floor_z:
                floor_z = avg_z
            if ceiling_z is None or avg_z > ceiling_z:
                ceiling_z = avg_z

    return floor_z, ceiling_z


def assign_labels(points: np.ndarray, planes, floor_z, ceiling_z, tolerance: float = 0.2):
    labels = np.zeros(len(points), dtype=int)

    for plane_model, original_indices in planes:
        a, b, c, _ = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)
        verticality = abs(normal[2])

        if verticality > 0.9:
            z_values = points[original_indices][:, 2]
            avg_z = float(np.mean(z_values))
            if floor_z is not None and abs(avg_z - floor_z) < tolerance:
                labels[original_indices] = LABEL_FLOOR
            elif ceiling_z is not None and abs(avg_z - ceiling_z) < tolerance:
                labels[original_indices] = LABEL_CEILING
            else:
                labels[original_indices] = LABEL_OTHER
        else:
            labels[original_indices] = LABEL_WALL

    return labels


def classify_project(project: str):
    base_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("NUBILUM_DATA_DIR", str(base_dir / "data")))
    processed_dir = data_dir / "processed"
    ai_project_dir = base_dir / "ai" / "raw_annotations" / project
    ai_project_dir.mkdir(parents=True, exist_ok=True)

    project_dir = processed_dir / project
    filtered_path = project_dir / f"{project}_filtrado.ply"
    if not filtered_path.exists():
        raise FileNotFoundError(f"No existe la nube filtrada del proyecto: {filtered_path}")

    pcd = o3d.io.read_point_cloud(str(filtered_path))
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("La nube filtrada está vacía")

    planes = detect_planes(pcd=pcd)
    floor_z, ceiling_z = estimate_floor_and_ceiling(points, planes)
    labels = assign_labels(points=points, planes=planes, floor_z=floor_z, ceiling_z=ceiling_z)

    labeled_data = np.hstack([points, labels.reshape(-1, 1)])
    labeled_npy_path = ai_project_dir / f"{project}_labeled.npy"
    np.save(labeled_npy_path, labeled_data)

    colors = np.array([LABEL_COLORS.get(int(label), [1.0, 1.0, 1.0]) for label in labels], dtype=float)
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    colored_ply_path = ai_project_dir / f"{project}_labeled_colored.ply"
    o3d.io.write_point_cloud(str(colored_ply_path), colored_pcd)

    unique, counts = np.unique(labels, return_counts=True)
    distribution = {LABEL_NAMES.get(int(u), str(int(u))): int(c) for u, c in zip(unique, counts)}

    report = {
        "project": project,
        "input_file": filtered_path.name,
        "points": int(len(points)),
        "planes_detected": int(len(planes)),
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "label_distribution": distribution,
        "legend": {
            "pared": "rojo",
            "suelo": "verde",
            "techo": "azul",
            "otros": "gris",
        },
        "outputs": {
            "labeled_npy": labeled_npy_path.name,
            "colored_ply": colored_ply_path.name,
        },
    }

    report_path = ai_project_dir / f"{project}_classification_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
