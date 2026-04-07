
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import os


def load_point_cloud(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".xyz":
        try:
            data = np.loadtxt(file_path, skiprows=1)
            if data.ndim == 2 and data.shape[1] >= 6:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                colors = data[:, 3:6].astype(float)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                return pcd
            elif data.ndim == 2 and data.shape[1] >= 3:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                return pcd
        except Exception:
            pass
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] >= 6:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                colors = data[:, 3:6].astype(float)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                return pcd
            elif data.ndim == 2 and data.shape[1] >= 3:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                return pcd
        except Exception:
            pass
    return o3d.io.read_point_cloud(str(file_path))


def center_point_cloud(pcd: o3d.geometry.PointCloud):
    center = pcd.get_center()
    pcd.translate(-center)
    return pcd


def filter_noise(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 16, std_ratio: float = 10.0):
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered_pcd


def create_quick_view(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.30):
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def save_outputs(project_dir: Path, stem: str, filtered_pcd, quick_view):
    project_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = project_dir / f"{stem}_filtrado.ply"
    quick_path = project_dir / f"{stem}_quick_view.ply"
    o3d.io.write_point_cloud(str(filtered_path), filtered_pcd)
    o3d.io.write_point_cloud(str(quick_path), quick_view)
    return filtered_path, quick_path


def build_report(input_name: str, original, filtered, quick, parameters: dict):
    orig_n = len(original.points)
    filt_n = len(filtered.points)
    quick_n = len(quick.points)
    return {
        "input_file": input_name,
        "original_points": orig_n,
        "filtered_points": filt_n,
        "quick_view_points": quick_n,
        "filter_reduction_percent": round(100 * (1 - filt_n / orig_n), 4) if orig_n else 0,
        "quick_view_reduction_percent": round(100 * (1 - quick_n / orig_n), 4) if orig_n else 0,
        "has_colors_original": original.has_colors(),
        "has_colors_filtered": filtered.has_colors(),
        "has_colors_quick_view": quick.has_colors(),
        "parameters": parameters,
        "quick_view_mode": "sin reducción" if parameters.get("voxel_size", 0) <= 0 else "reducida",
    }


def run_preprocess(file_name: str, voxel_size: float = 0.30, nb_neighbors: int = 16, std_ratio: float = 10.0):
    base_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("NUBILUM_DATA_DIR", str(base_dir / "data")))
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    file_path = raw_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")
    if nb_neighbors < 1:
        raise ValueError("El valor de análisis del ruido debe ser mayor o igual que 1")
    if std_ratio <= 0:
        raise ValueError("El valor de limpieza del ruido debe ser mayor que 0")

    pcd = load_point_cloud(file_path)
    if len(pcd.points) == 0:
        raise ValueError("La nube cargada tiene 0 puntos")

    pcd = center_point_cloud(pcd)
    filtered_pcd = filter_noise(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(filtered_pcd.points) == 0:
        raise ValueError("La nube se ha quedado vacía tras la limpieza")

    quick_view = create_quick_view(filtered_pcd, voxel_size=voxel_size)
    stem = Path(file_name).stem
    project_dir = processed_dir / stem
    filtered_path, quick_path = save_outputs(project_dir, stem, filtered_pcd, quick_view)

    parameters = {
        "voxel_size": voxel_size,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
    }
    report = build_report(input_name=file_name, original=pcd, filtered=filtered_pcd, quick=quick_view, parameters=parameters)
    report_path = project_dir / f"{stem}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return {
        "message": "Preprocesado completado",
        "project": stem,
        "report": report,
        "outputs": {
            "filtered_file": str(filtered_path),
            "quick_view_file": str(quick_path),
            "report_file": str(report_path),
        },
    }
