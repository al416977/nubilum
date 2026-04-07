# ============================================================
# preprocess_service.py
# Servicio de preprocesado de nubes de puntos para NUBILUM
# ============================================================

from pathlib import Path
import json
import numpy as np
import open3d as o3d
import os


def load_point_cloud(file_path: Path):
    """
    Carga una nube de puntos desde disco.

    Soporta especialmente archivos .xyz con dos variantes:
    - x y z
    - x y z r g b

    Además, intenta manejar archivos que tengan una cabecera
    en la primera línea, como por ejemplo:
    X Y Z R G B

    Parámetros
    ----------
    file_path : Path
        Ruta del archivo a cargar.

    Devuelve
    --------
    o3d.geometry.PointCloud
        Nube de puntos cargada.
    """
    suffix = file_path.suffix.lower()

    # ------------------------------------------------------------
    # Caso especial para .xyz
    # ------------------------------------------------------------
    if suffix == ".xyz":
        # Intento 1:
        # Cargar el archivo saltando una posible cabecera
        try:
            data = np.loadtxt(file_path, skiprows=1)

            # Si tiene 6 o más columnas: x y z r g b
            if data.ndim == 2 and data.shape[1] >= 6:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])

                colors = data[:, 3:6].astype(float)

                # Si el color viene en rango 0-255, lo normalizamos a 0-1
                if colors.max() > 1.0:
                    colors = colors / 255.0

                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                return pcd

            # Si tiene solo 3 o más columnas: x y z
            elif data.ndim == 2 and data.shape[1] >= 3:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                return pcd

        except Exception:
            # Si falla, seguimos con otro intento
            pass

        # Intento 2:
        # Cargar el archivo sin saltar cabecera
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

    # ------------------------------------------------------------
    # Resto de formatos o fallback general de Open3D
    # ------------------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(file_path))
    return pcd


def center_point_cloud(pcd: o3d.geometry.PointCloud):
    """
    Centra la nube en el origen restando su centro geométrico.
    """
    center = pcd.get_center()
    pcd.translate(-center)
    return pcd


def filter_noise(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 16,
    std_ratio: float = 10.0
):
    """
    Elimina ruido usando filtrado estadístico.

    Parámetros
    ----------
    nb_neighbors : int
        Número de vecinos usado por el filtro.
    std_ratio : float
        Sensibilidad del filtro. Cuanto menor, más agresivo.
    """
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return filtered_pcd


def create_quick_view(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.30
):
    """
    Reduce densidad de la nube con voxel downsampling.

    Parámetros
    ----------
    voxel_size : float
        Tamaño del voxel.
        Cuanto mayor, más reducción.
    """
    quick_view = pcd.voxel_down_sample(voxel_size=voxel_size)
    return quick_view


def save_outputs(project_dir: Path, stem: str, filtered_pcd, quick_view):
    """
    Guarda los archivos de salida del preprocesado.
    """
    project_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = project_dir / f"{stem}_filtrado.ply"
    quick_path = project_dir / f"{stem}_quick_view.ply"

    o3d.io.write_point_cloud(str(filtered_path), filtered_pcd)
    o3d.io.write_point_cloud(str(quick_path), quick_view)

    return filtered_path, quick_path


def build_report(input_name: str, original, filtered, quick, parameters: dict):
    """
    Construye un informe resumen del preprocesado.
    """
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
        "parameters": parameters
    }


def run_preprocess(
    file_name: str,
    voxel_size: float = 0.30,
    nb_neighbors: int = 16,
    std_ratio: float = 10.0
):
    """
    Ejecuta el flujo completo de preprocesado.

    Pasos:
    1. Cargar archivo
    2. Comprobar que no esté vacío
    3. Centrar nube
    4. Filtrar ruido
    5. Crear quick view
    6. Guardar resultados
    7. Generar informe
    """
    base_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("NUBILUM_DATA_DIR", str(base_dir / "data")))
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    file_path = raw_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    # 1) Carga
    pcd = load_point_cloud(file_path)

    # 2) Validación
    if len(pcd.points) == 0:
        raise ValueError("La nube cargada tiene 0 puntos")

    # 3) Centramos
    pcd = center_point_cloud(pcd)

    # 4) Filtramos ruido
    filtered_pcd = filter_noise(
        pcd,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # 5) Creamos quick view
    quick_view = create_quick_view(
        filtered_pcd,
        voxel_size=voxel_size
    )

    # 6) Guardamos resultados
    stem = Path(file_name).stem
    project_dir = processed_dir / stem
    filtered_path, quick_path = save_outputs(project_dir, stem, filtered_pcd, quick_view)

    # 7) Generamos informe
    parameters = {
        "voxel_size": voxel_size,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio
    }

    report = build_report(
        input_name=file_name,
        original=pcd,
        filtered=filtered_pcd,
        quick=quick_view,
        parameters=parameters
    )

    report_path = project_dir / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
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