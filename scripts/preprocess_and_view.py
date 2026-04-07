import open3d as o3d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

FILE_PATH = DATA_DIR / "edificio.xyz"

pcd = o3d.io.read_point_cloud(str(FILE_PATH))
print("Nube original:", pcd)
print("Número de puntos originales:", len(pcd.points))
print("¿La nube original tiene colores?", pcd.has_colors())

# Centrar nube
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# Filtrado estadístico
nn = 16
std_multiplier = 10

filtered_pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=nn,
    std_ratio=std_multiplier
)

print("Nube filtrada:", filtered_pcd)
print("Número de puntos filtrados:", len(filtered_pcd.points))
print("¿La nube filtrada tiene colores?", filtered_pcd.has_colors())

# Vista rápida mucho más ligera
quick_view = filtered_pcd.voxel_down_sample(voxel_size=0.30)
print("Nube quick view:", quick_view)
print("Número de puntos quick view:", len(quick_view.points))
print("¿La quick view tiene colores?", quick_view.has_colors())

o3d.visualization.draw_geometries(
    [quick_view],
    window_name="Quick view edificio",
    width=1000,
    height=700
)

# Guardado
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

o3d.io.write_point_cloud(str(processed_dir / "edificio_filtrado.ply"), filtered_pcd)
o3d.io.write_point_cloud(str(processed_dir / "edificio_quick_view.ply"), quick_view)

print("Archivos guardados en data/processed")