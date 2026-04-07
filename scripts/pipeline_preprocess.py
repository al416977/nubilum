from pathlib import Path
import open3d as o3d
import sys

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(str(file_path))
    print("Nube original:", pcd)
    print("Número de puntos originales:", len(pcd.points))
    return pcd

def center_point_cloud(pcd):
    center = pcd.get_center()
    pcd.translate(-center)
    return pcd

#filtro estático, es excelente para eliminar puntos ruidosos. 
def filter_noise(pcd, nb_neighbors=16, std_ratio=10):
    filtered_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    print("Número de puntos filtrados:", len(filtered_pcd.points))
    return filtered_pcd

#codigo de reduccion de puntos 
def create_quick_view(pcd, voxel_size=0.30):
    #voxel_down_sample: Realiza una reducción de puntos utilizando la técnica de voxelización.
    quick_view = pcd.voxel_down_sample(voxel_size=voxel_size) 
    #voxel_size: Determina el tamaño de los "voxeles", o la resolución. Puedes ajustarlo según lo que consideres adecuado
    print("Número de puntos quick view:", len(quick_view.points))
    return quick_view

def save_outputs(base_dir, filtered_pcd, quick_view):
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(processed_dir / "edificio_filtrado.ply"), filtered_pcd)
    o3d.io.write_point_cloud(str(processed_dir / "edificio_quick_view.ply"), quick_view)

def print_metrics(original, filtered, quick):
    orig_n = len(original.points)
    filt_n = len(filtered.points)
    quick_n = len(quick.points)

    print(f"Puntos originales: {orig_n}")
    print(f"Puntos filtrados: {filt_n}")
    print(f"Puntos quick view: {quick_n}")
    print(f"Reducción tras filtrado: {100*(1-filt_n/orig_n):.4f}%")
    print(f"Reducción total quick view: {100*(1-quick_n/orig_n):.4f}%")

def main():
    base_dir = Path(__file__).resolve().parents[1]

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "edificio.xyz"

    file_path = base_dir / "data" / "raw" / file_name
    print("Archivo seleccionado:", file_path)

if __name__ == "__main__":
    main()