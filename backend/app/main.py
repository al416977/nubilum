from pathlib import Path
import os
import json
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.app.services.preprocess_service import run_preprocess
from backend.app.services.classification_service import classify_project

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

# Ruta de datos configurable:
# - local: usa ./data
# - Render con disco: usa /var/data/nubilum (o la que pongas en la variable)
DATA_DIR = Path(os.getenv("NUBILUM_DATA_DIR", str(BASE_DIR / "data")))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AI_DIR = BASE_DIR / "ai" / "raw_annotations"
AI_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir CSS/JS/recursos del frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/projects-page", response_class=HTMLResponse)
def serve_projects_page():
    return (FRONTEND_DIR / "projects.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/files")
def list_files():
    if not RAW_DIR.exists():
        return {"files": []}

    allowed_ext = {".xyz", ".pts", ".ply", ".pcd", ".las", ".laz"}
    files = [
        f.name for f in RAW_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_ext
    ]
    files.sort()
    return {"files": files}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_ext = {".xyz", ".pts", ".ply", ".pcd", ".las", ".laz"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Formato no permitido: {suffix}")

    save_path = RAW_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    return {
        "message": "Archivo subido correctamente",
        "file_name": file.filename,
        "saved_to": str(save_path)
    }


@app.get("/preprocess")
def preprocess(
    file_name: str,
    voxel_size: float = 0.30,
    nb_neighbors: int = 16,
    std_ratio: float = 10.0
):
    try:
        print(f"[PREPROCESS] Iniciando: file_name={file_name}, voxel_size={voxel_size}, nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        result = run_preprocess(
            file_name=file_name,
            voxel_size=voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"[PREPROCESS] Completado: {file_name}")
        return result
    except FileNotFoundError as e:
        print(f"[PREPROCESS] FileNotFoundError: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print(f"[PREPROCESS] ValueError: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[PREPROCESS] Error inesperado: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(project: str):
    try:
        print(f"[CLASSIFY] Iniciando clasificación para: {project}")
        report = classify_project(project)
        print(f"[CLASSIFY] Clasificación completada para: {project}")
        return {
            "message": "Clasificación completada",
            "project": project,
            "report": report
        }
    except FileNotFoundError as e:
        print(f"[CLASSIFY] FileNotFoundError en {project}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print(f"[CLASSIFY] ValueError en {project}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[CLASSIFY] Error inesperado en {project}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/report")
def get_report(file_stem: str):
    report_path = PROCESSED_DIR / file_stem / f"{file_stem}_report.json"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No existe el informe solicitado")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@app.get("/download")
def download_file(project: str, file_name: str):
    file_path = PROCESSED_DIR / project / file_name
    print(f"[DOWNLOAD] project={project}")
    print(f"[DOWNLOAD] file_name={file_name}")
    print(f"[DOWNLOAD] file_path={file_path}")
    print(f"[DOWNLOAD] exists={file_path.exists()}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No existe el archivo solicitado")

    return FileResponse(path=file_path, filename=file_name)


@app.get("/view-file")
def view_file(project: str, file_name: str):
    file_path = PROCESSED_DIR / project / file_name
    print(f"[VIEW_FILE] project={project}")
    print(f"[VIEW_FILE] file_name={file_name}")
    print(f"[VIEW_FILE] file_path={file_path}")
    print(f"[VIEW_FILE] exists={file_path.exists()}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No existe el archivo solicitado")

    return FileResponse(path=file_path, media_type="application/octet-stream")

@app.get("/projects")
def list_projects():
    ai_dir = AI_DIR

    if not PROCESSED_DIR.exists():
        return {"projects": []}

    projects = []

    for project_dir in PROCESSED_DIR.iterdir():
        if project_dir.is_dir():
            project_name = project_dir.name

            filtered_file = f"{project_name}_filtrado.ply"
            quick_view_file = f"{project_name}_quick_view.ply"
            report_file = f"{project_name}_report.json"
            pseudo_label_file = f"{project_name}_labeled_colored.ply"
            class_report_file = f"{project_name}_classification_report.json"

            ai_project_dir = ai_dir / project_name

            projects.append({
                "name": project_name,
                "filtered_file": filtered_file if (project_dir / filtered_file).exists() else None,
                "quick_view_file": quick_view_file if (project_dir / quick_view_file).exists() else None,
                "report_file": report_file if (project_dir / report_file).exists() else None,
                "pseudo_label_file": pseudo_label_file if (ai_project_dir / pseudo_label_file).exists() else None,
                "classification_report_file": class_report_file if (ai_project_dir / class_report_file).exists() else None,
            })

    projects.sort(key=lambda x: x["name"].lower())
    return {"projects": projects}


@app.get("/view-ai-file")
def view_ai_file(project: str, file_name: str):
    file_path = AI_DIR / project / file_name
    print(f"[VIEW_AI_FILE] project={project}")
    print(f"[VIEW_AI_FILE] file_name={file_name}")
    print(f"[VIEW_AI_FILE] file_path={file_path}")
    print(f"[VIEW_AI_FILE] exists={file_path.exists()}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No existe el archivo de anotación solicitado")

    return FileResponse(path=file_path, media_type="application/octet-stream")


@app.get("/download-ai-file")
def download_ai_file(project: str, file_name: str):
    file_path = AI_DIR / project / file_name
    print(f"[DOWNLOAD_AI] project={project}")
    print(f"[DOWNLOAD_AI] file_name={file_name}")
    print(f"[DOWNLOAD_AI] file_path={file_path}")
    print(f"[DOWNLOAD_AI] exists={file_path.exists()}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No existe el archivo de anotación solicitado")

    return FileResponse(path=file_path, filename=file_name)
