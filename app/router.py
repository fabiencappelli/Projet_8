from fastapi import APIRouter, UploadFile, File

# import requests
import base64
from app.config import DATA_DIR, IMG_PATH_END, MASK_PATH_END
import os
from app.model import predict_mask
import io


def encode_file_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


router = APIRouter()


@router.get("/ids")
def get_ids():
    try:
        fichiers = [f for f in os.listdir(DATA_DIR) if f.endswith(IMG_PATH_END)]
        ids = [f[: -len(IMG_PATH_END)] for f in fichiers]
        return {"ids": ids}
    except Exception as e:
        return {"error": str(e), "ids": []}


@router.get("/image/{id}")
def get_image(id: str):
    img_path = os.path.join(DATA_DIR, f"{id}{IMG_PATH_END}")
    if not os.path.exists(img_path):
        return {"error": "Image not found", "id": id}
    img_b64 = encode_file_base64(img_path)
    return {"id": id, "image_b64": img_b64}


@router.get("/mask/{id}")
def get_mask(id: str):
    mask_path = os.path.join(DATA_DIR, f"{id}{MASK_PATH_END}")
    if not os.path.exists(mask_path):
        return {"error": "Mask not found", "id": id}
    mask_b64 = encode_file_base64(mask_path)
    return {"id": id, "mask_b64": mask_b64}


@router.post("/predict_upload")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred_mask_img = predict_mask(io.BytesIO(image_bytes))
    # Sérialise en base64 PNG
    buf = io.BytesIO()
    pred_mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"mask_b64": mask_b64}


@router.get("/predict/{id}")
def predict_from_id(id: str):
    # Cherche l'image dans le dossier local
    img_path = os.path.join(DATA_DIR, f"{id}{IMG_PATH_END}")
    if not os.path.exists(img_path):
        return {"error": "Image not found", "id": id}
    # Prédiction
    pred_mask_img = predict_mask(img_path)
    # Sérialise le mask prédit en base64 PNG
    buf = io.BytesIO()
    pred_mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"id": id, "mask_b64": mask_b64}


# @router.get("/ids")
# def get_ids():
#    try:
#        r = requests.get(DATA_DIR, timeout=3)
#        files = r.json()
#        fichiers = [f["name"] for f in files if f["type"] == "file"]
#        ids = [f[: -len(IMG_PATH_END)] for f in fichiers if f.endswith(IMG_PATH_END)]
#        return {"ids": ids}
#    except Exception as e:
#        return {"error": str(e), "ids": []}


# def build_raw_url(id, suffix):
#    return f"https://raw.githubusercontent.com/fabiencappelli/Projet_8/main/data/{id}{suffix}"


# @router.get("/image/{id}")
# def get_image(id: str):
#    url = build_raw_url(id, IMG_PATH_END)
#    resp = requests.get(url)
#    if resp.status_code != 200:
#        return {"error": "Not found", "id": id}
#    img_b64 = base64.b64encode(resp.content).decode("utf-8")
#    return {"id": id, "image_b64": img_b64}


# @router.get("/mask/{id}")
# def get_mask(id: str):
#    url = build_raw_url(id, MASK_PATH_END)
#    resp = requests.get(url)
#    if resp.status_code != 200:
#        return {"error": "Not found", "id": id}
#    msk_b64 = base64.b64encode(resp.content).decode("utf-8")
#    return {"id": id, "mask_b64": msk_b64}
