import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import numpy as np
import requests
from PIL import Image
import io
import base64

st.set_page_config(layout="wide")

API_URL = "http://api.gentlemoss-1a1ffa84.westeurope.azurecontainerapps.io"

colormap = cm.get_cmap("tab20")


def decode_img_b64(img_b64):
    """Prend un string base64 et retourne une image PIL"""
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes))


def decode_mask_b64(mask_b64):
    mask_bytes = base64.b64decode(mask_b64)
    mask = Image.open(io.BytesIO(mask_bytes))
    mask_array = np.array(mask)
    return mask_array


st.title("Cityscapes Segmentation Playground")
st.write("Choose your interaction mode in the left sidebar")

with st.sidebar:
    mode = st.radio(
        "Mode selection",
        ["From dataset", "From upload"],
        index=0,
    )

if mode == "From dataset":
    r = requests.get(f"{API_URL}/ids")
    ids = r.json().get("ids", [])
    options = [""] + ids
    selected_id = st.selectbox(
        "Choose an id",
        options,
        format_func=lambda x: "--- Select the wanted id ---" if x == "" else x,
    )
    col1, col2, col3 = st.columns(3)

    if selected_id:
        with col1:
            img_resp = requests.get(f"{API_URL}/image/{selected_id}")
            img_json = img_resp.json()
            if "image_b64" in img_json:
                image = decode_img_b64(img_json["image_b64"])
                st.image(image, caption="Real Image")
            else:
                st.write("Image not found.")
        with col2:
            mask_resp = requests.get(f"{API_URL}/mask/{selected_id}")
            mask_json = mask_resp.json()
            if "mask_b64" in mask_json:
                mask_array = decode_mask_b64(mask_json["mask_b64"])
                # Appliquer le colormap et convertir en RGB
                mask_rgb = (colormap(mask_array / 7.0)[:, :, :3] * 255).astype("uint8")
                st.image(mask_rgb, caption="Mask Ground Truth")
            else:
                st.write("Mask not found.")

    do_predict = st.button("Launch mask prediction on this photo")

    if do_predict and selected_id:
        with col3:
            pred_resp = requests.get(f"{API_URL}/predict/{selected_id}")
            try:
                pred_json = pred_resp.json()
            except Exception as e:
                st.error(f"Error while decoding JSON : {e}")
                st.write("Status code:", pred_resp.status_code)
                st.write("Response raw text:", pred_resp.text)
                pred_json = None
            if "mask_b64" in pred_json:
                pred_mask_array = decode_mask_b64(pred_json["mask_b64"])
                pred_mask_rgb = (
                    colormap(pred_mask_array / 7.0)[:, :, :3] * 255
                ).astype("uint8")
                st.image(pred_mask_rgb, caption="Predicted mask (2048x1024)")

            else:
                st.write("Prediction error or image not found.")

elif mode == "From upload":
    uploaded_file = st.file_uploader(
        "Upload your 2048x1024 image", type=["png", "jpg", "jpeg"]
    )
    col1, col2 = st.columns(2)
    if uploaded_file:
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Real Image")
        with col2:
            files = {"file": uploaded_file.getvalue()}
            resp = requests.post(f"{API_URL}/predict_upload", files=files)
            mask_b64 = resp.json().get("mask_b64")
            if mask_b64:
                pred_mask_array = decode_mask_b64(mask_b64)
                pred_mask_rgb = (
                    colormap(pred_mask_array / 7.0)[:, :, :3] * 255
                ).astype("uint8")
                st.image(pred_mask_rgb, caption="Predicted mask (2048x1024)")
            else:
                st.error("Predicted mask not received")
