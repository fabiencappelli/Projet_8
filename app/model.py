import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from PIL import Image
import albumentations as A
from app.config import MODEL_PATH


from tensorflow import keras

import segmentation_models as sm
import tensorflow as tf

from tensorflow.keras.models import load_model

preprocess_input = sm.get_preprocessing("efficientnetb0")

transform = A.Compose(
    [
        A.Resize(256, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        A.Lambda(image=preprocess_input),
    ]
)

model = load_model(MODEL_PATH, compile=False)


def preprocess_image(image_file):
    # image = np.array(Image.open(image_file).convert("RGB"))
    image = np.array(Image.open(image_file))
    image = transform(image=image)["image"]
    # On ajoute une dimension batch pour Keras/TensorFlow
    image = np.expand_dims(image, 0)  # (1, H, W, C)
    return image


def predict_mask(image_file):
    image = preprocess_image(image_file)
    print("Shape image pour prédiction:", image.shape)
    print("Min/max/mean image:", image.min(), image.max(), image.mean())
    print("dtype:", image.dtype)
    pred = model.predict(image)
    pred_mask = np.argmax(pred[0], axis=-1)
    print("Valeurs uniques du masque prédit:", np.unique(pred_mask))
    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
    pred_mask_img = pred_mask_img.resize((2048, 1024), resample=Image.NEAREST)
    return pred_mask_img
