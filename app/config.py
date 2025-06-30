import os

# Chemins vers les assets du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "FULL_efficientnetb0_FPN_BS_8_LR_0.001_DICE_LOSS.keras"
)

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data", "val_images")
VAL_MASKS_DIR = os.path.join(BASE_DIR, "data", "val_masks")
