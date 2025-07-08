import os

# Chemins vers les assets du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "FULL_efficientnetb0_FPN_BS_8_LR_0.001_DICE_LOSS.keras"
)

# DATA_DIR = "https://api.github.com/repos/fabiencappelli/Projet_8/contents/data"
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

IMG_PATH_END = "_leftImg8bit.png"
MASK_PATH_END = "_gtFine_labelIds.png"

dicoclasses = {
    0: "void",
    1: "flat",
    2: "construction",
    3: "object",
    4: "nature",
    5: "sky",
    6: "human",
    7: "vehicle",
}
