import os
from tensorflow import keras
from constants import MODEL
from constants import FILE_VALIDATION
import numpy as np


models = []
if MODEL.GET_ALL_MODELS:
    for model_name in os.listdir(MODEL.MODELS_PATH):
        if model_name.endswith(".keras"):
            models.append(keras.models.load_model(os.path.join(MODEL.MODELS_PATH, model_name)))
else:
    for model_name in MODEL.MODELS_NAMES:
        models.append(keras.models.load_model(os.path.join(MODEL.MODELS_PATH, model_name) + ".keras"))
        
default_model = keras.models.load_model(MODEL.DEFAULT_MODEL_PATH)

# try out each and every image on every model available
for image_folder in os.listdir(FILE_VALIDATION.VALIDATION_FILES_PATH):
    for image in os.listdir(os.path.join(FILE_VALIDATION.VALIDATION_FILES_PATH, image_folder)):
        currentImage = keras.utils.load_img(
    os.path.join(FILE_VALIDATION.VALIDATION_FILES_PATH, image_folder, image),
    color_mode="rgb",
    target_size=MODEL.IMAGE_SHAPE,
    # interpolation="nearest",
    keep_aspect_ratio=False,
)
        arr = np.array(currentImage, dtype=np.float32) / 255.0
        arr = arr.reshape(1, *MODEL.IMAGE_SIZE, 3)

        print("TESTING", image, "FROM", image_folder, "-----------")
        pred = default_model.predict(arr, verbose=0)
        print(" *", round(pred[0][1] * 100, 0), "% Wywy,", round(pred[0][0] * 100, 0), "% Other (DEFAULT MODEL)")
        for model in models:
            pred = model.predict(arr, verbose=0)
            print(" *", round(pred[0][1] * 100, 0), "% Wywy,", round(pred[0][0] * 100, 0), "% Other")

        print("-----------------------\n\n\n")