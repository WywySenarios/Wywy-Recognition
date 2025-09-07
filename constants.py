"""Constants file to consolidate all parameters to one place.
"""

import keras

class MODEL:
    SEED: int = 1234
    DATA_DIR: str = "data"
    IMAGE_SHAPE = (128, 128, 1)
    IMAGE_SIZE= (128, 128)
    BATCH_SIZE = 150
    DROPOUT_RATE = 0.25
    EPOCHS = 25
    CALLBACKS = [
        keras.callbacks.ModelCheckpoint("models/epoch_{epoch}.keras"),
    ]
    
    # models to test
    MODELS_PATH = "models"
    DEFAULT_MODEL_PATH = "models/epoch_13.keras"
    
    GET_ALL_MODELS = False
    MODELS_NAMES = ["epoch_13", "epoch_14"]

class PYGAME_UI:
    # Menu & UI
    WIDTH = 255
    HEIGHT = 255 + 70 + 3
    
    # brush settings
    BRUSH_SIZE = 5
    BRUSH_COLOR = "black"
    BRUSH_PREVIEW_COLOR = (136, 138, 137)
    
    # Print out the other models' predictions?
    VERBOSE = True