"""
Creates a model that, given a digital, monochrome 255x255 drawing, can distinguish between a wywy head and something that is not a wywy head.

This requires data to be stored in the data folder with a subfolder called "wywy" which contains many drawings of wywy heads and a subfolder called "doodles" which contains many drawings of doodles that are not wywy heads.
"""
import tensorflow as tf
# from tensorflow import data as tf_data
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
SEED = 1234
DATA_DIR = "data"
# What resolution should Keras load up the image data as?
IMAGE_SHAPE = (128, 128, 1)
IMAGE_SIZE= (128, 128)
BATCH_SIZE = 150
DROPOUT_RATE = 0.25
EPOCHS = 25
CALLBACKS = [
    keras.callbacks.ModelCheckpoint("models/epoch_{epoch}.keras"),
]

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255., label

def load_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    # train_ds = train_ds.shuffle(info.splits["train"].num_examples)
    # train_ds = train_ds.batch(128)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    # val_ds = val_ds.shuffle(info.splits["train"].num_examples)
    # val_ds = val_ds.batch(128)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def train_model(train, test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(height=128,width=128),
        tf.keras.layers.Flatten(input_shape=(128, 128)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    model.fit(
        train,
        epochs=EPOCHS,
        validation_data=test,
        callbacks=CALLBACKS,
    )

if __name__ == "__main__":
    train, test = load_dataset()
    
    train_model(train, test)