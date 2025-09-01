"""
Creates a model that, given a digital, monochrome 255x255 drawing, can distinguish between a wywy head and something that is not a wywy head.

This requires data to be stored in the data folder with a subfolder called "wywy" which contains many drawings of wywy heads and a subfolder called "doodles" which contains many drawings of doodles that are not wywy heads.
"""
import tensorflow as tf
from tensorflow import data as tf_data
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
IMAGE_SHAPE = (255, 255, 1)
IMAGE_SIZE = (255, 255)
BATCH_SIZE = 150
DROPOUT_RATE = 0.25
EPOCHS = 25
CALLBACKS = [
    # keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# @TODO
def create_model(input_shape, num_classes):
    """
    Creates and trains a model to detect drawings of Wywys from pieces of trash using data supplied from the filesystem.
    """
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    x = keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x

    # add a couple of nested activation blocks
    for size in [256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = keras.layers.add([x, residual]) # Add back residual
        previous_block_activation = x

    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    # set Dropout to combat overfitting
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    outputs = keras.layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

if __name__ == "__main__":
    # prepare to extract training & validation data
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "data",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    # visualize a small sample of the images (first few)
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title("Plot " + str(i))
            plt.axis("off")

    model = create_model(IMAGE_SHAPE, 2)

    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )

    print("Train cardinality:", tf.data.experimental.cardinality(train_ds))
    print("Val cardinality:", tf.data.experimental.cardinality(val_ds))

    model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=CALLBACKS,
        validation_data=val_ds,
        # verbose="1",
    )

    # save the model for testing & future use
    model.save("model.keras")


    

    
