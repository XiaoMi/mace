# Refer to https://www.tensorflow.org/model_optimization/guide

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


tfds.disable_progress_bar()
tf.enable_v2_behavior()

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        ),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

model.save("mnist.h5")

quantize_model = tfmot.quantization.keras.quantize_model

quantization_aware_model = quantize_model(model)

quantization_aware_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

quantization_aware_model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

quantization_aware_model.save("mnist-int8.h5")
