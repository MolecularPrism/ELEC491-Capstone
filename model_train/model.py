# model.py
# TensorFlow/Keras implementation of TinyCNN (equivalent to the PyTorch version).
# Notes:
# - Default input layout is NHWC: (batch, height=6, width=50, channels=1)
# - Output are logits with shape (batch, num_classes)

from __future__ import annotations

import tensorflow as tf


class TinyCNN(tf.keras.Model):
    def __init__(
        self,
        num_classes: int = 2,
        conv1_out: int = 8,
        conv2_out: int = 8,
        name: str = "TinyCNN",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_classes = int(num_classes)
        self.conv1_out = int(conv1_out)
        self.conv2_out = int(conv2_out)

        # Conv over time dimension (kernel=(1,16)) with SAME padding
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.conv1_out,
            kernel_size=(1, 16),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="conv1",
        )

        # Conv over channel/spatial dimension (kernel=(6,1)) with SAME padding
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.conv2_out,
            kernel_size=(6, 1),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="conv2",
        )

        self.relu = tf.keras.layers.ReLU(name="relu")
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")

        # Final classifier layer, outputs logits (no softmax here)
        self.fc = tf.keras.layers.Dense(self.num_classes, name="fc")

    def call(self, inputs, training: bool = False):
        """
        Forward pass.
        Expected input shape (NHWC): (B, 6, 50, 1)
        """
        x = tf.convert_to_tensor(inputs)

        # Allow (B,6,50) and auto-expand to (B,6,50,1)
        if x.shape.rank == 3:
            x = tf.expand_dims(x, axis=-1)

        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.relu(x)
        x = self.gap(x)
        logits = self.fc(x, training=training)
        return logits

    def get_config(self):
        # Enables model serialization (optional but useful).
        base = super().get_config()
        base.update(
            {
                "num_classes": self.num_classes,
                "conv1_out": self.conv1_out,
                "conv2_out": self.conv2_out,
                "name": self.name,
            }
        )
        return base


def build_tinycnn(
    input_shape=(6, 50, 1),
    num_classes: int = 2,
) -> tf.keras.Model:
    """
    Utility to build the model with an explicit input shape.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")
    model = TinyCNN(num_classes=num_classes)
    outputs = model(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="TinyCNN")
