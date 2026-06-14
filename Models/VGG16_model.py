"""VGG16-U-Net minimal pour segmentation 8 classes."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16


def build_vgg16_unet(input_shape=(256, 512, 3), n_classes=8):
    # Encodeur VGG16 pré-entraîné
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in vgg.layers:
        layer.trainable = False

    # Skip connections
    s1 = vgg.get_layer("block1_conv2").output   # 256x512
    s2 = vgg.get_layer("block2_conv2").output   # 128x256
    s3 = vgg.get_layer("block3_conv3").output   # 64x128
    s4 = vgg.get_layer("block4_conv3").output   # 32x64
    b  = vgg.get_layer("block5_conv3").output   # 16x32 (bottleneck)

    # Décodeur : UpSampling2D à la place de Conv2DTranspose
    def up(x, skip, f):
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return x

    d1 = up(b,  s4, 512)
    d2 = up(d1, s3, 256)
    d3 = up(d2, s2, 128)
    d4 = up(d3, s1, 64)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d4)
    return Model(vgg.input, outputs, name="VGG16_UNet")
