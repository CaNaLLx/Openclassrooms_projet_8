import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

N_CLASSES = 8
CLASSES = ["flat", "human", "vehicle", "construction",
           "object", "nature", "sky", "void"]


def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coef Keras. y_true = entiers, y_pred = probas (softmax)."""
    y_true = tf.cast(y_true, tf.int32)
    y_true_oh = tf.one_hot(tf.squeeze(y_true, -1) if y_true.shape[-1] == 1
                           else y_true, N_CLASSES)
    y_true_oh = tf.reshape(y_true_oh, [-1, N_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, N_CLASSES])

    inter = tf.reduce_sum(y_true_oh * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_oh + y_pred_f, axis=0)
    dice = (2. * inter + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def mean_iou_keras(y_true, y_pred, smooth=1e-6):
    """Mean IoU Keras."""
    y_true = tf.cast(y_true, tf.int32)
    y_true_oh = tf.one_hot(tf.squeeze(y_true, -1) if y_true.shape[-1] == 1
                           else y_true, N_CLASSES)
    y_true_oh = tf.reshape(y_true_oh, [-1, N_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, N_CLASSES])

    inter = tf.reduce_sum(y_true_oh * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_oh + y_pred_f, axis=0) - inter
    iou = (inter + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def combined_loss(y_true, y_pred):
    """Cross Entropy + Dice (souvent meilleur pour la segmentation)."""
    cce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cce) + dice_loss(y_true, y_pred)
