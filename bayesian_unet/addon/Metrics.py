import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def ioU():
    def cal_iou(y_true, y_pred):
        y_pred = y_pred[..., 0][..., np.newaxis]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)
        iou = tf.math.divide(K.sum(y_true * y_pred) + 0, K.sum(y_true) + K.sum(y_pred) - K.sum(y_true * y_pred) + 0)        
        return K.mean(iou)
    return cal_iou


def dice():
    def cal_dice(y_true, y_pred):
        y_pred = y_pred[..., 0][..., np.newaxis]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)
        dice = tf.math.divide(2 * K.sum(y_true * y_pred) + 0, K.sum(y_true) + K.sum(y_pred) + 0)       
        return K.mean(dice)
    return cal_dice

def tversky(smooth=1e-7, alpha=0.7):
    def tversky_cal(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    return tversky_cal
        

def mAP(threshold=0.5):  
    def cal_map( y_true, y_pred):
        y_pred = y_pred[..., 0][..., np.newaxis]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        pos = tf.math.reduce_sum(y_pred, axis=[1,2])
        mask_p = tf.math.greater(pos, 0.00)
        p = len(tf.boolean_mask(pos, mask_p))
        iou = tf.math.divide(tf.math.reduce_sum(y_true * y_pred, axis=[1,2]), 
                             tf.math.reduce_sum(y_true, axis=[1,2]) + tf.math.reduce_sum(y_pred, axis=[1,2]) - tf.math.reduce_sum(y_true*y_pred, axis=[1,2]))
        mask_iou = tf.math.greater(iou, threshold)
        tp = len(tf.boolean_mask(iou, mask_iou))

        return tp / p
    return cal_map
