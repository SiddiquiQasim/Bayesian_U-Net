import tensorflow as tf
import tensorflow.keras.backend as K
import Metrics

def crossEntropyLoss():

    ce_loss = tf.keras.losses.BinaryCrossentropy()
    return ce_loss


def weightedCrossEntropyLoss(beta):

    '''beta => value can be used to tune false negatives and false
    positives. E.g; If you want to reduce the number of false
    negatives then set beta > 1, similarly to decrease the number
    of false positives, set bata < 1'''
    def wce_loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        b = y_true * beta + (1 - y_true) * (1 - beta)
        ce = tf.keras.losses.BinaryCrossentropy()
        ce = ce(y_true, y_pred)
        loss = - (b * ce)          
        return K.mean(loss)
    return wce_loss


def focalLoss(alpha=0.25, gamma=2):
    '''
    Here, gamma > 0 and when gamma = 1 Focal Loss works like Cross-
    Entropy loss function
    alpha => range from [0,1]
    '''
    def focal_loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = K.flatten(y_pred)        
        y_true = K.flatten(y_true)

        ce = tf.keras.losses.BinaryCrossentropy()
        ce = ce(y_true, y_pred)
        a = y_true * alpha + (1 - y_true) * (1 - alpha)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        return K.mean(a * (1 - pt) ** gamma * ce)
    return focal_loss   


def ioULoss():
    def iou_loss(y_true, y_pred):
        iou = Metrics.ioU()
        return 1 - iou(y_true, y_pred)
    return iou_loss


def diceLoss():
    def dice_loss(y_true, y_pred):
        dice = Metrics.dice()
        return 1 - dice(y_true, y_pred)
    return dice_loss


def tverskyLoss(smooth=1e-7, alpha=0.7):
    def tversky_loss(y_true, y_pred):
        tversky = Metrics.tversky(smooth=1e-7, alpha=0.7)
        return 1 - tversky(y_true, y_pred)
    return tversky_loss


def focal_tverskyLoss(gamma=2):
    def focal_tversky_loss(y_true, y_pred):
        tversky = Metrics.tversky(smooth=1e-7, alpha=0.7)
        return K.pow((1 - tversky(y_true, y_pred)), gamma)
    return focal_tversky_loss


def logCoshDiceLoss():
    def lcDice_loss(y_true, y_pred):
        dL = diceLoss()
        loss = tf.math.log(tf.math.cosh(dL(y_true, y_pred)))
        return loss
    return lcDice_loss
    
def ELBOLoss():
    '''https://www.kaggle.com/piesposito/bayesian-nerual-networks-with-tensorflow-2-0'''
    def ELBO(y_true, y_pred):
        ce = tf.keras.losses.BinaryCrossentropy()
        ce = ce(y_true, y_pred)
        kl = tf.keras.losses.KLDivergence()
        kl = kl(y_true, y_pred)
        loss = tf.reduce_mean(ce + kl)
        return loss
    return ELBO


def negative_loglikelihoodLoss():
    '''https://keras.io/examples/keras_recipes/bayesian_neural_networks/'''
    def negative_loglikelihood(targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)
    return negative_loglikelihood
    
    
def deep_ensemble_regression_nll_loss(sigma_sq, epsilon = 1e-6):
    """
        Regression loss for a Deep Ensemble, using the negative log-likelihood loss.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll_loss(y_true, y_pred):
        return 0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - y_pred) / (sigma_sq + epsilon))

    return nll_loss
