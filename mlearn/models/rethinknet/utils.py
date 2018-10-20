
import numpy as np
from keras import backend as K
from keras.layers import (
    LeakyReLU,
    LSTM,
    GRU,
    SimpleRNN,
    Bidirectional,
)
from keras.regularizers import l2
from keras.callbacks import  Callback
from keras import backend as K

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

_EPSILON = 10e-8

def get_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state

def w_bin_xentropy(y_pred, y_true, label_weight):
    return np.mean(-(y_true * np.log(y_pred).clip(-1e10, 1e10) \
            + (1. - y_true) * np.log(1. - y_pred).clip(-1e10, 1e10)))

def weighted_binary_crossentropy(label_weight):
    def binary_crossentropy(y_true, y_pred):
        #weight = K.reshape(label_weight, (K.shape(label_weight)[1], -1))
        weight = label_weight
        return K.mean(weight * K.binary_crossentropy(
                                    output=y_pred, target=y_true), axis=-1)
    return binary_crossentropy

def ex_weighted_binary_crossentropy(label_weight, alpha, B, K):
    def binary_crossentropy(y_true, y_pred):
        #weight = K.reshape(label_weight, (K.shape(label_weight)[1], -1))
        weight = label_weight
        true_log = y_true * K.log(K.clip(y_true, K.epsilon(), None))
        pred_log = y_pred * K.log(K.clip(y_pred, K.epsilon(), None))

        reg = -alpha * (true_log + pred_log)
        for i in range(B):
            reg[:, i*K: (i+1)*K] * (K-i-1)
        reg = K.mean(reg, axis=-1)

        return reg + K.mean(weight * K.binary_crossentropy(
                                    output=y_pred, target=y_true), axis=-1)
    return binary_crossentropy

def reweight_with_scoring_fn(truth, pred, scoring_fn, b):
    for k in range(truth.shape[1]):
        t0 = np.copy(pred[:, j, :])
        t0[:, k] = 0
        t1 = np.copy(pred[:, j, :])
        t1[:, k] = 1

        weight[:, j, k] = \
            np.abs(self.scoring_fn(truth[:, j, :], t0) \
                   - self.scoring_fn(truth[:, j, :], t1))
    weight *= weight.size / weight.sum()
    return weight

def get_rnn_unit(rnn_unit, shape, inputs, l2w,
        activation='sigmoid', recurrent_dropout=0.5, **kwargs):
    regularizer = l2w

    if rnn_unit == 'simplernn':
        outputs = SimpleRNN(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    elif rnn_unit == 'lstm':
        outputs = LSTM(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    elif rnn_unit == 'gru':
        outputs = GRU(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    else:
        raise NotImplementedError
    return outputs
