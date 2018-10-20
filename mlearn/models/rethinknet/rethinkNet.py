from os.path import join
import threading
import itertools

from mkdir_p import mkdir_p
import numpy as np
from keras.layers import (
    Input,
    Merge,
    Dense,
    Dropout,
    LSTM,
    GRU,
    SimpleRNN,
    Activation,
    RepeatVector,
)
from keras.regularizers import l2, l1
from keras.models import Model
from keras.optimizers import Nadam, Adam
import keras.optimizers.Optimizer
from keras import backend as K
import scipy.sparse as ss

import tensorflow as tf

from sklearn.model_selection import train_test_split
from bistiming import IterTimer, SimpleTimer

from .utils import get_random_state, weighted_binary_crossentropy, \
    get_rnn_unit, w_bin_xentropy
from ..calc_score import (
    reweight_pairwise_f1,
    reweight_pairwise_hamming,
    p_reweight_pairwise_f1,
    sparse_reweight_pairwise_f1,
    sparse_reweight_pairwise_rankloss,
    sparse_reweight_pairwise_acc,
    reweight_pairwise_rankloss,
)


def arch_001(input_shape, n_labels, weight_input_shape, l2w=1e-5, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = Dense(128, kernel_regularizer=regularizer, activation='relu')(x)

    x = get_rnn_unit(rnn_unit, 128, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input


class RethinkNet(object):

    def __init__(self, n_features:int, n_labels:int, scoring_fn,
            architecture:str="arch_001", b:int=3, batch_size:int=256,
            nb_epoches:int=100, reweight='None', optimizer=None,
            random_state=None, predict_period=10):
        self.random_state = get_random_state(random_state)
        self.batch_size = batch_size
        self.b = b
        self.scoring_fn = scoring_fn
        self.predict_period = predict_period

        if reweight in ['balanced', 'None']:
            self.reweight_scoring_fn = None
        elif reweight in ['hw', 'vw']:
            if 'pairwise_hamming' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = reweight_pairwise_hamming
            elif 'pairwise_rankloss' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_rankloss
            elif 'pairwise_acc' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_acc
            elif 'pairwise_f1' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_f1
            else:
                raise ValueError(self.scoring_fn, "not supported")

        self.nb_epoches = nb_epoches
        self.reweight = reweight

        self.n_labels = n_labels
        self.n_features = n_features
        self.input_shape = ((self.b, ) + (n_features, ))
        self.weight_input_shape = ((self.b, self.n_labels, ))
        model, weight_input = \
                globals()[architecture](self.input_shape, self.n_labels,
                        self.weight_input_shape)
        model.summary()
        self.nb_params = int(model.count_params())

        if optimizer is None:
            optimizer = Nadam()
        if not isinstance(optimizer, keras.optimizers.Optimizer):
            raise ValueError("optimizer should be keras.optimizers.Optimizer."
                             "got :", optimizer)

        self.loss = weighted_binary_crossentropy(weight_input)

        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def _prep_X(self, X):
        X = X.toarray()
        return X

    def _prep_Y(self, Y):
        Y = Y.toarray()
        Y = np.repeat(Y[:, np.newaxis, :], self.b, axis=1)
        return Y

    def _prep_weight(self, trn_pred, trnY):
        weight = np.ones((trnY.shape[0], self.b, self.n_labels),
                         dtype='float32')
        i_start = 1
        if 'vw' in self.reweight:
            i_start = 0
        for i in range(i_start, self.b):
            if self.reweight == 'balanced':
                weight[:, i, :] = trnY.astype('float32') * (
                        1. / self.ones_weight - 1.)
                weight[:, i, :] += 1.
            elif self.reweight == 'None':
                pass
            elif self.reweight_scoring_fn in [
                    sparse_reweight_pairwise_acc,
                    sparse_reweight_pairwise_f1,
                    sparse_reweight_pairwise_rankloss]:
                trn_pre = trn_pred[i-1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                weight[:, i, :]  = self.reweight_scoring_fn(
                                        trnY, trn_pre,
                                        use_true=('truth' in self.reweight))
            elif self.reweight_scoring_fn is not None:
                trn_pre = trn_pred[i-1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                w = self.reweight_scoring_fn(
                        trnY,
                        trn_pre.toarray(),
                        use_true=('truth' in self.reweight))
                weight[:, i, :] = np.abs(w[:, :, 0] - w[:, :, 1])
            else:
                raise NotImplementedError()
            weight[:, i, :] *= weight[:, i, :].size / weight[:, i, :].sum()

        return weight.astype('float32')

    def train(self, X, Y, callbacks=[]):
        self.history = []
        nb_epoches = self.nb_epoches
        X = ss.csr_matrix(X).astype('float32')
        Y = ss.csr_matrix(Y).astype(np.int8)

        if self.reweight == 'balanced':
            self.ones_weight = trnY.astype(np.int32).sum() / \
                               trnY.shape[0] / trnY.shape[1]

        trn_pred = []
        for i in range(self.b):
            trn_pred.append(
                ss.csr_matrix((trnX.shape[0], self.n_labels), dtype=np.int8))

        predict_period = self.predict_period
        for epoch_i in range(0, nb_epoches, predict_period):
            input_generator = InputGenerator(
                self, trnX, trnY, trn_pred, shuffle=False,
                batch_size=self.batch_size, random_state=self.random_state)
            #input_generator.next()
            history = self.model.fit_generator(
                input_generator,
                steps_per_epoch=((trnX.shape[0] - 1) // self.batch_size) + 1,
                epochs=epoch_i + predict_period,
                max_q_size=32,
                workers=8,
                pickle_safe=True,
                initial_epoch=epoch_i,
                callbacks=callbacks)

            trn_scores = []

            trn_pred = self.predict_chain(trnX)
            for j in range(self.b):
                trn_scores.append(np.mean(self.scoring_fn(trnY, trn_pred[j])))
            print("[epoch %6d] trn:" % (epoch_i + predict_period), trn_scores)

            self.history.append({
                'epoch_nb': epoch_i,
                'trn_scores': trn_scores,
            })

    def predict_chain(self, X):
        ret = [[] for i in range(self.b)]
        batches = range(X.shape[0] // self.batch_size
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        with IterTimer("Predicting training data", total=len(batches)) as timer:
            for bs in batches:
                timer.update(bs)
                if (bs+1) * self.batch_size > X.shape[0]:
                    batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
                else:
                    batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs+1) * self.batch_size]

                pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])
                pred_chain = pred_chain > 0.5

                for i in range(self.b):
                    ret[i].append(ss.csr_matrix(pred_chain[:, i, :], dtype=np.int8))
        for i in range(self.b):
            ret[i] = ss.vstack(ret[i])
        return ret


    def predict(self, X):
        X = ss.csr_matrix(X)
        pred_chain = self.predict_chain(X)

        if method == 'last':
            pred = pred_chain[-1]

        return pred

    def predict_topk(self, X, k=5):
        ret = np.zeros((self.b, X.shape[0], k), np.float32)
        batches = range(X.shape[0] // self.batch_size \
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        for bs in batches:
            if (bs+1) * self.batch_size > X.shape[0]:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
            else:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs+1) * self.batch_size]

            pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])

            for i in range(self.b):
                ind = np.argsort(pred_chain[:, i, :], axis=1)[:, -k:][:, ::-1]
                ret[i, batch_idx, :] = ind

        return ret


class InputGenerator(object):
    def __init__(self, model, X, Y=None, pred=None, shuffle=False,
            batch_size=256, random_state=None):
        self.model = model
        self.X = X
        self.Y = Y
        self.lock = threading.Lock()
        if random_state is None:
            self.random_state = np.random.RandomState()
        self.index_generator = self._flow_index(X.shape[0], batch_size, shuffle,
                random_state)
        self.dummy_weight = np.ones((batch_size, self.model.b, self.model.n_labels),
                dtype=float)

        self.pred = pred

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _flow_index(self, n, batch_size, shuffle, random_state):
        index = np.arange(n)
        for epoch_i in itertools.count():
            if shuffle:
                random_state.shuffle(index)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                yield epoch_i, index[batch_start: batch_end]

    def next(self):
        with self.lock:
            epoch_i, index_array = next(self.index_generator)
        batch_X = self.X[index_array]
        preped_X = self.model._prep_X(batch_X)

        if self.Y is None:
            return [preped_X, self.dummy_weight]
        else:
            batch_Y = self.Y[index_array]
            preped_Y = self.model._prep_Y(batch_Y)
            pred = [self.pred[j][index_array] for j in range(self.model.b)]

            if self.model.reweight_scoring_fn in [
                    #reweight_pairwise_hamming,
                    sparse_reweight_pairwise_acc,
                    sparse_reweight_pairwise_f1,
                    sparse_reweight_pairwise_rankloss]:
                lbl_weight = self.model._prep_weight(pred, batch_Y)
            else:
                lbl_weight = self.model._prep_weight(pred, preped_Y[:, 0, :])
                return [preped_X, lbl_weight], preped_Y
