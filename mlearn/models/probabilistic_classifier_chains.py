"""
Contributed by Kuan-Hao Huang
"""
import copy

import numpy as np

from ..utils import seed_random_state
from .model_wrapper import ModelWrapper

class ProbabilisticClassifierChains():
    """
    References
    ----------
    .. [1] Cheng, Weiwei, Eyke Hüllermeier, and Krzysztof J. Dembczynski. "Bayes
           optimal multilabel classification via probabilistic classifier
           chains." Proceedings of the 27th international conference on machine
           learning (ICML-10). 2010.
    """
    def __init__(self, base_model, cost, n_samples=100, random_state=None):
        self.random_state = seed_random_state(random_state)
        self.base_model = base_model
        self.model = PCCModel(self.base_model, cost, n_samples, random_state)

    def fit(self, X, Y):
        self.model.train(X, Y)

    def predict(self, X):
        return self.model.predict(X)

def pairwise_rankloss(Z, Y): #truth(Z), prediction(Y)
    """
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    #return -(rankloss + tie0 + tie1)
    return (rankloss + tie0 + tie1)

class PCCModel():
    def __init__(self, base_model, cost, n_samples, random_state=None):
        if cost not in ['f1', 'hamming', 'rankloss', 'acc']:
            raise NotImplementedError('cost {} is not implemented'.format(cost))
        self.base_model = base_model
        self.cost = cost
        self.n_samples = n_samples
        self.random_state = seed_random_state(random_state)

        self.clfs = None
        self.K = None

    def init(self, data_y):
        self.K = data_y.shape[1]

    def new_clfs(self, K):
        return [ModelWrapper(copy.deepcopy(self.base_model)) for i in range(K)]

    def predict_prob(self, data_x):
        pred = np.zeros((data_x.shape[0], len(self.clfs)))
        for i in range(len(self.clfs)):
            pred[:, i] = 1.0 - self.clfs[i].predict_proba(np.concatenate((data_x, (pred[:, :i]>0.5).astype(int)), axis=1))[:, 0]
        return pred

    def predict_one(self, x, pb):
        prob = np.repeat(pb, self.n_samples).reshape((pb.shape[0], self.n_samples)).T
        y_sample = (np.random.random((self.n_samples, self.K))<prob).astype(int)
        if self.cost == "rankloss":
            thr = 0.0
            pred = (pb>thr).astype(int)
            p_sample = np.repeat(pred, self.n_samples).reshape((pred.shape[0], self.n_samples)).T
            score = pairwise_rankloss(y_sample, p_sample).mean()
            for p in pb:
                pred = (pb>p).astype(int)
                p_sample = np.repeat(pred, self.n_samples).reshape((pred.shape[0], self.n_samples)).T
                score_t = pairwise_rankloss(y_sample, p_sample).mean()
                if score_t < score:
                    score = score_t
                    thr = p
            return (pb>thr).astype(int)

        elif self.cost == "hamming":
            return (pb>0.5).astype(int)

        elif self.cost == "f1" or self.cost == "acc":
            s_idxs = y_sample.sum(axis=1)
            P = np.zeros((self.K, self.K))
            for i in range(self.K):
                P[i, :] = y_sample[s_idxs==(i+1), :].sum(axis=0)*1.0/self.n_samples

            W = 1.0 / (np.cumsum(np.ones((self.K, self.K)), axis=1) + np.cumsum(np.ones((self.K, self.K)), axis=0))
            F = P*W
            idxs = (-F).argsort(axis=1)
            H = np.zeros((self.K, self.K), dtype=int)
            for i in range(self.K):
                H[i, idxs[i, :i+1]] = 1
            scores = (F*H).sum(axis=1)
            pred = H[scores.argmax(), :]
            # if (s_idxs==0).mean() > 2*scores.max():
            #   pred = np.zeros((self.K, ), dtype=int)
            return pred

    def predict(self, data_x):
        prob = self.predict_prob(data_x)
        pred = np.zeros((data_x.shape[0], self.K), dtype=int)
        for i in range(data_x.shape[0]):
            pred[i, :] = self.predict_one(data_x[i, :], prob[i, :])
        return pred

    def train(self, X, Y):
        self.init(Y)

        self.clfs = self.new_clfs(self.K)
        for i in range(self.K):
            self.clfs[i].fit(np.concatenate((X, Y[:, :i]), axis=1), Y[:, i])
            try:
                self.clfs[i].set_params(n_jobs=1)
            except:
                pass
