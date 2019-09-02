import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from mlearn.models import CSRPE, RethinkNet
from mlearn.utils import load_data
from mlearn.criteria import pairwise_f1_score, sparse_pairwise_f1_score

def CSRPE_example():
    X, Y = load_data('./examples/data/scene')
    scoring_fn = pairwise_f1_score
    clf = CSRPE(scoring_fn=scoring_fn, base_clf=LogisticRegression(), n_clfs=200)
    train_id, test_id = train_test_split(range(X.shape[0]), test_size=0.5)

    clf.train(X[train_id], Y[train_id])

    train_pred = clf.predict(X[train_id])
    test_pred = clf.predict(X[test_id])

    train_avg_score = np.mean(scoring_fn(Y[train_id], train_pred.astype(int)))
    test_avg_score = np.mean(scoring_fn(Y[test_id], test_pred.astype(int)))
    print(train_avg_score, test_avg_score)

def RethinkNet_example():
    X, Y = load_data('./examples/data/scene')
    scoring_fn = pairwise_f1_score
    clf = RethinkNet(n_features=X.shape[1], n_labels=Y.shape[1],
                     scoring_fn=sparse_pairwise_f1_score)
    train_id, test_id = train_test_split(range(X.shape[0]), test_size=0.5)

    clf.train(X[train_id], Y[train_id])

    train_pred = clf.predict(X[train_id]).todense()
    test_pred = clf.predict(X[test_id]).todense()

    train_avg_score = np.mean(scoring_fn(Y[train_id], train_pred.astype(int)))
    test_avg_score = np.mean(scoring_fn(Y[test_id], test_pred.astype(int)))
    print(train_avg_score, test_avg_score)

def main():
    print("Running CSRPE ...")
    CSRPE_example()
    print("Running RethinkNet ...")
    RethinkNet_example()


if __name__ == '__main__':
    main()
