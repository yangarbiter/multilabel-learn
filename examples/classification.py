import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from mlearn.models import CSRPE
from mlearn.utils import load_data
from mlearn.criteria import pairwise_f1_score

def main():
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


if __name__ == '__main__':
    main()
