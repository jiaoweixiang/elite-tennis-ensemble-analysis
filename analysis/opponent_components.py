"""Technical screening with a separately retained opponent context covariate."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import revision_models as rm

class KeepOpponentScreen(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        X = np.asarray(X, float)
        assert np.isfinite(X[:, -1]).all()
        self.n_features_in_ = X.shape[1]
        self.technical_ = rm.FoldScreen().fit(X[:, :-1], y)
        self.keep_ = np.r_[self.technical_.keep_, X.shape[1] - 1]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        assert X.shape[1] == self.n_features_in_ and np.isfinite(X[:, -1]).all()
        return np.column_stack([self.technical_.transform(X[:, :-1]), X[:, -1]])

def adjusted_pipeline(name):
    return Pipeline([('screen', KeepOpponentScreen()), ('score', rm.RawScoreClassifier(rm.models()[name]))])
