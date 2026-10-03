"""Leakage-controlled grouped modelling components, sklearn 1.6 compatible."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin,clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEED=42
NAMES=['LR','Ridge','Lasso','RF','GB','SVR','AdaB','DT','KNN','MLP','GNB','LDA']
def scaled(est):return Pipeline([('scale',StandardScaler()),('model',est)])
def models():
    return {'LR':scaled(LinearRegression()),'Ridge':scaled(Ridge(alpha=1.0)),
      'Lasso':scaled(Lasso(alpha=.1)),'RF':RandomForestRegressor(n_estimators=100,random_state=SEED),
      'GB':GradientBoostingRegressor(n_estimators=100,random_state=SEED),
      'SVR':scaled(SVR(C=1.,kernel='rbf',gamma='scale')),
      'AdaB':AdaBoostRegressor(n_estimators=100,random_state=SEED),
      'DT':DecisionTreeRegressor(random_state=SEED),'KNN':scaled(KNeighborsRegressor(n_neighbors=5)),
      'MLP':scaled(MLPRegressor(hidden_layer_sizes=(64,),max_iter=2000,random_state=SEED)),
      'GNB':scaled(GaussianNB()),'LDA':scaled(LinearDiscriminantAnalysis())}

class FoldScreen(TransformerMixin,BaseEstimator):
    """Learn means, drop constants and supervised collinearity reps in fit only."""
    def __init__(self,threshold=.9,fixed_indices=None):
        self.threshold=threshold;self.fixed_indices=fixed_indices
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);self.n_features_in_=X.shape[1]
        self.means_=np.nanmean(np.where(np.isfinite(X),X,np.nan),axis=0)
        assert np.isfinite(self.means_).all()
        X=np.where(np.isfinite(X),X,self.means_)
        if self.fixed_indices is not None:
            self.keep_=np.asarray(self.fixed_indices,dtype=int);return self
        active=np.flatnonzero(np.ptp(X,axis=0)>1e-12)
        assert len(active)>0
        corr=np.nan_to_num(np.corrcoef(X[:,active].T))
        if len(active)==1:corr=np.ones((1,1))
        A=np.abs(corr)>=self.threshold
        yc=np.array([abs(np.corrcoef(X[:,j],y)[0,1]) for j in active]);yc=np.nan_to_num(yc)
        seen=set();kept=[]
        for i in range(len(active)):
            if i in seen:continue
            stack=[i];component=[]
            while stack:
                u=stack.pop()
                if u in seen:continue
                seen.add(u);component.append(u)
                for v in range(len(active)):
                    if v!=u and A[u,v] and v not in seen:stack.append(v)
            kept.append(active[max(component,key=lambda j:yc[j])])
        self.keep_=np.array(sorted(kept),dtype=int)
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=float)
        assert X.shape[1]==self.n_features_in_
        return np.where(np.isfinite(X),X,self.means_)[:,self.keep_]

class RawScoreClassifier(ClassifierMixin,BaseEstimator):
    """Expose original regression/probability scores to sigmoid calibration."""
    def __init__(self,estimator):self.estimator=estimator
    def fit(self,X,y):
        self.model_=clone(self.estimator).fit(X,y);self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1]
        return self
    def decision_function(self,X):
        if hasattr(self.model_,'predict_proba'):return self.model_.predict_proba(X)[:,1]
        return self.model_.predict(X)
    def predict(self,X):return (self.decision_function(X)>=.5).astype(int)

def pipeline(name,fixed_indices=None):
    return Pipeline([('screen',FoldScreen(fixed_indices=fixed_indices)),('score',RawScoreClassifier(models()[name]))])

BASELINE_NAMES=[f'Logistic_{penalty}_C{c:g}' for penalty in ['l2','l1'] for c in [.1,1.,10.]]+['RandomForestClassifier']
def baseline_pipeline(name):
    if name=='RandomForestClassifier':
        est=RandomForestClassifier(n_estimators=200,min_samples_leaf=3,random_state=SEED)
    else:
        _,penalty,cs=name.split('_')
        est=scaled(LogisticRegression(C=float(cs[1:]),penalty=penalty,solver='liblinear',max_iter=5000,random_state=SEED))
    return Pipeline([('screen',FoldScreen()),('model',est)])
