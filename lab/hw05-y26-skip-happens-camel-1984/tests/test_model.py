import numpy as np
import pandas as pd
import pytest
from model import Model

def make(**kwargs):
    return Model(case='careful',features=['energy'],max_depth=kwargs.pop('max_depth',1),min_samples_leaf=kwargs.pop('min_samples_leaf',1),**kwargs)

def sample():
    return pd.DataFrame({'energy':[.1,.2,.8,.9]},index=[7,2,9,1]),pd.Series([0,0,1,1],index=[7,2,9,1])

def test_simple_split_and_smoothing():
    X,y=sample();m=make(smoothing=1)
    assert m.fit(X,y) is m
    np.testing.assert_allclose(m.predict_proba(X),[.25,.25,.75,.75])
    np.testing.assert_array_equal(m.predict(X),[0,0,1,1])

def test_minimum_leaf_prevents_split():
    X,y=sample();m=make(min_samples_leaf=3,smoothing=0).fit(X,y)
    np.testing.assert_allclose(m.predict_proba(X),.5)

def test_threshold_and_row_order():
    X,y=sample();m=make(smoothing=1,threshold=.75).fit(X,y);query=X.iloc[[3,0,2,1]]
    p=np.asarray(m.predict_proba(query))
    assert p.shape==(4,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.testing.assert_array_equal(m.predict(query),(p>=.75).astype(int))
    np.testing.assert_allclose(p,[.75,.25,.75,.25])

def test_save_load_and_determinism(tmp_path):
    X,y=sample();m=make().fit(X,y);path=tmp_path/'model.bin';m.save(path)
    assert path.is_file()
    np.testing.assert_array_equal(m.predict_proba(X),Model.load(path).predict_proba(X))
    np.testing.assert_array_equal(m.predict(X),Model.load(path).predict(X))
    np.testing.assert_array_equal(m.predict_proba(X),make().fit(X,y).predict_proba(X))

@pytest.mark.parametrize('label',[0,1])
def test_one_class(label):
    X,y=sample();m=make(smoothing=0).fit(X,y*0+label)
    np.testing.assert_allclose(m.predict_proba(X),label)
