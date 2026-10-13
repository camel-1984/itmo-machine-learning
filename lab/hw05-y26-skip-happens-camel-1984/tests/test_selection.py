import numpy as np
import pandas as pd
import pytest
from selection import permutation_importance,select_features

class FeatureModel:
    def fit(self,*args):raise AssertionError('importance must not refit')
    def predict(self,X):return X['energy'].to_numpy()

def test_permutation_sign_reproducibility_and_no_mutation():
    X=pd.DataFrame({'energy':np.arange(20.),'tempo':np.ones(20)})
    original=X.copy(deep=True);y=X.energy.to_numpy().copy();m=FeatureModel()
    loss=lambda model,x,y:float(np.mean((model.predict(x)-y)**2))
    a=permutation_importance(m,X,y,loss,False,n_repeats=3,seed=8)
    b=permutation_importance(m,X,y,lambda m,x,y:-loss(m,x,y),True,n_repeats=3,seed=8)
    assert set(a)==set(X) and a['energy']>0 and a['tempo']==pytest.approx(0)
    assert a==b==permutation_importance(m,X,y,loss,False,n_repeats=3,seed=8)
    pd.testing.assert_frame_equal(X,original)
    np.testing.assert_array_equal(y,original.energy)

def test_selection_and_ties():
    assert select_features({'z':.3,'a':.3,'b':-.1},2)==['a','z']

@pytest.mark.parametrize('n',[0,4,1.5])
def test_selection_invalid_size(n):
    with pytest.raises(ValueError):select_features({'a':1,'b':0,'c':-1},n)
