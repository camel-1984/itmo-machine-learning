import numpy as np
import pytest
import metrics

@pytest.mark.parametrize('name,expected',[('precision',2/3),('recall',2/3),('fbeta',2/3),('error_cost',2/6)])
def test_confusion_example(name,expected):
    assert getattr(metrics,name)([1,1,1,0,0,0],[1,1,0,1,0,0])==pytest.approx(expected)

@pytest.mark.parametrize('name',['precision','recall','fbeta'])
def test_zero_denominator(name):
    assert getattr(metrics,name)([0,0],[0,0])==0.

def test_beta_and_weights():
    assert metrics.fbeta([1,1,1,0],[1,0,0,1],beta=3)==pytest.approx(10/29)
    assert metrics.error_cost([1,0],[0,1],fp_cost=2,fn_cost=3)==pytest.approx(2.5)

def test_log_loss_and_clipping():
    assert metrics.binary_log_loss([0,1],[.25,.75])==pytest.approx(-np.log(.75))
    assert np.isfinite(metrics.binary_log_loss([0,1],[1,0]))

@pytest.mark.parametrize('name',['precision','recall','fbeta','error_cost','binary_log_loss'])
@pytest.mark.parametrize('y,p',[([],[]),([0,1],[0]),([0,2],[0,1]),([[0,1]],[[0,1]])])
def test_invalid_inputs(name,y,p):
    with pytest.raises(ValueError):getattr(metrics,name)(y,p)
