import numpy as np
from svm_optimization.models.svm import KernelSVM


def test_linear_svm_learns_toy_problem():
    X=np.array([[-1.,-1.],[1.,1.],[-2.,-1.],[2.,1.]])
    y=np.array([-1.,1.,-1.,1.])
    model=KernelSVM(C=1.0,kernel="linear",tol=1e-6,max_passes=20).fit(X,y)
    assert np.array_equal(model.predict(X),y)
    assert np.all(model.alpha_>=-1e-8) and np.all(model.alpha_<=1.0+1e-8)
    assert abs(np.dot(model.alpha_,y))<1e-6
    assert np.isclose(model.optimizer_.objective_value_,0.25,atol=1e-4)


def test_rbf_svm_learns_xor():
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y=np.array([-1.,1.,1.,-1.])
    model=KernelSVM(C=10.0,kernel="rbf",gamma=2.0,tol=1e-4,max_passes=20).fit(X,y)
    assert np.array_equal(model.predict(X),y)
