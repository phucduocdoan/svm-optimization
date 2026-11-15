import numpy as np
from svm_optimization.math.dual import dual_objective, check_dual_feasibility
from svm_optimization.math.geometry import margin_width, point_to_hyperplane_distance
from svm_optimization.math.kkt import kkt_report
from svm_optimization.math.primal import hinge_losses, soft_margin_objective
from svm_optimization.models.kernel import linear_kernel, rbf_kernel


def toy_problem():
    X=np.array([[-1.,-1.],[1.,1.],[-2.,-1.],[2.,1.]])
    y=np.array([-1.,1.,-1.,1.]); w=np.array([0.5,0.5]); b=0.0
    alpha=np.array([0.25,0.25,0.0,0.0])
    return X,y,w,b,alpha


def test_hand_solution_primal_equals_dual_and_kkt():
    X,y,w,b,alpha=toy_problem(); scores=X@w+b
    xi=np.maximum(0.0,1.0-y*scores); K=linear_kernel(X,X)
    primal=soft_margin_objective(w,y,scores,C=1.0); dual=dual_objective(alpha,y,K)
    assert np.isclose(primal,0.25) and np.isclose(dual,0.25)
    assert np.isclose(primal,dual)
    assert check_dual_feasibility(alpha,y,C=1.0)["feasible"]
    assert kkt_report(X,y,w,b,xi,alpha,C=1.0)["kkt_satisfied"]


def test_geometry_and_hinge_loss():
    X,y,w,b,_=toy_problem()
    assert np.isclose(margin_width(w),2.0/np.sqrt(0.5))
    assert np.isclose(point_to_hyperplane_distance(np.array([1.,1.]),w,b),np.sqrt(2))
    assert np.allclose(hinge_losses(y,X@w),[0.,0.,0.,0.])


def test_kernel_shapes_and_values():
    X=np.array([[0.,0.],[1.,0.]])
    assert np.allclose(linear_kernel(X,X),[[0.,0.],[0.,1.]])
    K=rbf_kernel(X,X,gamma=1.0)
    assert K.shape==(2,2) and np.allclose(np.diag(K),1.0) and K[0,1]<1.0
