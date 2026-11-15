import numpy as np
import pytest
from sklearn.svm import SVC

from svm_optimization.models.kernel import rbf_kernel
from svm_optimization.models.svm import KernelSVM


X = np.array([[-1., -1.], [1., 1.], [-2., -1.], [2., 1.]])
Y = np.array([-1, 1, -1, 1])


def test_rbf_preserves_small_distances_at_large_offsets():
    x = np.array([[1e8, 1e8], [1e8 + 1, 1e8]])
    result = rbf_kernel(x, x, gamma=1.0)
    assert np.allclose(result, [[1, np.exp(-1)], [np.exp(-1), 1]])
    assert np.all(np.linalg.eigvalsh(result) >= -1e-12)


@pytest.mark.parametrize('gamma', [0, -1, np.nan, np.inf, -np.inf])
def test_nonpositive_and_nonfinite_gamma_rejected(gamma):
    with pytest.raises(ValueError, match='gamma'):
        rbf_kernel(X, X, gamma)
    with pytest.raises(ValueError, match='gamma'):
        KernelSVM(kernel='rbf', gamma=gamma).fit(X, Y)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_predict_rejects_nonfinite_values(invalid, kernel):
    model = KernelSVM(kernel=kernel).fit(X, Y)
    with pytest.raises(ValueError, match='NaN|infinity'):
        model.predict([[invalid, 0]])


def test_predict_rejects_wrong_number_of_features():
    model = KernelSVM().fit(X, Y)
    with pytest.raises(ValueError, match='đặc trưng'):
        model.predict([[1, 2, 3]])


def test_failed_refit_cannot_use_inconsistent_fitted_state():
    model = KernelSVM().fit(X, Y)
    scores = model.decision_function(X)
    model.C = -1
    with pytest.raises(ValueError):
        model.fit(X[:2], Y[:2])
    assert np.allclose(model.decision_function(X), scores)
    assert model.X_train_.shape == X.shape


def test_rbf_scores_match_sklearn_on_held_out_points():
    samples = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels = np.array([-1, 1, 1, -1])
    custom = KernelSVM(C=10, kernel='rbf', gamma=2, tol=1e-6, max_passes=30).fit(samples, labels)
    reference = SVC(C=10, kernel='rbf', gamma=2, tol=1e-6).fit(samples, labels)
    points = np.array([[.1, .1], [.1, .9], [.8, .2], [.8, .8]])
    assert np.array_equal(custom.predict(points), reference.predict(points))
    assert np.allclose(custom.decision_function(points), reference.decision_function(points), atol=1e-4)
