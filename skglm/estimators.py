# License: BSD 3 clause

import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from numbers import Integral, Real

from sklearn.utils.validation import (check_is_fitted, check_array,
                                      check_consistent_length)
from sklearn.linear_model._base import (
    RegressorMixin, LinearModel,
    LinearClassifierMixin, SparseCoefMixin, BaseEstimator
)
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.multiclass import OneVsRestClassifier, check_classification_targets

from skglm.solvers import AndersonCD, MultiTaskBCD, GroupBCD, ProxNewton, LBFGS
from skglm.datafits import (
    Cox, Quadratic, Logistic, QuadraticSVC,
    QuadraticMultiTask, QuadraticGroup, ClippedQuadratic,
    LeakyClippedQuadratic, CensoredQuadratic)
from skglm.penalties import (L1, WeightedL1, L1_plus_L2, L2, WeightedGroupL2,
                             MCPenalty, WeightedMCPenalty, IndicatorBox, L2_1)
from skglm.utils.data import grp_converter
from skglm.utils.jit_compilation import compiled_clone
from sklearn.utils.validation import validate_data


def _glm_fit(X, y, model, datafit, penalty, solver):
    import time
    t_start = time.perf_counter()

    is_classif = isinstance(datafit, (Logistic, QuadraticSVC))
    fit_intercept = solver.fit_intercept

    if is_classif:
        check_classification_targets(y)
        enc = LabelEncoder()
        y = enc.fit_transform(y)
        model.classes_ = enc.classes_
        n_classes_ = len(model.classes_)
        is_sparse = issparse(X)
        if n_classes_ <= 2:
            y = 2 * y - 1
        X = check_array(
            X, accept_sparse="csc", dtype=np.float64, accept_large_sparse=False)
        y = check_array(
            y, ensure_2d=False, dtype=X.dtype.type, accept_large_sparse=False)
        check_consistent_length(X, y)
    else:
        check_X_params = dict(
            dtype=[np.float64, np.float32], order='F',
            accept_sparse='csc', copy=fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')

        X, y = validate_data(
            model, X, y, validate_separately=(check_X_params, check_y_params))
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=False, accept_large_sparse=False)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)

    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn("DataConversionWarning('A column-vector y"
                      " was passed when a 1d array was expected")
        y = y[:, 0]

    if not hasattr(model, "n_features_in_"):
        model.n_features_in_ = X.shape[1]

    n_samples = X.shape[0]
    if n_samples != y.shape[0]:
        raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                         % (n_samples, y.shape[0]))

    # if not model.warm_start or not hasattr(model, "coef_"):
    if not solver.warm_start or not hasattr(model, "coef_"):
        model.coef_ = None

    if is_classif and n_classes_ > 2:
        model.coef_ = np.empty([len(model.classes_), X.shape[1]])
        if isinstance(datafit, QuadraticSVC):
            model.dual_coef_ = np.empty([len(model.classes_), X.shape[0]])
        model.intercept_ = 0
        multiclass = OneVsRestClassifier(model).fit(X, y)
        model.coef_ = np.array(
            [clf.coef_[0] for clf in multiclass.estimators_])
        if isinstance(datafit, QuadraticSVC):
            model.dual_coef_ = np.array(
                [clf.dual_coef_[0] for clf in multiclass.estimators_])
        model.n_iter_ = max(
            clf.n_iter_ for clf in multiclass.estimators_)
        return model

    if is_classif and n_classes_ <= 2 and isinstance(datafit, QuadraticSVC):
        if is_sparse:
            yXT = (X.T).multiply(y)
            yXT = yXT.tocsc()
        else:
            yXT = (X * y[:, None]).T
        X_ = yXT
    else:
        X_ = X

    n_samples, n_features = X_.shape

    # if model.warm_start and hasattr(model, 'coef_') and model.coef_ is not None:
    if solver.warm_start and hasattr(model, 'coef_') and model.coef_ is not None:
        if isinstance(datafit, QuadraticSVC):
            w = model.dual_coef_[0, :].copy()
        elif is_classif:
            w = model.coef_[0, :].copy()
        else:
            w = model.coef_.copy()
        if fit_intercept:
            w = np.hstack([w, model.intercept_])
        Xw = X_ @ w[:w.shape[0] - fit_intercept] + fit_intercept * w[-1]
    else:
        # TODO this should be solver.get_init() do delegate the work
        if y.ndim == 1:
            w = np.zeros(n_features + fit_intercept, dtype=X_.dtype)
            Xw = np.zeros(n_samples, dtype=X_.dtype)
        else:  # multitask
            w = np.zeros((n_features + fit_intercept, y.shape[1]), dtype=X_.dtype)
            Xw = np.zeros(y.shape, dtype=X_.dtype)

    if isinstance(penalty, WeightedL1):
        if len(penalty.weights) != n_features:
            raise ValueError(
                "The size of the WeightedL1 penalty weights should be n_features, "
                "expected %i, got %i." % (X_.shape[1], len(penalty.weights)))

    t_val = time.perf_counter() - t_start
    t0 = time.perf_counter()

    coefs, p_obj, kkt = solver.solve(X_, y, datafit, penalty, w, Xw)
    
    t_solve = time.perf_counter() - t0
    if getattr(model, "verbose_debug", False):
        t_total = time.perf_counter() - t_start
        import sys
        sys.stderr.write(f"[Classic Lasso DEBUG] Validation: {t_val:.3f}s | Solve+Warmup: {t_solve:.3f}s | Total: {t_total:.3f}s\n")
    model.coef_, model.stop_crit_ = coefs[:n_features], kkt
    if y.ndim == 1:
        model.intercept_ = coefs[-1] if fit_intercept else 0.
    else:
        model.intercept_ = coefs[-1, :] if fit_intercept else np.zeros(
            y.shape[1])

    model.n_iter_ = len(p_obj)

    if is_classif and n_classes_ <= 2:
        model.coef_ = coefs[np.newaxis, :n_features]
        if isinstance(datafit, QuadraticSVC):
            if is_sparse:
                primal_coef = ((yXT).multiply(model.coef_[0, :])).T
            else:
                primal_coef = (yXT * model.coef_[0, :]).T
            primal_coef = primal_coef.sum(axis=0)
            model.coef_ = np.array(primal_coef).reshape(1, -1)
            model.dual_coef_ = coefs[np.newaxis, :]
    return model


class GeneralizedLinearEstimator(LinearModel):
    r"""Generic generalized linear estimator.

    This estimator takes a penalty and a datafit and runs a coordinate descent solver
    to solve the optimization problem. It handles classification and regression tasks.

    Parameters
    ----------
    datafit : instance of BaseDatafit, optional
        Datafit. If ``None``, ``datafit`` is initialized as a :class:`.Quadratic`
        datafit.  ``datafit`` is replaced by a JIT-compiled instance when calling fit.

    penalty : instance of BasePenalty, optional
        Penalty. If ``None``, ``penalty`` is initialized as a :class:`.L1` penalty.
        ``penalty`` is replaced by a JIT-compiled instance when
        calling fit.

    solver : instance of BaseSolver, optional
        Solver. If ``None``, ``solver`` is initialized as an :class:`.AndersonCD`
        solver.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_features, n_tasks)
        parameter array (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) or (n_features, n_tasks)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : array, shape (n_tasks,)
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, datafit=None, penalty=None, solver=None):
        super(GeneralizedLinearEstimator, self).__init__()
        self.penalty = penalty
        self.datafit = datafit
        self.solver = solver

    def __repr__(self):
        """Get string representation of the estimator.

        Returns
        -------
        repr : str
            String representation.
        """
        return (
            'GeneralizedLinearEstimator(datafit=%s, penalty=%s, alpha=%s)'
            % (self.datafit.__class__.__name__, self.penalty.__class__.__name__,
               self.penalty.alpha))

    def fit(self, X, y):
        """Fit estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,) or (n_samples, n_tasks)
            Target array.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas) or (n_features, n_tasks, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        # TODO: add support for Cox datafit in `_glm_fit`
        # `_glm_fit` interpret `Cox` as a multitask datafit which breaks down solvers
        if isinstance(self.datafit, Cox):
            raise ValueError(
                "`GeneralizedLinearEstimator` doesn't currently support "
                "`Cox` datafit"
            )

        self.penalty = self.penalty if self.penalty else L1(1.)
        self.datafit = self.datafit if self.datafit else Quadratic()
        self.solver = self.solver if self.solver else AndersonCD()

        return _glm_fit(X, y, self, self.datafit, self.penalty, self.solver)

    def predict(self, X):
        """Predict target values for samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data matrix to predict from.

        Returns
        -------
        y_pred : array, shape (n_samples)
            Contain the target values for each sample.
        """
        if isinstance(self.datafit, (Logistic, QuadraticSVC)):
            scores = self._decision_function(X).ravel()
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]
        else:
            return self.datafit.inverse_link(self._decision_function(X))

    def get_params(self, deep=False):
        """Get parameters of the estimators including the datafit's and penalty's.

        Parameters
        ----------
        deep : bool
            Whether or not return the parameters for contained subobjects estimators.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        params = super().get_params(deep)
        filtered_types = (float, int, str, np.ndarray)
        penalty_params = [('penalty__', p, getattr(self.penalty, p)) for p in
                          dir(self.penalty) if p[0] != "_" and
                          type(getattr(self.penalty, p)) in filtered_types]
        datafit_params = [('datafit__', p, getattr(self.datafit, p)) for p in
                          dir(self.datafit) if p[0] != "_" and
                          type(getattr(self.datafit, p)) in filtered_types]
        for p_prefix, p_key, p_val in penalty_params + datafit_params:
            params[p_prefix + p_key] = p_val
        return params


class Lasso(RegressorMixin, LinearModel):
    r"""Lasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for Lasso is:

    .. math::
        1 / (2 xx n_"samples")  ||y - Xw||_2 ^ 2 + alpha ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``"fixpoint"`` or ``"subdiff"``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    WeightedLasso : Weighted Lasso regularization.
    MCPRegression : Sparser regularization than L1 norm.
    """

    def __init__(self, alpha=1., max_iter=50, max_epochs=50_000, p0=10, verbose=0,
                 tol=1e-4, positive=False, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff", verbose_debug=False):
        super().__init__()
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_debug = verbose_debug

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        # TODO: Add Gram solver
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), L1(self.alpha, self.positive), solver)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Lasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from
            ``coef_init``.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        penalty = L1(self.alpha, self.positive)
        datafit = Quadratic()
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class ClippedLasso(RegressorMixin, LinearModel):
    r"""Lasso estimator with strictly clipped predictions.

    The optimization objective for ClippedLasso is:

    .. math::
        1 / (2 xx n_"samples")  ||y - clip(Xw, a, b)||_2 ^ 2 + alpha ||w||_1

    Parameters
    ----------
    a : float, optional (default=-1.)
        Lower bound of the clipping range.

    b : float, optional (default=1.)
        Upper bound of the clipping range.

    alpha : float, optional (default=1.)
        Penalty strength.

    max_iter : int, optional (default=50)
        The maximum number of iterations (subproblem definitions).

    max_epochs : int (default=50,000)
        Maximum number of CD epochs on each subproblem.

    p0 : int (default=10)
        First working set size.

    verbose : bool or int (default=0)
        Amount of verbosity.

    tol : float, optional (default=1e-4)
        Stopping criterion for the optimization.

    positive : bool, optional (default=False)
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str (default="subdiff")
        The score used to build the working set. Can be ``"fixpoint"`` or ``"subdiff"``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, a=-np.inf, b=np.inf, alpha=1., max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.a = a
        self.b = b
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        datafit = ClippedQuadratic(self.a, self.b)
        penalty = L1(self.alpha, self.positive)
        return _glm_fit(X, y, self, datafit, penalty, solver)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = L1(self.alpha, self.positive)
        datafit = ClippedQuadratic(self.a, self.b)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class LeakyClippedLasso(RegressorMixin, LinearModel):
    r"""Lasso estimator with leaky clipped predictions.

    Parameters
    ----------
    a : float, optional (default=-1.)
        Lower bound of the clipping range.

    b : float, optional (default=1.)
        Upper bound of the clipping range.

    alpha_leak : float, optional (default=0.1)
        Leakage slope outside ``[a, b]``.

    alpha : float, optional (default=1.)
        Penalty strength.

    max_iter : int, optional (default=50)
        The maximum number of iterations (subproblem definitions).

    max_epochs : int (default=50,000)
        Maximum number of CD epochs on each subproblem.

    p0 : int (default=10)
        First working set size.

    verbose : bool or int (default=0)
        Amount of verbosity.

    tol : float, optional (default=1e-4)
        Stopping criterion for the optimization.

    positive : bool, optional (default=False)
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str (default="subdiff")
        The score used to build the working set. Can be ``"fixpoint"`` or ``"subdiff"``.
    """

    def __init__(self, a=-np.inf, b=np.inf, alpha_leak=0.1, alpha=1., max_iter=50,
                 max_epochs=50_000, p0=10, verbose=0, tol=1e-4, positive=False,
                 fit_intercept=True, warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.a = a
        self.b = b
        self.alpha_leak = alpha_leak
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        datafit = LeakyClippedQuadratic(self.a, self.b, self.alpha_leak)
        penalty = L1(self.alpha, self.positive)
        return _glm_fit(X, y, self, datafit, penalty, solver)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = L1(self.alpha, self.positive)
        datafit = LeakyClippedQuadratic(self.a, self.b, self.alpha_leak)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class CensoredLasso(RegressorMixin, LinearModel):
    r"""Lasso estimator for censored data (Tobit regression).

    Parameters
    ----------
    a : float, optional (default=-1.)
        Lower bound of the censoring range.

    b : float, optional (default=1.)
        Upper bound of the censoring range.

    alpha : float, optional (default=1.)
        Penalty strength.

    max_iter : int, optional (default=50)
        The maximum number of iterations (subproblem definitions).

    max_epochs : int (default=50,000)
        Maximum number of CD epochs on each subproblem.

    p0 : int (default=10)
        First working set size.

    verbose : bool or int (default=0)
        Amount of verbosity.

    tol : float, optional (default=1e-4)
        Stopping criterion for the optimization.

    positive : bool, optional (default=False)
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str (default="subdiff")
        The score used to build the working set. Can be ``"fixpoint"`` or ``"subdiff"``.
    """

    def __init__(self, a=-np.inf, b=np.inf, alpha=1., max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.a = a
        self.b = b
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        datafit = CensoredQuadratic(self.a, self.b)
        penalty = L1(self.alpha, self.positive)
        return _glm_fit(X, y, self, datafit, penalty, solver)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = L1(self.alpha, self.positive)
        datafit = CensoredQuadratic(self.a, self.b)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class CachedQuadratic(Quadratic):
    """Quadratic datafit with pre-computed, cached Lipschitz constants.

    This class extends the standard Quadratic datafit by storing a
    pre-computed Lipschitz array, so the solver can skip the O(np)
    recomputation on every fit call.
    """

    def __init__(self, lipschitz):
        self.lipschitz = lipschitz

    def get_spec(self):
        import numba
        # Use numba.typeof to derive the exact array type (float32 or float64)
        # from the actual lipschitz array instance.
        spec = super().get_spec() + (
            ('lipschitz', numba.typeof(self.lipschitz)),
        )
        return spec

    def params_to_dict(self):
        return dict(lipschitz=self.lipschitz)

    def get_lipschitz(self, X, y):
        return self.lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        return self.lipschitz


# Two explicit subclasses give Numba two distinct class identities.
# This prevents JIT cache collisions when switching between float32/float64
# in the same Python session (which would otherwise cause a deadlock).
class CachedQuadratic32(CachedQuadratic):
    """float32-specialized cached quadratic datafit."""
    pass


class CachedQuadratic64(CachedQuadratic):
    """float64-specialized cached quadratic datafit."""
    pass


class LassoFastLoop(RegressorMixin, LinearModel):
    """Zero-overhead Lasso for high-throughput multi-target regression.

    Caches the design matrix, Lipschitz constants, and compiled solver
    kernels across calls to ``fit``. When ``fit`` is called again with the
    same design matrix, all O(np) overhead is skipped entirely.

    Parameters
    ----------
    alpha : float, default=1.0
        L1 regularisation strength.

    max_iter : int, default=50
        Maximum number of working-set iterations.

    max_epochs : int, default=50_000
        Maximum number of CD epochs per working-set subproblem.

    p0 : int, default=10
        Initial working-set size.

    tol : float, default=1e-4
        Convergence tolerance.

    positive : bool, default=False
        Enforce non-negative coefficients.

    fit_intercept : bool, default=True
        Whether to fit an intercept.

    warm_start : bool, default=False
        Reuse the previous solution as the starting point for the next fit.

    ws_strategy : str, default="subdiff"
        Working-set strategy: ``'subdiff'`` or ``'fixpoint'``.

    verbose_debug : bool, default=False
        Print per-call timing (Check / Warmup / Solve) to stderr.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients from the last ``fit`` call.

    intercept_ : float
        Intercept from the last ``fit`` call.
    """

    def __init__(self, alpha=1.0, max_iter=50, max_epochs=50_000, p0=10,
                 tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff", verbose_debug=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.ws_strategy = ws_strategy
        self.verbose_debug = verbose_debug

        # Cache state — populated on first fit, reused when X is the same
        self.X_ = None
        self._X_input_ref = None
        self._datafit = None
        self._penalty = None
        self._solver = None

    def _is_same_X(self, X):
        """O(1) identity check against the cached design matrix."""
        if self._X_input_ref is None:
            return False
        if X is self._X_input_ref:
            return True
        if X.shape != self._X_input_ref.shape or X.dtype != self._X_input_ref.dtype:
            return False
        if issparse(X) and issparse(self._X_input_ref):
            return X.data.ctypes.data == self._X_input_ref.data.ctypes.data
        if not issparse(X) and not issparse(self._X_input_ref):
            return X.ctypes.data == self._X_input_ref.ctypes.data
        return False

    def _warm_up(self, X):
        """Build all X-dependent cache: validate, Lipschitz, JIT-compile."""
        self._X_input_ref = X
        self.X_ = check_array(
            X, accept_sparse="csc", dtype=[np.float64, np.float32], order="F",
            copy=False, accept_large_sparse=False)

        y_dummy = np.zeros(self.X_.shape[0], dtype=self.X_.dtype)
        datafit = Quadratic()
        if issparse(self.X_):
            lipschitz = datafit.get_lipschitz_sparse(
                self.X_.data, self.X_.indptr, self.X_.indices, y_dummy)
        else:
            lipschitz = datafit.get_lipschitz(self.X_, y_dummy)

        # Pick the explicit precision subclass so Numba always gets a unique
        # class identity and never collides between float32 and float64.
        to_float32 = self.X_.dtype == np.float32
        Klass = CachedQuadratic32 if to_float32 else CachedQuadratic64

        self._datafit = compiled_clone(Klass(lipschitz), to_float32=to_float32)
        self._penalty = compiled_clone(L1(self.alpha, self.positive), to_float32=to_float32)
        self._solver = AndersonCD(
            max_iter=self.max_iter, max_epochs=self.max_epochs, p0=self.p0,
            tol=self.tol, ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept, warm_start=self.warm_start,
            verbose=0)

    def fit(self, X, y, assume_same_X=True):
        """Fit the model, reusing cached X-state when possible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.

        y : array-like of shape (n_samples,)
            Target values.

        assume_same_X : {True, False, None}, default=True
            Cache control:

            - ``True``  — always reuse cached state (fastest; safe when X is
              mathematically identical across calls even if it's a new object).
            - ``False`` — always rebuild the cache.
            - ``None``  — auto-detect via O(1) memory-address check.

        Returns
        -------
        self : LassoFastLoop
        """
        import time
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        if assume_same_X is True and self.X_ is not None:
            is_same = True
        elif assume_same_X is False:
            is_same = False
        else:
            is_same = self._is_same_X(X)
        t_check = time.perf_counter() - t0

        t_warmup = 0.0
        if not is_same:
            t0 = time.perf_counter()
            self._warm_up(X)
            t_warmup = time.perf_counter() - t0

        y = np.asanyarray(y, dtype=self.X_.dtype)
        if not y.flags.c_contiguous:
            y = np.ascontiguousarray(y)

        # Initialize w and Xw with the exact dtype of X to avoid Numba precision errors
        n_samples, n_features = self.X_.shape
        n_w = n_features + 1 if self.fit_intercept else n_features
        w = np.zeros(n_w, dtype=self.X_.dtype)
        Xw = np.zeros(n_samples, dtype=self.X_.dtype)

        t0 = time.perf_counter()
        w, _, _ = self._solver._solve(self.X_, y, self._datafit, self._penalty, w, Xw)
        t_solve = time.perf_counter() - t0

        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0.

        if self.verbose_debug:
            t_total = time.perf_counter() - t_start
            if assume_same_X is True and is_same:
                cache_status = "HIT (forced)"
            elif assume_same_X is False:
                cache_status = "MISS (forced)"
            else:
                cache_status = "HIT (auto)" if is_same else "MISS (auto)"
            
            format_str = "sparse" if issparse(self.X_) else "dense"
            shape_str = f"{self.X_.shape[0]}x{self.X_.shape[1]}"
            dtype_str = self.X_.dtype.name
            
            import sys
            sys.stderr.write(
                f"[FastLoop] {dtype_str} {format_str} {shape_str} | tol={self.tol:.0e} | "
                f"Cache: {cache_status} | "
                f"Check: {t_check*1000:.2f}ms | Warmup: {t_warmup:.3f}s | "
                f"Solve: {t_solve:.3f}s | Total: {t_total:.3f}s\n")

        return self





class WeightedLasso(RegressorMixin, LinearModel):
    r"""WeightedLasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for WeightedLasso is:

    .. math::
        1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2 + alpha ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    MCPRegression : Sparser regularization than L1 norm.
    Lasso : Unweighted Lasso regularization.

    Notes
    -----
    Supports weights equal to 0, i.e. unpenalized features.
    """

    def __init__(self, alpha=1., weights=None, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Weighted Lasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        weights = np.ones(X.shape[1]) if self.weights is None else self.weights
        if X.shape[1] != len(weights):
            raise ValueError("The number of weights must match the number of \
                              features. Got %s, expected %s." % (
                len(weights), X.shape[1]))
        penalty = WeightedL1(self.alpha, weights, self.positive)
        datafit = Quadratic()
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        if self.weights is None:
            warnings.warn('Weights are not provided, fitting with Lasso penalty')
            penalty = L1(self.alpha, self.positive)
        else:
            penalty = WeightedL1(self.alpha, self.weights, self.positive)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class ElasticNet(RegressorMixin, LinearModel):
    r"""Elastic net estimator.

    The optimization objective for Elastic net is:

    .. math::
        1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2
        + tt"l1_ratio" xx alpha ||w||_1
        + (1 - tt"l1_ratio") xx alpha/2 ||w||_2 ^ 2

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    Lasso : Lasso regularization.
    """

    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.warm_start = warm_start
        self.verbose = verbose

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Elastic Net path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        penalty = L1_plus_L2(self.alpha, self.l1_ratio, self.positive)
        datafit = Quadratic()
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(),
                        L1_plus_L2(self.alpha, self.l1_ratio, self.positive), solver)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MCPRegression(RegressorMixin, LinearModel):
    r"""Linear regression with MCP penalty estimator.

    The optimization objective for MCPRegression is, with :math:`x >= 0`:

    .. math::
        "pen"(x) = {(alpha x - x^2 / (2 gamma), if x <= alpha gamma),
                    (gamma alpha^2 / 2        , if x > alpha gamma):}

    .. math::
        "obj" = 1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2
              + sum_(j=1)^(n_"features") "pen"(|w_j|)

    For more details see
    `Coordinate descent algorithms for nonconvex penalized regression,
    with applications to biological feature selection, Breheny and Huang
    <https://doi.org/10.1214/10-aoas388>`_.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    gamma : float, default=3
        If ``gamma = 1``, the prox of MCP is a hard thresholding.
        If ``gamma = np.inf`` it is a soft thresholding.
        Should be larger than (or equal to) 1.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    Lasso : Lasso regularization.
    """

    def __init__(self, alpha=1., gamma=3, weights=None, max_iter=50, max_epochs=50_000,
                 p0=10, verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.verbose = verbose
        self.tol = tol
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.ws_strategy = ws_strategy

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute MCPRegression path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm start is enabled, the optimization problem restarts from
            ``coef_init``.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        if self.weights is None:
            penalty = MCPenalty(self.alpha, self.gamma, self.positive)
        else:
            if X.shape[1] != len(self.weights):
                raise ValueError(
                    "The number of weights must match the number of features. "
                    f"Got {len(self.weights)}, expected {X.shape[1]}."
                )
            penalty = WeightedMCPenalty(
                self.alpha, self.gamma, self.weights, self.positive)

        datafit = Quadratic()
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        if self.weights is None:
            penalty = MCPenalty(self.alpha, self.gamma, self.positive)
        else:
            if X.shape[1] != len(self.weights):
                raise ValueError(
                    "The number of weights must match the number of features. "
                    f"Got {len(self.weights)}, expected {X.shape[1]}."
                )
            penalty = WeightedMCPenalty(
                self.alpha, self.gamma, self.weights, self.positive)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class SparseLogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    r"""Sparse Logistic regression estimator.

    The optimization objective for sparse Logistic regression is:

    .. math::
        1 / n_"samples" \sum_{i=1}^{n_"samples"} log(1 + exp(-y_i x_i^T w))
        + tt"l1_ratio" xx alpha ||w||_1
        + (1 - tt"l1_ratio") xx alpha/2 ||w||_2 ^ 2

    By default, ``l1_ratio=1.0`` corresponds to Lasso (pure L1 penalty).
    When ``0 < l1_ratio < 1``, the penalty is a convex combination of L1 and L2
    (i.e., ElasticNet). ``l1_ratio=0.0`` corresponds to Ridge (pure L2), but note
    that pure Ridge is not typically used with this class.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        Only used when ``penalty="l1_plus_l2"``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        ``For l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    tol : float, optional
        Stopping criterion for the optimization.

    max_iter : int, optional
        The maximum number of outer iterations (subproblem definitions).

    max_epochs : int
        Maximum number of prox Newton iterations on each subproblem.

    verbose : bool or int
        Amount of verbosity.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        ``coef_`` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray, shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, alpha=1.0, l1_ratio=1.0, tol=1e-4, max_iter=20, max_epochs=1_000,
                 verbose=0, fit_intercept=True, warm_start=False):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        solver = ProxNewton(
            max_iter=self.max_iter, max_pn_iter=self.max_epochs, tol=self.tol,
            fit_intercept=self.fit_intercept, warm_start=self.warm_start,
            verbose=self.verbose)
        return _glm_fit(X, y, self, Logistic(), L1_plus_L2(self.alpha, self.l1_ratio),
                        solver)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where ``n_samples`` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        if len(self.classes_) > 2:
            # Code taken from https://github.com/scikit-learn/scikit-learn/
            # blob/c900ad385cecf0063ddd2d78883b0ea0c99cd835/sklearn/
            # linear_model/_base.py#L458
            def _predict_proba_lr(X):
                """Probability estimation for OvR logistic regression.

                Positive class probabilities are computed as
                ``1. / (1. + np.exp(-self.decision_function(X)))``;
                multiclass is handled by normalizing that over all classes.
                """
                prob = self.decision_function(X)
                expit(prob, out=prob)
                if prob.ndim == 1:
                    return np.vstack([1 - prob, prob]).T
                else:
                    # OvR normalization, like LibLinear's predict_probability
                    prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
                    return prob
            # OvR normalization, like LibLinear's
            return _predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)


class LinearSVC(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    r"""LinearSVC estimator, with hinge loss.

    The optimization objective for LinearSVC is:

    .. math:: C xx sum_(i=1)^(n_"samples") max(0, 1 - y_i beta^T X[i, :])
        + 1/2 ||beta||_2 ^ 2

    i.e. hinge datafit loss (non-smooth) + l2 regularization (smooth)

    To solve this, we solve the dual optimization problem to stay in our
    framework of smooth datafit and non-smooth penalty.
    The dual optimization problem of SVC is:

    .. math::
        1/2 ||(yX)^T w||_2 ^ 2
        - \sum_(i=1)^(n_"samples") w_i
        + \sum_(i=1)^(n_"samples") [0 <= w_i <= C]

    The primal-dual relation is given by:

    .. math:: w = \sum_(i=1)^(n_"samples") y_i w_i X[i, :]

    Parameters
    ----------
    C : float, optional
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    dual_ : array, shape (n_samples,)
        dual of the solution.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, C=1., max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        super().__init__()
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        """Fit LinearSVC classifier.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        Returns
        -------
        self
            Fitted estimator.
        """
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=False,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, QuadraticSVC(), IndicatorBox(self.C), solver)

    # TODO add predict_proba for LinearSVC


class CoxEstimator(LinearModel):
    r"""Elastic Cox estimator with Efron and Breslow estimate.

    Refer to :ref:`Mathematics behind Cox datafit <maths_cox_datafit>`
    for details about the datafit expression. The data convention for the estimator is

    - ``X`` the design matrix with ``n_features`` predictors
    - ``y`` a two-column array where the first ``tm`` is of event time occurrences
      and the second ``s`` is of censoring.

    For L2-regularized Cox (``l1_ratio=0.``) :ref:`LBFGS <skglm.solvers.LBFGS>`
    is the used solver, otherwise it is :ref:`ProxNewton <skglm.solvers.ProxNewton>`.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength. It must be strictly positive.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    method : {'efron', 'breslow'}, default='efron'
        The estimate used for the Cox datafit. Use ``efron`` to
        handle tied observations.

    tol : float, optional
        Stopping criterion for the optimization.

    max_iter : int, optional
        The maximum number of iterations to solve the problem.

    verbose : bool or int
        Amount of verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Parameter vector of Cox regression.

    stop_crit_ : float
        The value of the stopping criterion at convergence.
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "method": [StrOptions({"efron", "breslow"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["boolean", Interval(Integral, 0, 2, closed="both")],
    }

    def __init__(self, alpha=1., l1_ratio=0.7, method="efron", tol=1e-4,
                 max_iter=50, verbose=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        """Fit Cox estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        y : array-like, shape (n_samples, 2)
            Two-column array where the first is of event time occurrences
            and the second is of censoring. If it is of dimension 1, it is
            assumed to be the times vector and there no censoring.

        Returns
        -------
        self :
            The fitted estimator.
        """
        self._validate_params()

        # validate input data
        X = check_array(
            X,
            accept_sparse="csc",
            order="F",
            dtype=[np.float64, np.float32],
            input_name="X",
        )
        if y is None:
            # Needed to pass check estimator. Message error is
            # copy/paste from https://github.com/scikit-learn/scikit-learn/blob/ \
            # 23ff51c07ebc03c866984e93c921a8993e96d1f9/sklearn/utils/ \
            # estimator_checks.py#L3886
            raise ValueError("requires y to be passed, but the target y is None")
        y = check_array(
            y,
            accept_sparse=False,
            order="F",
            dtype=X.dtype,
            ensure_2d=False,
            input_name="y",
        )
        if y.ndim == 1:
            warnings.warn(
                f"{repr(self)} requires the vector of response `y` to have "
                f"two columns. Got one column.\nAssuming that `y` "
                "is the vector of times and there is no censoring."
            )
            y = np.column_stack((y, np.ones_like(y))).astype(X.dtype, order="F")
        elif y.shape[1] > 2:
            raise ValueError(
                f"{repr(self)} requires the vector of response `y` to have "
                f"two columns. Got {y.shape[1]} columns."
            )

        check_consistent_length(X, y)

        # init datafit and penalty
        datafit = Cox(self.method)

        if self.l1_ratio == 1.:
            penalty = L1(self.alpha)
        elif 0. < self.l1_ratio < 1.:
            penalty = L1_plus_L2(self.alpha, self.l1_ratio)
        else:
            penalty = L2(self.alpha)

        # init solver
        if self.l1_ratio == 0.:
            solver = LBFGS(max_iter=self.max_iter, tol=self.tol, verbose=self.verbose)
        else:
            solver = ProxNewton(
                max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                fit_intercept=False,
            )

        # solve problem
        if not issparse(X):
            datafit.initialize(X, y)
        else:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)

        w, _, stop_crit = solver.solve(X, y, datafit, penalty)

        # save to attribute
        self.coef_ = w
        self.stop_crit_ = stop_crit

        self.intercept_ = 0.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.arange(X.shape[1])

        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MultiTaskLasso(RegressorMixin, LinearModel):
    r"""MultiTaskLasso estimator.

    The optimization objective for MultiTaskLasso is:

    .. math:: 1 / (2 xx n_"samples") ||y - XW||_2 ^ 2 + alpha ||W||_(21)

    Parameters
    ----------
    alpha : float, optional
        Regularization strength (constant that multiplies the L21 penalty).

    copy_X : bool, optional (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.
    """

    def __init__(self, alpha=1., copy_X=True, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        self.tol = tol
        self.alpha = alpha
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.max_epochs = max_epochs
        self.verbose = verbose

    def fit(self, X, Y):
        """Fit MultiTaskLasso model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Design matrix.

        Y : ndarray, shape (n_samples, n_tasks)
            Observation matrix.

        Returns
        -------
        self :
            The fitted estimator.
        """
        # Below is copied from sklearn, with path replaced by our path.
        # Need to validate separately here.
        # We can't pass multi_output=True because that would allow y to be csr.
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              accept_sparse='csc',
                              copy=self.copy_X and self.fit_intercept)
        check_Y_params = dict(ensure_2d=False, order='F')
        X, Y = validate_data(self, X, Y, validate_separately=(check_X_params,
                                                              check_Y_params))
        Y = Y.astype(X.dtype)

        if Y.ndim == 1:
            raise ValueError("For mono-task outputs, use Lasso")

        n_samples = X.shape[0]

        if n_samples != Y.shape[0]:
            raise ValueError("X and Y have inconsistent dimensions (%d != %d)"
                             % (n_samples, Y.shape[0]))

        # X, Y, X_offset, Y_offset, X_scale = _preprocess_data(
        #     X, Y, self.fit_intercept, copy=False)

        # TODO handle and test warm start for MTL
        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        datafit = QuadraticMultiTask()
        penalty = L2_1(self.alpha)

        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        W, obj_out, kkt = solver.solve(X, Y, datafit, penalty)

        self.coef_ = W[:X.shape[1], :].T
        self.intercept_ = self.fit_intercept * W[-1, :]
        self.stopping_crit = kkt
        self.n_iter_ = len(obj_out)

        return self

    def path(self, X, Y, alphas, coef_init=None, return_n_iter=False, **params):
        """Compute MultitaskLasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm start is enabled, the optimization problem restarts from
            ``coef_init``.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_tasks, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        datafit = QuadraticMultiTask()
        penalty = L2_1(self.alpha)
        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, Y, datafit, penalty, alphas, coef_init, return_n_iter)


class GroupLasso(RegressorMixin, LinearModel):
    r"""GroupLasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for GroupLasso is:

    .. math::
        1 / (2 xx n_"samples") ||y - X w||_2 ^ 2 + alpha \sum_g
        weights_g ||w_{[g]}||_2

    with :math:`w_{[g]}` the coefficients of the g-th group.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints
        Partition of features used in the penalty on ``w``.
        If an int is passed, groups are contiguous blocks of features, of size
        ``groups``.
        If a list of ints is passed, groups are assumed to be contiguous,
        group number ``g`` being of size ``groups[g]``.
        If a list of lists of ints is passed, ``groups[g]`` contains the
        feature indices of the group number ``g``.

    alpha : float, optional
        Penalty strength.

    weights : array, shape (n_groups,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

    max_iter : int, optional (default=50)
        The maximum number of iterations (subproblem definitions).

    max_epochs : int, optional (default=50_000)
        Maximum number of CD epochs on each subproblem.

    p0 : int, optional (default=10)
        First working set size.

    verbose : bool or int, optional (default=0)
        Amount of verbosity.

    tol : float, optional (default=1e-4)
        Stopping criterion for the optimization.

    positive : bool, optional (defautl=False)
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str, optional (default="subdiff")
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    Notes
    -----
    Supports weights equal to ``0``, i.e. unpenalized features.
    """

    def __init__(self, groups, alpha=1., weights=None, max_iter=50, max_epochs=50_000,
                 p0=10, verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.groups = groups
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to ``X``.

        Returns
        -------
        self : Instance of GroupLasso
            Fitted estimator.
        """
        grp_indices, grp_ptr = grp_converter(self.groups, X.shape[1])
        group_sizes = np.diff(grp_ptr)

        n_features = np.sum(group_sizes)
        if X.shape[1] != n_features:
            raise ValueError(
                "The total number of group members must equal the number of features. "
                f"Got {n_features}, expected {X.shape[1]}.")

        weights = np.ones(len(group_sizes)) if self.weights is None else self.weights
        group_penalty = WeightedGroupL2(alpha=self.alpha, grp_ptr=grp_ptr,
                                        grp_indices=grp_indices, weights=weights,
                                        positive=self.positive)
        quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
        solver = GroupBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            fit_intercept=self.fit_intercept, warm_start=self.warm_start,
            verbose=self.verbose)

        return _glm_fit(X, y, self, quad_group, group_penalty, solver)
