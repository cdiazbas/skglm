import numpy as np
import numba
from numba import njit, prange
from scipy import sparse
from sklearn.utils import check_array
from skglm.solvers.common import (
    construct_grad, construct_grad_sparse, dist_fix_point_cd
)
from skglm.solvers.base import BaseSolver
from skglm.utils.anderson import AndersonAcceleration
from skglm.utils.validation import check_attrs


class AndersonCD(BaseSolver):
    """Coordinate descent solver with working sets and Anderson acceleration.

    fit_intercept : bool
        Whether or not to fit an intercept.

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set).

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    References
    ----------
    .. [1] Bertrand, Q. and Klopfenstein, Q. and Bannier, P.-A. and Gidel, G.
           and Massias, M.
           "Beyond L1: Faster and Better Sparse Models with skglm", 2022
           https://arxiv.org/abs/2204.07826

    .. [2] Bertrand, Q. and Massias, M.
           "Anderson acceleration of coordinate descent", AISTATS, 2021
           https://proceedings.mlr.press/v130/bertrand21a.html
           code: https://github.com/mathurinm/andersoncd
    """

    _datafit_required_attr = ("get_lipschitz", "gradient_scalar")
    _penalty_required_attr = ("prox_1d",)

    def __init__(self, max_iter=50, max_epochs=50_000, p0=10,
                 tol=1e-4, ws_strategy="subdiff", fit_intercept=True,
                 warm_start=False, verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def _solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        if self.ws_strategy not in ("subdiff", "fixpoint"):
            raise ValueError(
                'Unsupported value for self.ws_strategy:', self.ws_strategy)
        n_samples, n_features = X.shape
        w = np.zeros(n_features + self.fit_intercept) if w_init is None else w_init
        Xw = np.zeros(n_samples) if Xw_init is None else Xw_init
        pen = penalty.is_penalized(n_features)
        unpen = ~pen
        n_unpen = unpen.sum()
        obj_out = []
        all_feats = np.arange(n_features)
        stop_crit = np.inf  # initialize for case n_iter=0
        w_acc, Xw_acc = np.zeros(n_features + self.fit_intercept), np.zeros(n_samples)

        is_sparse = sparse.issparse(X)
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
            lipschitz = datafit.get_lipschitz_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)
            lipschitz = datafit.get_lipschitz(X, y)

        if len(w) != n_features + self.fit_intercept:
            if self.fit_intercept:
                val_error_message = (
                    "w should be of size n_features + 1 when using fit_intercept=True: "
                    f"expected {n_features + 1}, got {len(w)}.")
            else:
                val_error_message = (
                    "w should be of size n_features: "
                    f"expected {n_features}, got {len(w)}.")
            raise ValueError(val_error_message)

        for t in range(self.max_iter):
            if is_sparse:
                grad = datafit.full_grad_sparse(
                    X.data, X.indptr, X.indices, y, Xw)
            else:
                grad = construct_grad(X, y, w[:n_features], Xw, datafit, all_feats)

            # The intercept is not taken into account in the optimality conditions since
            # the derivative w.r.t. to the intercept may be very large. It is not likely
            # to change significantly the optimality conditions.
            # TODO: MM I don't understand the comment above: the intercept is
            # taken into account intercept_opt 6 lines below
            if self.ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(w[:n_features], grad, all_feats)
            elif self.ws_strategy == "fixpoint":
                opt = dist_fix_point_cd(
                    w[:n_features], grad, lipschitz, datafit, penalty, all_feats
                )

            if self.fit_intercept:
                intercept_opt = np.abs(datafit.intercept_update_step(y, Xw))
            else:
                intercept_opt = 0.

            stop_crit = max(np.max(opt), intercept_opt)

            if self.verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            if stop_crit <= self.tol:
                break
            # 1) select features : all unpenalized, + 2 * (nnz and penalized)
            ws_size = max(min(self.p0 + n_unpen, n_features),
                          min(2 * penalty.generalized_support(w[:n_features]).sum() -
                              n_unpen, n_features))

            opt[unpen] = np.inf  # always include unpenalized features
            opt[penalty.generalized_support(w[:n_features])] = np.inf

            # here use topk instead of np.argsort(opt)[-ws_size:]
            ws = np.argpartition(opt, -ws_size)[-ws_size:]

            # re init AA at every iter to consider ws
            accelerator = AndersonAcceleration(K=5)
            w_acc[:] = 0.
            # ws to be used in AndersonAcceleration
            ws_intercept = np.append(ws, -1) if self.fit_intercept else ws

            if self.verbose:
                print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

            # 2) do iterations on smaller problem
            is_sparse = sparse.issparse(X)
            for epoch in range(self.max_epochs):
                if is_sparse:
                    _cd_epoch_sparse(
                        X.data, X.indptr, X.indices, y, w[:n_features], Xw,
                        lipschitz, datafit, penalty, ws)
                else:
                    _cd_epoch(
                        X, y, w[:n_features], Xw, lipschitz, datafit, penalty, ws
                    )

                # update intercept
                if self.fit_intercept:
                    intercept_old = w[-1]
                    w[-1] -= datafit.intercept_update_step(y, Xw)
                    Xw += (w[-1] - intercept_old)

                # 3) do Anderson acceleration on smaller problem
                w_acc[ws_intercept], Xw_acc[:], is_extrap = accelerator.extrapolate(
                    w[ws_intercept], Xw)

                if is_extrap:  # avoid computing p_obj for un-extrapolated w, Xw
                    # TODO : manage penalty.value(w, ws) for weighted Lasso
                    p_obj = (datafit.value(y, w[:n_features], Xw) +
                             penalty.value(w[:n_features]))
                    p_obj_acc = (datafit.value(y, w_acc[:n_features], Xw_acc) +
                                 penalty.value(w_acc[:n_features]))

                    if p_obj_acc < p_obj:
                        w[:], Xw[:] = w_acc, Xw_acc
                        p_obj = p_obj_acc

                if epoch % 10 == 0:
                    if is_sparse:
                        grad_ws = construct_grad_sparse(
                            X.data, X.indptr, X.indices, y, w, Xw, datafit, ws)
                    else:
                        grad_ws = construct_grad(X, y, w, Xw, datafit, ws)
                    if self.ws_strategy == "subdiff":
                        opt_ws = penalty.subdiff_distance(w[:n_features], grad_ws, ws)
                    elif self.ws_strategy == "fixpoint":
                        opt_ws = dist_fix_point_cd(
                            w[:n_features], grad_ws, lipschitz[ws], datafit, penalty, ws
                        )

                    stop_crit_in = np.max(opt_ws)
                    if max(self.verbose - 1, 0):
                        p_obj = (datafit.value(y, w[:n_features], Xw) +
                                 penalty.value(w[:n_features]))
                        print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                              f"stopping crit {stop_crit_in:.2e}")
                    if ws_size == n_features:
                        if stop_crit_in <= self.tol:
                            break
                    else:
                        if stop_crit_in < 0.3 * stop_crit:
                            if max(self.verbose - 1, 0):
                                print("Early exit")
                            break
            p_obj = datafit.value(y, w[:n_features], Xw) + penalty.value(w[:n_features])
            obj_out.append(p_obj)
        return w, np.array(obj_out), stop_crit

    def _solve_multi(self, X, Y_Kn, alpha):
        """Solve K Lasso problems simultaneously using block coordinate descent.

        Reads each column of X **once** per CD epoch regardless of the number
        of targets, then parallelises the per-target gradient / prox / update
        steps with Numba ``prange``.

        Parameters
        ----------
        X : array, shape (n_samples, n_features), F-contiguous
            Design matrix.  Must already be validated (dtype, order).

        Y_Kn : array, shape (K, n_samples), C-contiguous
            Targets stored as *rows* for cache-friendly per-target access.

        alpha : float
            L1 penalty strength.

        Returns
        -------
        W : array, shape (n_features, K)
            Coefficient matrix.
        obj_out : ndarray
            Stop-criterion value recorded at the end of each outer iteration.
        stop_crit : float
            Final stopping criterion value.
        """
        n_samples, n_features = X.shape
        K = Y_Kn.shape[0]

        W = np.zeros((n_features, K), dtype=X.dtype)
        XW_Kn = np.zeros((K, n_samples), dtype=X.dtype)  # (K, n) contiguous rows

        # Lipschitz constants: L[j] = ||X[:, j]||² / n  (same for all targets)
        lipschitz = np.einsum('ij,ij->j', X, X) / n_samples  # (n_features,)

        stop_crit = np.inf
        obj_out = []

        try:
            from tqdm.auto import tqdm as _tqdm
            _have_tqdm = True
        except ImportError:
            _have_tqdm = False

        outer_range = range(self.max_iter)
        if self.verbose and _have_tqdm:
            outer_range = _tqdm(
                outer_range,
                desc=f"MultiLasso K={K} p={n_features}",
                unit="iter",
                dynamic_ncols=True,
            )

        for t in outer_range:
            # ---- Working-set selection: max |grad_k| across all K targets ----
            # grad_k[j] = X[:,j].T @ (XW_k - Y_k) / n  -- shape (K, p)
            R_Kn = XW_Kn - Y_Kn                               # (K, n)
            G_Kp = R_Kn @ X / n_samples                       # (K, p)
            # opt[j] = max_k subdiff distance for feature j
            grad_max = np.max(np.abs(G_Kp), axis=0)           # (p,)
            opt = np.maximum(0.0, grad_max - alpha) / alpha
            stop_crit = float(np.max(opt))

            if self.verbose and _have_tqdm:
                outer_range.set_postfix(stop_crit=f"{stop_crit:.2e}",
                                        ws=0, refresh=False)
            elif self.verbose:
                print(f"[solve_multi] outer iter {t + 1}, "
                      f"stop_crit={stop_crit:.2e}")

            if stop_crit <= self.tol:
                break

            # Support: any non-zero across all K targets
            support = np.any(W != 0, axis=1)           # (p,) bool
            ws_size = max(
                min(self.p0, n_features),
                min(2 * int(support.sum()), n_features),
            )
            opt_sel = opt.copy()
            opt_sel[support] = np.inf   # always keep support in WS
            ws = np.argpartition(opt_sel, -ws_size)[-ws_size:].astype(np.int32)

            if self.verbose and _have_tqdm:
                outer_range.set_postfix(stop_crit=f"{stop_crit:.2e}",
                                        ws=ws_size, refresh=True)

            # ---- Inner epochs ---------------------------------------------------
            for epoch in range(self.max_epochs):
                _block_cd_epoch_multi(X, Y_Kn, W, XW_Kn, lipschitz, ws, alpha)

                if epoch % 10 == 0:
                    Rws_Kn = XW_Kn - Y_Kn                       # (K, n)
                    Gws_Kp = Rws_Kn @ X[:, ws] / n_samples      # (K, ws_size)
                    gmax_ws = np.max(np.abs(Gws_Kp), axis=0)    # (ws_size,)
                    opt_ws = np.maximum(0.0, gmax_ws - alpha) / alpha
                    stop_in = float(np.max(opt_ws))

                    if max(self.verbose - 1, 0):
                        print(f"  epoch {epoch + 1}, inner stop={stop_in:.2e}")

                    if ws_size == n_features:
                        if stop_in <= self.tol:
                            break
                    else:
                        if stop_in < 0.3 * stop_crit:
                            break

            obj_out.append(stop_crit)

        return W, np.array(obj_out), stop_crit

    def path(self, X, y, datafit, penalty, alphas=None, w_init=None,
             return_n_iter=False):
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=False, accept_large_sparse=False)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)
        if sparse.issparse(X):
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)
        n_features = X.shape[1]
        if alphas is None:
            raise ValueError('alphas should be passed explicitly')

        n_alphas = len(alphas)
        coefs = np.zeros((n_features + self.fit_intercept, n_alphas), order='F',
                         dtype=X.dtype)
        stop_crits = np.zeros(n_alphas)
        p0 = self.p0

        if return_n_iter:
            n_iters = np.zeros(n_alphas, dtype=int)

        for t in range(n_alphas):
            alpha = alphas[t]
            penalty.alpha = alpha
            if self.verbose:
                to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
                print("#" * len(to_print))
                print(to_print)
                print("#" * len(to_print))
            if t > 0:
                w = coefs[:, t - 1].copy()
                # TODO tmp fix debug for L05:  p0 > replace by 1 (?)
                p0 = max(np.sum(penalty.generalized_support(w)), p0)
            else:
                if w_init is not None:
                    w = w_init.copy()
                    supp_size = penalty.generalized_support(w[:n_features]).sum()
                    p0 = max(supp_size, p0)
                    if supp_size:
                        Xw = X @ w[:n_features] + self.fit_intercept * w[-1]
                    # TODO explain/clean this hack
                    else:
                        Xw = np.zeros_like(y)
                else:
                    w = np.zeros(n_features + self.fit_intercept, dtype=X.dtype)
                    Xw = np.zeros(X.shape[0], dtype=X.dtype)

            sol = self.solve(X, y, datafit, penalty, w, Xw)

            coefs[:, t] = sol[0]
            stop_crits[t] = sol[-1]

            if return_n_iter:
                n_iters[t] = len(sol[1])

        results = alphas, coefs, stop_crits
        if return_n_iter:
            results += (n_iters,)
        return results

    def custom_checks(self, X, y, datafit, penalty):
        # check datafit support sparse data
        check_attrs(
            datafit, solver=self,
            required_attr=self._datafit_required_attr,
            support_sparse=sparse.issparse(X)
        )

        # ws strategy
        if self.ws_strategy == "subdiff" and not hasattr(penalty, "subdiff_distance"):
            raise AttributeError(
                "Penalty must implement `subdiff_distance` "
                "to use ws_strategy='subdiff' in solver AndersonCD."
            )


@njit
def _cd_epoch(X, y, w, Xw, lc, datafit, penalty, ws):
    """Run an epoch of coordinate descent in place.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    lc : array, shape (n_features,)
        Coordinatewise gradient Lipschitz constants.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    ws : array, shape (ws_size,)
        The range of features.
    """
    for j in ws:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = penalty.prox_1d(
            old_w_j - datafit.gradient_scalar(X, y, w, Xw, j) * stepsize,
            stepsize, j)
        if w[j] != old_w_j:
            Xw += (w[j] - old_w_j) * Xj


@njit
def _cd_epoch_sparse(X_data, X_indptr, X_indices, y, w, Xw, lc, datafit, penalty, ws):
    """Run an epoch of coordinate descent in place for a sparse CSC array.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
        `data` attribute of the sparse CSC matrix X.

    X_indptr : array, shape (n_features + 1,)
        `indptr` attribute of the sparse CSC matrix X.

    X_indices : array, shape (n_elements,)
        `indices` attribute of the sparse CSC matrix X.

    y : array, shape (n_samples,)
        Target vector.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    lc :  array, shape (n_features,)
        Coordinatewise gradient Lipschitz constants.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    ws : array, shape (ws_size,)
        The working set.
    """
    for j in ws:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000

        old_w_j = w[j]
        gradj = datafit.gradient_scalar_sparse(X_data, X_indptr, X_indices, y, Xw, j)
        w[j] = penalty.prox_1d(
            old_w_j - gradj * stepsize, stepsize, j)
        diff = w[j] - old_w_j
        if diff != 0:
            for i in range(X_indptr[j], X_indptr[j + 1]):
                Xw[X_indices[i]] += diff * X_data[i]


@njit(parallel=True)
def _block_cd_epoch_multi(X, Y_Kn, W, XW_Kn, lc, ws, alpha):
    """Block CD epoch: update all K targets for each feature in ``ws``.

    ``X[:, j]`` is read from memory **once** and shared across all K targets,
    giving a K-fold reduction in design-matrix memory traffic compared with K
    independent single-target epochs.  The per-target gradient / prox / residual-
    update steps are parallelised over K with Numba ``prange``.

    Parameters
    ----------
    X : F-contiguous array, shape (n_samples, n_features)
        Design matrix.  F-order ensures column j is a contiguous slice.

    Y_Kn : C-contiguous array, shape (K, n_samples)
        Targets stored as rows so ``Y_Kn[k]`` is a contiguous length-n slice.

    W : C-contiguous array, shape (n_features, K)
        Coefficient matrix, updated in-place.

    XW_Kn : C-contiguous array, shape (K, n_samples)
        Running fitted values, updated in-place.

    lc : array, shape (n_features,)
        Coordinatewise Lipschitz constants.

    ws : int32 array, shape (ws_size,)
        Working-set feature indices.

    alpha : float
        L1 penalty strength.
    """
    n_samples = X.shape[0]
    K = Y_Kn.shape[0]

    for idx in range(ws.shape[0]):          # sequential over features (XW_Kn shared)
        j = ws[idx]
        lc_j = lc[j]
        stepsize = 1.0 / lc_j if lc_j != 0.0 else 1000.0
        thresh = alpha * stepsize
        Xj = X[:, j]                        # contiguous column (F-order), loaded once

        # --- Phase 1: gradient + prox for every target (parallel over K) --------
        diffs = np.zeros(K, dtype=Xj.dtype)
        for k in prange(K):
            XW_k = XW_Kn[k]                # contiguous row  (n,)
            Y_k = Y_Kn[k]                  # contiguous row  (n,)
            g = 0.0
            for i in range(n_samples):
                g += Xj[i] * (XW_k[i] - Y_k[i])
            g /= n_samples

            old_w = W[j, k]
            v = old_w - g * stepsize
            if v > thresh:
                new_w = v - thresh
            elif v < -thresh:
                new_w = v + thresh
            else:
                new_w = 0.0
            W[j, k] = new_w
            diffs[k] = new_w - old_w

        # --- Phase 2: update residuals (parallel over K, independent per k) ------
        for k in prange(K):
            d = diffs[k]
            if d != 0.0:
                XW_k = XW_Kn[k]
                for i in range(n_samples):
                    XW_k[i] += d * Xj[i]
