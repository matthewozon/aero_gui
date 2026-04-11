"""
modules/param_estimation.py
---------------------------
Parameter estimation methods: fit instrument or model parameters to data.

These are stubs matching the interface of your Julia implementations.
Each estimator exposes fit(A, counts, **kwargs) → ParameterResult.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from scipy.optimize import minimize, differential_evolution


# ============================================================
#  Result container
# ============================================================

@dataclass
class ParameterResult:
    params_estimated: dict          # name → value
    residual: float
    converged: bool
    n_iterations: int
    info: dict = field(default_factory=dict)


# ============================================================
#  Base class
# ============================================================

class ParameterEstimator:
    name: str = "Base"

    def fit(self, forward_model: Callable,
            observations: np.ndarray,
            initial_params: dict,
            bounds: dict = None) -> ParameterResult:
        raise NotImplementedError


# ============================================================
#  Method 1 – Least-squares (Levenberg-Marquardt via scipy)
# ============================================================

class LeastSquaresEstimator(ParameterEstimator):
    """
    Minimises sum of squared residuals between forward_model(params) and observations.
    Uses scipy.optimize.minimize with the L-BFGS-B method.
    """

    name = "Least Squares (L-BFGS-B)"

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, forward_model: Callable,
            observations: np.ndarray,
            initial_params: dict,
            bounds: dict = None) -> ParameterResult:

        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[k] for k in param_names])

        if bounds:
            scipy_bounds = [bounds.get(k, (None, None)) for k in param_names]
        else:
            scipy_bounds = None

        def objective(x):
            params = {k: x[i] for i, k in enumerate(param_names)}
            predicted = forward_model(params)
            return np.sum((predicted - observations) ** 2)

        result = minimize(objective, x0, method="L-BFGS-B",
                          bounds=scipy_bounds,
                          options={"maxiter": self.max_iter, "ftol": self.tol})

        estimated = {k: result.x[i] for i, k in enumerate(param_names)}
        residual = float(np.sqrt(result.fun / len(observations)))

        return ParameterResult(
            params_estimated=estimated,
            residual=residual,
            converged=result.success,
            n_iterations=result.nit,
            info={"message": result.message}
        )


# ============================================================
#  Method 2 – Maximum Likelihood (Poisson noise model)
# ============================================================

class PoissonMLEEstimator(ParameterEstimator):
    """
    Maximises Poisson log-likelihood:
      L = sum_i [ y_i * log(mu_i) - mu_i ]
    where mu_i = forward_model(params)[i].
    Suitable for particle count data.
    """

    name = "Maximum Likelihood (Poisson)"

    def __init__(self, max_iter: int = 500):
        self.max_iter = max_iter

    def fit(self, forward_model: Callable,
            observations: np.ndarray,
            initial_params: dict,
            bounds: dict = None) -> ParameterResult:

        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[k] for k in param_names])

        if bounds:
            scipy_bounds = [bounds.get(k, (1e-10, None)) for k in param_names]
        else:
            scipy_bounds = [(1e-10, None)] * len(param_names)

        def neg_log_likelihood(x):
            params = {k: x[i] for i, k in enumerate(param_names)}
            mu = forward_model(params)
            mu = np.maximum(mu, 1e-30)
            y = observations
            return -np.sum(y * np.log(mu) - mu)

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          bounds=scipy_bounds,
                          options={"maxiter": self.max_iter})

        estimated = {k: result.x[i] for i, k in enumerate(param_names)}
        params_est = {k: result.x[i] for i, k in enumerate(param_names)}
        mu_final = forward_model(params_est)
        residual = float(np.linalg.norm(mu_final - observations) /
                         (np.linalg.norm(observations) + 1e-30))

        return ParameterResult(
            params_estimated=estimated,
            residual=residual,
            converged=result.success,
            n_iterations=result.nit,
            info={"neg_log_likelihood": float(result.fun)}
        )


# ============================================================
#  Method 3 – Differential Evolution (global optimiser)
# ============================================================

class DifferentialEvolutionEstimator(ParameterEstimator):
    """
    Global parameter search using differential evolution.
    Requires bounds for all parameters.
    Best for non-convex problems.
    """

    name = "Differential Evolution (global)"

    def __init__(self, max_iter: int = 200, popsize: int = 15, tol: float = 1e-5):
        self.max_iter = max_iter
        self.popsize = popsize
        self.tol = tol

    def fit(self, forward_model: Callable,
            observations: np.ndarray,
            initial_params: dict,
            bounds: dict = None) -> ParameterResult:

        if bounds is None:
            raise ValueError("DifferentialEvolution requires bounds for all parameters.")

        param_names = list(initial_params.keys())
        scipy_bounds = [bounds[k] for k in param_names]

        def objective(x):
            params = {k: x[i] for i, k in enumerate(param_names)}
            predicted = forward_model(params)
            return np.sum((predicted - observations) ** 2)

        result = differential_evolution(
            objective, scipy_bounds,
            maxiter=self.max_iter,
            popsize=self.popsize,
            tol=self.tol,
            seed=42,
        )

        estimated = {k: result.x[i] for i, k in enumerate(param_names)}
        residual = float(np.sqrt(result.fun / len(observations)))

        return ParameterResult(
            params_estimated=estimated,
            residual=residual,
            converged=result.success,
            n_iterations=result.nit,
            info={"message": result.message}
        )


# ============================================================
#  Registry
# ============================================================

ESTIMATION_METHODS = {
    LeastSquaresEstimator.name: LeastSquaresEstimator,
    PoissonMLEEstimator.name: PoissonMLEEstimator,
    DifferentialEvolutionEstimator.name: DifferentialEvolutionEstimator,
}
