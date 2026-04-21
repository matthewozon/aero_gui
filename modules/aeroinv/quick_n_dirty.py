"""
quick_n_dirty.py -- Simple, non-iterative aerosol inversion methods.

Transliterated from quick_n_dirty.jl (AeroInv.jl) by Matthew Ozon.
Copyright (C) 2018-2019  Matthew Ozon  (MIT "Expat" Licence)
"""

import numpy as np


def aeroinv_quick(
    K: np.ndarray,
    diameter: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """Rough, direct estimate of the size density.

    For each measurement channel the function finds the diameter range in
    which the channel sensitivity is at least 50 % of its peak, estimates
    an effective bin width from that range, and divides the normalised
    measurement by that width.

    The result can be negative; it is intended as a quick sanity check, not
    a physically rigorous inversion.

    Parameters
    ----------
    K : ndarray, shape (nb_ch, mod_dim)
        Measurement model (sensitivity) matrix.
    diameter : ndarray, shape (mod_dim,)
        Diameter grid at which K is evaluated.
    data : ndarray, shape (nb_ch,) or (nb_ch, Nt)
        Measured data -- single snapshot or time series.

    Returns
    -------
    ndarray
        Estimated density with the same leading shape as ``data``.

    Raises
    ------
    ValueError
        If model and data dimensions are inconsistent.
    """
    nb_ch, mod_dim = K.shape
    if mod_dim != len(diameter):
        raise ValueError("aeroinv_quick: model dimension mismatch (K cols != len(diameter))")
    if nb_ch != data.shape[0]:
        raise ValueError("aeroinv_quick: data dimension mismatch (K rows != data.shape[0])")

    # Maximum sensitivity in each channel
    max_K = np.max(K, axis=1)  # (nb_ch,)

    # Index range where each channel is at >= 50 % of its peak sensitivity
    idx_bin_low = np.empty(nb_ch, dtype=int)
    idx_bin_hig = np.empty(nb_ch, dtype=int)
    for i in range(nb_ch):
        idx = np.where(K[i, :] >= 0.5 * max_K[i])[0]
        idx_bin_low[i] = idx[0]
        idx_bin_hig[i] = idx[-1]

    diameter_low = diameter[idx_bin_low]  # (nb_ch,)
    diameter_hig = diameter[idx_bin_hig]  # (nb_ch,)
    delta_bin = diameter_hig - diameter_low  # (nb_ch,)

    if data.ndim == 1:
        return (data / max_K) / delta_bin
    else:
        # Broadcast channel-wise arrays as column vectors
        return (data / max_K[:, np.newaxis]) / delta_bin[:, np.newaxis]


def precond_reg_inv(
    K: np.ndarray,
    data: np.ndarray,
    u0: float = 10000.0,
    rel_un: float = 0.1,
) -> np.ndarray:
    """Preconditioned, regularised inversion without a positivity constraint.

    Solves the penalised least-squares problem::

        n = argmin  ||E (K n - data)||^2  +  ||J n||^2

    where

    * ``E = diag(1 / K_uncertainty)`` is a channel-wise preconditioner
      derived from a model uncertainty estimate, and
    * ``J`` is a second-order finite-difference operator that enforces
      smoothness.

    The solution is obtained by the closed-form normal equation.
    **No positivity constraint** is applied; the returned density may
    contain negative values.

    Parameters
    ----------
    K : ndarray, shape (n_data, n_model)
        Measurement model matrix.
    data : ndarray, shape (n_data,) or (n_data, Nt)
        Measured data.
    u0 : float, optional
        Characteristic density scale (e.g. expected maximum), default 10000.
    rel_un : float, optional
        Relative amplitude of model uncertainty in (0, 1), default 0.1.

    Returns
    -------
    ndarray, shape (n_model,) or (n_model, Nt)
        Estimated density.
    """
    n_data_dim, n_model_dim = K.shape

    # Model uncertainty in the data space: K * (rel_un * u0 * 1)
    K_uncertainty = K @ (rel_un * u0 * np.ones(n_model_dim))  # (n_data_dim,)

    # Preconditioned operators
    inv_unc = 1.0 / K_uncertainty                              # (n_data_dim,)
    En = inv_unc[:, np.newaxis] * data if data.ndim > 1 else inv_unc * data
    EK = inv_unc[:, np.newaxis] * K   # (n_data_dim, n_model_dim)

    # Second-order finite-difference regularisation operator J
    J = (
        np.diag(-2.0 * np.ones(n_model_dim))
        + np.diag(np.ones(n_model_dim - 1), k=1)
        + np.diag(np.ones(n_model_dim - 1), k=-1)
    )
    # Zero out the boundary rows (no derivative at edges)
    J[0, 0] = 0.0
    J[0, 1] = 0.0
    J[-1, -1] = 0.0
    J[-1, -2] = 0.0
    J = 1.0e-2 * J

    # Normal equation: (EK'EK + J'J) n = EK' En
    lhs = EK.T @ EK + J.T @ J          # (n_model_dim, n_model_dim)
    rhs = EK.T @ En                     # (n_model_dim,) or (n_model_dim, Nt)
    return np.linalg.solve(lhs, rhs)
