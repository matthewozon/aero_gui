"""
aeroinv - Aerosol inversion Python package.

Transliterated from AeroInv.jl (Julia) by Matthew Ozon.
Solves ill-posed inverse problems to retrieve particle size density
distributions from aerosol measurement data.

Public API
----------
aeroinv_quick       : rough, direct size-density estimate
precond_reg_inv     : preconditioned regularised inversion (no positivity)
aeroinv_reg         : Chambolle-Pock inversion, Gaussian noise + positivity
alg2_cp_poisson     : Chambolle-Pock, Poisson noise + positivity
ozon2020            : Newton-like outer loop with Chambolle-Pock inner solver
alg2_chpo_nso       : Chambolle-Pock, Gaussian noise, null-space optimisation
"""

from .quick_n_dirty import aeroinv_quick, precond_reg_inv
from .chambolle_pock_alg_2 import aeroinv_reg
from .cp_poisson import (
    F_poisson,
    G_poisson,
    F_convex_conjugate_poisson,
    prox_F_conj_poisson,
    prox_G_poisson,
    alg2_cp_poisson,
)
from .cp_poisson_approx import (
    phi_pois_ap,
    grad_phi_pois_ap,
    hess_phi_diag_pois_ap,
    ozon2020_iteration,
    ozon2020,
)
from .cp_nso import (
    F_nso,
    F_convex_conjugate_nso,
    prox_F_conj_nso,
    G_nso,
    prox_G_nso,
    alg2_chpo_nso,
)

__all__ = [
    "aeroinv_quick",
    "precond_reg_inv",
    "aeroinv_reg",
    "F_poisson",
    "G_poisson",
    "F_convex_conjugate_poisson",
    "prox_F_conj_poisson",
    "prox_G_poisson",
    "alg2_cp_poisson",
    "phi_pois_ap",
    "grad_phi_pois_ap",
    "hess_phi_diag_pois_ap",
    "ozon2020_iteration",
    "ozon2020",
    "F_nso",
    "F_convex_conjugate_nso",
    "prox_F_conj_nso",
    "G_nso",
    "prox_G_nso",
    "alg2_chpo_nso",
]
