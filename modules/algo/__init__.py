# modules/algo/__init__.py
# Pure-Python / NumPy implementations of the BAYROSOL + NMOpt algorithms.
# No Julia, no external runtime required.

from .kalman      import (KFWorkspace, kalman_filter, kalman_filter_smoother,
                          make_linear_callbacks, filtered_states, smoothed_states,
                          filtered_stds, smoothed_stds)

from .gde         import (AeroSys, coagulation_coefficient,
                          init_coagulation_indices,
                          iter as gde_iter, jacobian_GDE,
                          simulate as gde_simulate,
                          wall_deposition_rate,
                          cond_err, loss_err, discretization_err)

from .measurement import (SMPSKernel, smps3936_transfer_function,
                          smps3936_forward, charge_probability,
                          mobility_from_size_and_charge,
                          size_from_mobility_and_charge,
                          impactor_efficiency, cpc_efficiency,
                          neutraliser_Kr85, dma_size_density,
                          dma_operator, dma_operator_matrix,
                          set_seed as measurement_set_seed)

from .stochproc   import (linSP, chol_psd,
                          covariance_process_2nd, covariance_process_array,
                          covariance_process_range_limit,
                          covariance_2nd_order_space_1st_order_time,
                          space_covariance_chol, space_covariance_array,
                          space_covariance_range_limit,
                          covariance_second_order_process,
                          SP_1T, SP_1TC, SP_1TCV,
                          SP_2T, SP_2TC, SP_2TCV, SP_2TCV_mat,
                          SP_1T_V, SP_1TC_V,
                          iterator, timeBlockMatrix, initGaussRep,
                          mean_and_variance_estimation,
                          percentile_estimation,
                          multivariate_probability,
                          uniform_sample_unit_sphere,
                          uniform_sample_unit_ball,
                          set_seed as stochproc_set_seed)

from .optimisation import (BFGSParam, OptimResult, line_search,
                           BFGS, BFGSB, LBFGS, LBFGSB)

from .utils import (softplus, softplus_deriv, softplus_scaled,
                    softplus_scaled_deriv, softplus_inv, softplus_inv_deriv,
                    softplus_inv_scaled, softplus_inv_scaled_deriv,
                    logistic, logistic_inv, logistic_deriv,
                    cauchy, cauchy_deriv, phi_hl, phi_hl_deriv,
                    riemann, e_0, e_k, e_M)
