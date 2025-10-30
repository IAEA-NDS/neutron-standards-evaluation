import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, restrimap, exptable, expcov, priorvals, is_adj, usu_df, red_usu_df, num_covpars = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'restrimap', 'exptable', 'expcov', 'priorvals',
                 'is_adj', 'usu_df', 'red_usu_df', 'num_covpars')

# not used here, but to appease expectations of downstream scripts
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian

# quantities used in GLS
refvals = priorvals[is_adj]
expvals = exptable.DATA.to_numpy()
propfun = tf.function(restrimap.propagate)
jacfun = tf.function(restrimap.jacobian)

# GLS algo

num_iters = 10
tol = 1e-8
damp_unc = 5  # 500% damping uncertainty
solve = np.linalg.solve

newvals = refvals.copy()

for i in range(num_iters):
    print(f'iteration {i}')
    curvals = newvals
    propvals = propfun(curvals).numpy()
    S = tf.sparse.to_dense(jacfun(curvals)).numpy()
    expcov_abs = expcov * (propvals.reshape(-1,1) * propvals.reshape(1,-1))
    inv_postcov = S.T @ solve(expcov_abs, S)
    # poor-man LM algorithm: constant damping term
    damp_abs = 1/np.square(damp_unc) * np.diag(1/(curvals**2))
    inv_postcov_reg = inv_postcov + damp_abs

    d = expvals.reshape(-1,1) - propvals.reshape(-1,1)
    rhs = S.T @ solve(expcov_abs, d)
    delta = solve(inv_postcov_reg, rhs).flatten()
    newvals = curvals + delta

    relative_change = np.linalg.norm(delta) / np.linalg.norm(curvals)
    print(f'relative change: {relative_change}')

    if relative_change < tol:
        break


class DotDict:
    """Dictionary with dot-access to keys."""
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


optres = DotDict(position=newvals, converged=True)


# save the optimized parameters
refvals = optres.position

opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'usu_df', 'red_usu_df', 'opt_neg_hessian', 'opt_neg_jac')
