import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import (
    iterative_gls_estimate,
)
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
expvals = post.get_data_vector()
propfun = post.get_model_prediction
jacfun = post.get_model_jacobian
cov_linop_fun = post.get_covariance_linop

# GLS algo
optres = iterative_gls_estimate(
    refvals, propfun, jacfun, expvals, cov_linop_fun, ret_optres=True
)

# save the optimized parameters
refvals = optres.position

opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'usu_df', 'red_usu_df', 'opt_neg_hessian', 'opt_neg_jac')
