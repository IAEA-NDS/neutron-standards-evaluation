import sys
import pathlib
# resolve gmapy to the submodule of this repository (must match the
# version used to create the pickles loaded below)
sys.path.insert(
    0, (pathlib.Path(__file__).resolve().parents[1] / 'gmapy').as_posix()
)
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate_precond_lbfgs
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

import os
outdir = os.environ.get('EVAL_OUTPUT_DIR', 'output')

post, likelihood, priorvals, is_adj, usu_df, red_usu_df, num_covpars = \
    load_objects(f'{outdir}/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj',
                 'usu_df', 'red_usu_df', 'num_covpars')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian

covpars = np.full(num_covpars, 1.)
refvals = likelihood.combine_pars(priorvals[is_adj], covpars)

batched_neg_log_prob = tf.function(post.neg_log_prob_batch)

optres = determine_MAP_estimate_precond_lbfgs(
    refvals, neg_log_prob_and_gradient, neg_log_post_hessian,
    max_iters=10000, nugget=1e-3, hessian_refresh_interval=100,
    batch_neg_log_prob=batched_neg_log_prob, saddle_free='auto',
    checkpoint_file=f'{outdir}/02_optimizer_checkpoint.npz',
    ret_optres=True, must_converge=True
)

# save the optimized parameters
refvals = optres.position
red_usu_df['USU'] = refvals.numpy()[-num_covpars:]
params, covpars = likelihood.split_pars(refvals)

opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

save_objects(f'{outdir}/02_parameter_optimization_output.pkl', locals(),
             'optres', 'params', 'covpars', 'usu_df', 'red_usu_df',
             'opt_neg_hessian', 'opt_neg_jac')
