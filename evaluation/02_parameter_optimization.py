import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, likelihood, priortable, priorvals, is_adj, num_covpars = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priortable', 'priorvals', 'is_adj',
                 'num_covpars')


# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian

refvals = priorvals[is_adj]


# debug
# calculate covariance matrix according to cut posterior approach
from cut_posterior_tools import cut_inference
restrimap, exptable, expcov_cut_rel, cut_idcs, = load_objects(
    'output/01_model_preparation_output.pkl', 'restrimap',
    'exptable', 'expcov_cut', 'cut_idcs',
)

# debug start
refvals_cut = refvals.copy()
S_cut = tf.sparse.to_dense(restrimap.jacobian(refvals_cut)).numpy()
predvals_cut = restrimap.propagate(refvals_cut).numpy()
expvals_cut = exptable.DATA.to_numpy()
expcov_cut = expcov_cut_rel * np.outer(predvals_cut, predvals_cut)
rpriortable = priortable.loc[is_adj].copy()
postvals_cut_debug, postcov_cut_debug = cut_inference(refvals_cut, S_cut, predvals_cut, expvals_cut, expcov_cut, cut_idcs=cut_idcs)
# debug stop

optres = determine_MAP_estimate(
    refvals, neg_log_prob_and_gradient,
    neg_log_post_hessian, max_inner_iters=500, max_outer_iters=100, nugget=1e-8,
    ret_optres=True, must_converge=True
)

# save the optimized parameters
refvals = optres.position

opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

# calculate covariance matrix according to cut posterior approach
from cut_posterior_tools import cut_inference
restrimap, exptable, expcov_cut_rel, = load_objects(
    'output/01_model_preparation_output.pkl', 'restrimap', 'exptable', 'expcov_cut'
)

refvals_cut = optres.position.numpy()
S_cut = tf.sparse.to_dense(restrimap.jacobian(refvals_cut)).numpy()
predvals_cut = restrimap.propagate(refvals_cut).numpy()
expvals_cut = exptable.DATA.to_numpy()
expcov_cut = expcov_cut_rel * np.outer(predvals_cut, predvals_cut)
postvals_cut, postcov_cut = cut_inference(
    refvals_cut, S_cut, predvals_cut, expvals_cut, expcov_cut, cut_idcs=cut_idcs
)


save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'opt_neg_hessian', 'opt_neg_jac',
             'postvals_cut', 'postcov_cut', 'postvals_cut_debug', 'postcov_cut_debug')
