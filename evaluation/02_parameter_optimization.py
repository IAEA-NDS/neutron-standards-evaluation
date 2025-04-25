import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, likelihood, priorvals, is_adj, usu_df, red_usu_df, num_covpars = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj',
                 'usu_df', 'red_usu_df', 'num_covpars')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian
refvals = tf.constant(priorvals, dtype=tf.float64)


optres = determine_MAP_estimate(
    refvals, neg_log_prob_and_gradient,
    neg_log_post_hessian, max_inner_iters=500, max_outer_iters=50, nugget=1e-3,
    ret_optres=True, must_converge=True
)

# save the optimized parameters
refvals = optres.position


opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

post_covmat = np.linalg.inv(opt_neg_hessian + np.diag([1e-5]*len(priortable)))
priortable['POSTUNC'] = post_covmat.diagonal() * 4*priortable['POST'] / priortable['POST'] * 100

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'opt_neg_hessian', 'opt_neg_jac')


if True:
    sys.exit(0)


# look at trends
postvals = tf.constant(priortable['POST'])
gma_input, axt_input = post._distribute_params(postvals)

predvals = propfun(gma_input)
exptable['PRED'] = predvals

exptable[['DATA', 'PRED']]


tnc_datasets = set(exptable.loc[exptable.ENERGY == 2.53e-8, 'NODE'])
high_datasets = set(exptable.loc[exptable.ENERGY > 2.53e-8, 'NODE'])
both_datasets = tnc_datasets.intersection(high_datasets)

import matplotlib.pyplot as plt

for ds in ('exp_602', 'exp_631'): # both_datasets:
    print(ds)
    curds = exptable[exptable.NODE == ds]
    plt.plot(curds['ENERGY'], (curds['DATA'] - curds['PRED']) / curds['PRED'], color='red')
    plt.plot(curds['ENERGY'], curds['UNC'] / curds['PRED'], color='black')
    plt.plot(curds['ENERGY'], -curds['UNC'] / curds['PRED'], color='black')
    plt.axhline(y=0, color='gray')
    plt.xscale('log')
    reac = curds['REAC'].iloc[0]
    plt.title(f'dataset: {ds} - reac: {reac}')
    plt.show()
