import re
import time
import numpy as np
import tensorflow as tf
from data_preparation import (
    priortable,
    rpriortable,
    exptable,
    rS,
    covmat_blocks,
    inv_covmat_blocks,
    S_blocks,
    block_starts,
    block_stops,
    expcov as expcov_rel,
    expcov_abs,
    propfun,
    jacfun,
    likelihood,
    like_cov_fun,
    refvals,
    rrefvals,
    optres,
    num_covpars,
    expcov_linop,
)
from unctools import (
    calc_target_unc,
    calc_rel_target_unc,
    calc_postmean,
    calc_postcov,
    get_S_by_blocks,
    get_idcs_by_blocks,
    get_exptable_by_blocks,
    get_expcov_by_blocks,
    get_normalization_unc,
    add_normunc,
    calc_postcov2,
    _to_dense_array,
    cov2cor,
    plot_cormat,
)
from scipy.sparse import coo_matrix, csr_matrix, diags
from genetic_algo import Evolution
import pandas as pd
import re
from unctools import plot_cormat, cov2cor


# include a mass determination USU

def get_mass_usu_sens(exp_reacs):
    row_idcs = []
    col_idcs = []
    vals = []
    for i, reac in enumerate(exp_reacs): 
        if re.match('MT:([136]|10)-', reac):
            if 'R1:8' in reac:
                row_idcs.append(i)
                col_idcs.append(0)
                vals.append(1.)
            elif 'R1:9' in reac:
                row_idcs.append(i)
                col_idcs.append(1)
                vals.append(1.)
            elif 'R1:10' in reac: 
                row_idcs.append(i)
                col_idcs.append(2)
                vals.append(1.)
        if reac.startswith('MT:3-') or reac.startswith('MT:10-'):
            if 'R2:8' in reac:
                row_idcs.append(i)
                col_idcs.append(0)
                vals.append(-1.)
            elif 'R2:9' in reac:
                row_idcs.append(i)
                col_idcs.append(1)
                vals.append(-1.)
            elif 'R2:10' in reac: 
                row_idcs.append(i)
                col_idcs.append(2)
                vals.append(-1.)
    return csr_matrix(coo_matrix((vals, (row_idcs, col_idcs))))


Susu = get_mass_usu_sens(exptable.REAC.tolist())
mass_usu = diags(np.square([0.007, 0.007, 0.007]))
usu_cov = Susu @ mass_usu @ Susu.T

new_expcov = expcov_rel + usu_cov 

postcov = calc_postcov2(
    rS, new_expcov, exptable.PRED, rpriortable.PRED, block_starts, block_stops
)


sel_idcs = rpriortable.loc[
    (rpriortable.REAC == 'MT:1-R1:9') &
    (rpriortable.ENERGY >= 1.) & 
    (rpriortable.ENERGY <= 6.)
].index

rpriortable.loc[sel_idcs]
np.sqrt(np.diagonal(postcov)[sel_idcs])

sacs_idx = exptable.loc[exptable.REAC == 'MT:6-R1:8'].index
sacs_idx = exptable.loc[exptable.REAC == 'MT:6-R1:9'].index
sacs_idx = exptable.loc[exptable.REAC == 'MT:10-R1:9-R2:8'].index
rrS = rS[sacs_idx,:]
np.sqrt(rrS @ postcov @ rrS.T)

ens = rpriortable.loc[sel_idcs, 'ENERGY'].to_numpy()
cormat = cov2cor(postcov[np.ix_(sel_idcs, sel_idcs)])
plot_cormat(ens, ens, cormat)

# compare the experimental covariance matrix
#   1) constructed explicitly here
#   vs 2) constructed via the create_likecov_fun in the data preparation step

expcov_rel_cmp = like_cov_fun(tf.constant([7e-3, 7e-3, 7e-3], dtype=tf.float64)).to_dense()
tmp = np.abs(new_expcov.toarray() - expcov_rel_cmp)
np.max(tmp)
# we know that like_cov_fun is also used in the initialization
# of the likelihood function.

# calculate using sandwich formula
postcov = calc_postcov2(
    rS, expcov_rel_cmp.numpy(), exptable.PRED, rpriortable.PRED, block_starts, block_stops
)
np.sqrt(np.diagonal(postcov))[sel_idcs]

# compare the two Posterior approximations:
#  1) negative of inverse Hessian of log-likelihood
#  vs 2) Sandwhich formula (assuming linear model link)
errefvals = tf.concat([rrefvals, [7e-3]*3], axis=0)
post_hess = likelihood.log_prob_hessian(errefvals)

postcov_cmp = -np.linalg.inv(post_hess.numpy())
predvals_cmp = propfun(rrefvals).numpy()
postcov_cmp_rel = postcov_cmp / errefvals.numpy().reshape(-1,1) / errefvals.numpy().reshape(1,-1) 

post_hess2 = post_hess[:-3, :-3]
postcov_cmp2 = -np.linalg.inv(post_hess2.numpy())
postcov_cmp_rel2 = postcov_cmp2 / errefvals[:-3].numpy().reshape(-1,1) / errefvals[:-3].numpy().reshape(1,-1) 

np.sqrt(np.diagonal(postcov_cmp_rel))[sel_idcs]
np.sqrt(np.diagonal(postcov_cmp_rel2))[sel_idcs]
np.sqrt(np.diagonal(postcov))[sel_idcs]

