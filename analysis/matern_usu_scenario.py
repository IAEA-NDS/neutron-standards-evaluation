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

usu_reacs = [
'MT:1-R1:8',
'MT:2-R1:8',
'MT:1-R1:9',
'MT:2-R1:9',
'MT:1-R1:10',
'MT:2-R1:10',
'MT:3-R1:9-R2:8',
'MT:4-R1:9-R2:8',
'MT:3-R1:10-R2:8',
'MT:4-R1:10-R2:8',
]

# remove all U5,U9,U10 shape and absolute measurements
expmask = exptable.REAC.isin(usu_reacs)
expmask = ~expmask
sel_idcs = np.where(expmask)[0]

# only pick a single energy slice
expmask &= exptable.ENERGY == 2.0 
sel_idcs = np.where(expmask)[0]

# do the evaluation
postcov = calc_postcov2(
    rS, expcov_rel, exptable.PRED, rpriortable.PRED, block_starts, block_stops,
    idcs=sel_idcs
)

# show uncertainties
sel_idcs = rpriortable[rpriortable.REAC.str.match('MT:1-R1:8')].index
rpriortable['POSTUNC'] = np.sqrt(np.diag(postcov))
rpriortable.loc[sel_idcs]


def get_matern32_cov(e1, e2, s, r):
    d = np.abs(e1[:,None] - e2)
    z1 = (1 + np.sqrt(3)*d/r)
    z2 = np.exp(-np.sqrt(3)*d/r)
    covmat = s*s * z1 * z2
    return covmat


# include a mass determination USU
def get_usu_cov(exptable, s, r):
    vals = []
    usu_cov = np.zeros([len(exptable)]*2, dtype=float)
    for reac in usu_reacs: 
        curdt = exptable[exptable.REAC == reac]
        idx = curdt.index.to_numpy() 
        ens = curdt.ENERGY.to_numpy()
        curcov = get_matern32_cov(ens, ens, s, r)
        usu_cov[np.ix_(idx, idx)] = curcov
    return usu_cov


usu_cov = get_usu_cov(exptable, 0.015, 5)  
new_expcov = csr_matrix(expcov_rel + usu_cov)


expmask = exptable.REAC.str.match('MT:(6|10)-')
expmask = ~expmask
sel_idcs = np.where(expmask)[0]

postcov = calc_postcov2(
    rS, new_expcov, exptable.PRED, rpriortable.PRED, block_starts, block_stops,
    idcs = sel_idcs,
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

