import re
import time
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from data_preparation import (
    priortable,
    rpriortable,
    exptable,
    rS,
    S as Sexp,
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
    get_matern32_cov,
    cut_inference,
)
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.linalg import block_diag
from genetic_algo import Evolution
import pandas as pd
import re
from unctools import plot_cormat, cov2cor


refvals = priortable.PRED.to_numpy()
predvals = np.concatenate([refvals, exptable.PRED.to_numpy()])

fis_mask = priortable.NODE == 'fis'
Sprior = np.identity(Sexp.shape[1], dtype=float)
Sexp = _to_dense_array(Sexp)
Sred = Sexp[:,~fis_mask].copy()
A = np.identity(Sexp.shape[1], dtype=float) * 1e8
B = _to_dense_array(expcov_abs)


###########################################
###     Pure SACS evaluation            ###     
###########################################

B = _to_dense_array(expcov_abs)

#  a) extract the SACS measurements
sacs_mask = exptable.REAC.str.match('MT:(6|10)($|-)')
exptable.loc[sacs_mask, 'REAC'].drop_duplicates()
B_sacs = B[np.ix_(sacs_mask, sacs_mask)]
exptable_sacs = exptable.loc[sacs_mask].copy()
#  b) convert SACS MT:6 to MT:1 and MT:10 to MT:3
exptable_sacs.REAC = exptable_sacs.REAC.str.replace('MT:6-', 'MT:1-')
exptable_sacs.REAC = exptable_sacs.REAC.str.replace('MT:10-', 'MT:3-')
exptable_sacs.ENERGY = 1.0
priortable_sacs = pd.DataFrame({
    'NODE': ['xsid_8', 'xsid_9', 'xsid_10'],
    'REAC': ['MT:1-R1:8', 'MT:1-R1:9', 'MT:1-R1:10'],
    'ENERGY': [1.0, 1.0, 1.0],
    'PRIOR': [1.0, 1.0, 1.0],
})
#  c) prepare sensitivity matrix for evaluation
refvals_sacs = tf.constant([1.210, 1.812, 0.3257], dtype=tf.float64)
priortable_sacs['PRIOR'] = refvals_sacs
expvals_sacs = exptable_sacs.DATA.to_numpy()
compmap_sacs = CompoundMapTF((priortable_sacs, exptable_sacs), reduce=True)
compmap_sacs.propagate(refvals_sacs)
S_sacs = tf.sparse.to_dense(compmap_sacs.jacobian(refvals_sacs)).numpy()
#  d) perform evaluation
d_sacs = (expvals_sacs - compmap_sacs.propagate(refvals_sacs)).numpy().reshape(-1,1)
postcov_sacs = np.linalg.inv(S_sacs.T @ np.linalg.inv(B_sacs) @ S_sacs)
postvals_sacs = (
    (postcov_sacs @ S_sacs.T @ np.linalg.inv(B_sacs) @ d_sacs).flatten()
    + refvals_sacs.numpy()
)
priortable_sacs['DATA'] = postvals_sacs

# 2) add the pure SACS evaluation result as data to the experiment dataframe
exptable_sacs2 = pd.DataFrame({
    'NODE': ['exp_7000', 'exp_7001', 'exp_7002'],
    'REAC': ['MT:6-R1:8', 'MT:6-R1:9', 'MT:6-R1:10'],
    'ENERGY': np.nan,
    'DATA': postvals_sacs,
    'UNC': np.sqrt(np.diag(postcov_sacs)) / postvals_sacs
})

# Impute the frozen distribution for SACS in the extended covariance matrix.
# Here we can potentially include other frozen distributions,
# e.g. coming from R-matrix fits or TNC evaluation

exptable_nonsacs = exptable[~sacs_mask]
B_nonsacs = B[np.ix_(~sacs_mask, ~sacs_mask)]

exptable_cut = pd.concat([exptable_nonsacs, exptable_sacs2])
compmap_cut = CompoundMapTF([priortable, exptable_cut], reduce=True)
expcov_cut = block_diag(B_nonsacs, postcov_sacs) 
S_cut = tf.sparse.to_dense(compmap_cut.jacobian(refvals)).numpy()
predvals_cut = compmap_cut.propagate(refvals).numpy()
expvals_cut = exptable_cut.DATA.to_numpy()

# remove fission spectrum because fixed
fis_mask = priortable.NODE == 'fis'
S_cut = S_cut[:, ~fis_mask].copy()
refvals_cut = refvals[~fis_mask].copy()
cut_idcs = np.where(exptable_cut.REAC.str.match('MT:6-'))[0]


###########################################
###     THE REAL CUT !!!!!!!!!!!!!!!!!  ###
###########################################

cutpost = cut_inference(refvals_cut, S_cut, predvals_cut, expvals_cut, expcov_cut, cut_idcs=cut_idcs, cut_unc=1e-6)

# some checking 
testvec = predvals_cut.reshape(-1,1) + S_cut @ (cutpost[0] - refvals_cut).reshape(-1,1)

np.sqrt(np.diag((S_cut @ cutpost[1] @ S_cut.T)[np.ix_(cut_idcs, cut_idcs)])) / expvals_cut[-3:]

sel_idcs = rpriortable.query('REAC == "MT:1-R1:8" & ENERGY >= 1 & ENERGY <= 5').index.to_numpy()
(np.sqrt(np.diag(cutpost[1])) / cutpost[0])[sel_idcs]

