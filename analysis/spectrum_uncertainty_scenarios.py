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

# Different options for uncertainties on Cf-252 fission spectrum

fis_idcs = priortable.index[priortable.NODE == 'fis']
fis_ens = priortable.ENERGY[fis_idcs].to_numpy().reshape(-1, 1)
fis_vals = priortable.PRED[fis_idcs].to_numpy().reshape(-1, 1)

## Scenario 1: no uncertainty in fission spectrum
A[fis_idcs, fis_idcs] = np.square(priortable.loc[fis_idcs, 'PRIOR'] * 1e-2)

## Scenario 2: five percent fully correlated 
# Conclusion: a fully correlated uncertainty does not help for increasing SACS uncertainty
A[np.ix_(fis_idcs, fis_idcs)] = refvals[fis_idcs].reshape(-1, 1) * refvals[fis_idcs].reshape(1,-1) * np.square(100)
A[fis_idcs, fis_idcs] += np.square(refvals[fis_idcs] * 0.1)

## Scenario 3: use a Matern covariance matrix
matern_relcov = get_matern32_cov(fis_ens, fis_ens, 0.1, 5)
A[np.ix_(fis_idcs, fis_idcs)] =  matern_relcov * (fis_vals @ fis_vals.T)
A[fis_idcs, fis_idcs] += np.square(fis_vals.flatten() * 1e-4) # small regularization

# Assembly of the matrices
S = np.vstack([Sprior, Sexp])
C = block_diag(A, B) 

# calculate the covariance matrix
postcov = np.linalg.inv(S.T @ np.linalg.inv(C) @ S)
predcov = S @ postcov @ S.T

postcov_rel = postcov / refvals.reshape(-1, 1) / refvals.reshape(1, -1)
predcov_rel = predcov / predvals.reshape(-1, 1) / predvals.reshape(1, -1)

postunc_rel = np.sqrt(np.diagonal(postcov_rel))
predunc_rel = np.sqrt(np.diagonal(predcov_rel))
predunc_rel = np.sqrt(np.diagonal(predcov_rel))

# U5 sacs
predunc_rel[6664 + 1236]
# Pu9 sacs
predunc_rel[6673 + 1236]
# Pu9/U5 sacs
predunc_rel[6667 + 1236]

# show U5 uncertainty
mask = (priortable.REAC == 'MT:1-R1:9') & (priortable.ENERGY >= 1) & (priortable.ENERGY <= 5)
postunc_rel[mask]


mask = (priortable.REAC == 'MT:1-R1:8').to_numpy() & (postunc_rel > 0.05)
postcov[mask,:] = 0 
postcov[:,mask] = 0

# due to numerical instabilities we need to do also this
mask = (priortable.NODE == 'fis').to_numpy()
postcov[mask,:] = 0 
postcov[:,mask] = 0 

# Try cut-posterior 

# Level 1
# All other measurements

B = _to_dense_array(expcov_abs)
expmask = exptable.REAC.str.match('.*R.:(9|10)($|-)')
expmask |= exptable.REAC.str.match('MT:(6|10)-') 

# no significant change
expmask |= exptable.REAC == 'MT:2-R1:8'
expmask |= exptable.REAC == 'MT:4-R1:6-R2:8'
expmask |= exptable.REAC == 'MT:3-R1:30-R2:8'
expmask |= exptable.REAC == 'MT:3-R1:14-R2:8'
expmask |= exptable.REAC == 'MT:3-R1:20-R2:8'
expmask |= exptable.REAC == 'MT:4-R1:8-R2:1'
expmask |= exptable.REAC == 'MT:3-R1:8-R2:4'
expmask |= exptable.REAC == 'MT:4-R1:8-R2:4'
expmask |= exptable.REAC == 'MT:9-R1:8-R2:3-R3:4'
expmask |= exptable.REAC == 'MT:4-R1:1-R2:8'
expmask |= exptable.REAC == 'MT:4-R1:7-R2:8'
expmask |= exptable.REAC == 'MT:3-R1:7-R2:8'
expmask |= exptable.REAC == 'MT:3-R1:8-R2:6'
expmask |= exptable.REAC == 'MT:3-R1:6-R2:8'
# testing


exptable.loc[expmask, 'REAC'].drop_duplicates()

B[expmask, expmask] += 1e16
C = block_diag(A, B) 

m = ~expmask & (exptable.REAC.str.match('.*R.:9'))
exptable.loc[m, 'REAC'].drop_duplicates()
exptable.loc[m, 'REAC']
exptable.loc[m & (exptable.ENERGY > 1.0) & (exptable.ENERGY < 5.0), 'NODE'].drop_duplicates()



dt = exptable.loc[m & (exptable.ENERGY > 1e-6) & (exptable.ENERGY < 3e3)]
import matplotlib.pyplot as plt
plt.plot(dt.ENERGY, dt.DATA, 'ro')
plt.show()


dt = exptable
idcs = dt.index
d = (dt.PRED - dt.DATA)[idcs].to_numpy()
U = B[np.ix_(idcs, idcs)]

d.T @ np.linalg.inv(U) @ d

import re
re.match('.*R-9($|-)', 'MT:2-R1:9')
re.match('.*R.:(9|10)($|-)', 'MT:4-R1:10-R2:8')

# Only keep Pu9 related data
B = _to_dense_array(expcov_abs)
expmask = exptable.REAC.str.match('.*R.:8($|-)')
# no influence
expmask |= exptable.REAC.str.match('MT:1-R1:8')
expmask |= exptable.REAC.str.match('MT:1-R1:9')
expmask |= exptable.REAC.str.match('MT:2-R1:9')
expmask |= exptable.REAC.str.match('MT:9-R1:9-R2:3-R3:4')
expmask |= exptable.REAC.str.match('MT:4-R1:9-R2:10')
expmask |= exptable.REAC.str.match('MT:4-R1:7-R2:9')
expmask |= exptable.REAC.str.match('MT:4-R1:9-R2:1')
# test

B[expmask, expmask] += 1e16
C = block_diag(A, B) 

m = ~expmask & (exptable.REAC.str.match('.*R.:9'))
exptable.loc[m, 'REAC'].drop_duplicates()
