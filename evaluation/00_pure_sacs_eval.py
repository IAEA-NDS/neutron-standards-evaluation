import re
import pandas as pd
from scipy.sparse import block_diag, csr_matrix
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    save_objects
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table,
)
from gmapy.mappings.priortools import (
    attach_shape_prior,
    initialize_shape_prior,
    remove_dummy_datasets
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.tf.energy_dependent_absolute_usu_map_tf import (
    EnergyDependentAbsoluteUSUMap,
    create_endep_abs_usu_df
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)
from gmapy.tf_uq.inference import determine_MAP_estimate

tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = '../data/data.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'], relative=True)
exptable['UNC'] = np.sqrt(expcov.diagonal())

# only keep sacs values in exptable
B = np.array(expcov.copy().todense())

sacs_mask = exptable.REAC.str.match('MT:(6|10)($|-)')
exptable.loc[sacs_mask, 'REAC'].drop_duplicates()
expcov_sacs = B[np.ix_(sacs_mask, sacs_mask)]
exptable_sacs = exptable.loc[sacs_mask].copy()
exptable_nonsacs = exptable.loc[~sacs_mask].copy()
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

expcov_chol_sacs = tf.linalg.cholesky(expcov_sacs)
expcov_chol_sacs_op = tf.linalg.LinearOperatorLowerTriangular(
    expcov_chol_sacs, is_positive_definite=True
)

propfun = tf.function(compmap_sacs.propagate)
jacfun = tf.function(compmap_sacs.jacobian)

likelihood = MultivariateNormalLikelihood(
    len(priortable_sacs), propfun, jacfun, expvals_sacs, expcov_chol_sacs_op,
    approximate_hessian=True, relative=True
)

neg_log_prob_and_gradient = tf.function(likelihood.neg_log_prob_and_gradient)
neg_log_post_hessian = likelihood.neg_log_prob_hessian

refvals = priortable_sacs.PRIOR.to_numpy() 
refvals = tf.constant([1.20, 1.85, 0.3], dtype=tf.float64)  # priortable_sacs.PRIOR.to_numpy() 

optres = determine_MAP_estimate(
    refvals, neg_log_prob_and_gradient, neg_log_post_hessian,
    max_inner_iters=100, max_outer_iters=50, nugget=1e-2,
    ret_optres=True, must_converge=True
)

priortable_sacs['POST'] = optres.position.numpy() 

updvals = optres.position
y = propfun(updvals).numpy()
S = tf.sparse.to_dense(jacfun(updvals)).numpy()
C = np.diag(y) @ expcov_sacs @ np.diag(y)

postcov = np.linalg.inv(S.T @ np.linalg.inv(C) @ S)
updvals = updvals + np.linalg.inv(S.T @ np.linalg.inv(C) @ S) @ S.T @ np.linalg.inv(C) @ (expvals_sacs - y)

# Approximate Hessian versus sandwich formula

# 1) approximate Hessian (neglecting change in covariance matrix)
Hapx = likelihood.log_prob_hessian(optres.position) 
postcov_hess = -np.linalg.inv(Hapx) 

print(
    '\nRelative difference between posterior covariance matrix (sandwich) and '
    'better Hessian approximation (accounting for forward-mapping derivatives): '
)
print((postcov - postcov_hess) / np.abs(postcov))
print('\nRelative difference in uncertainties: ')
print((np.sqrt(np.diag(postcov)) - np.sqrt(np.diag(postcov_hess))) / np.sqrt(np.diag(postcov))) 

# 2) exact Hessian 
x = tf.constant(optres.position)
with tf.GradientTape() as t2:
    t2.watch(x)
    with tf.GradientTape() as t1:
        t1.watch(x)
        y = likelihood.log_prob(x)
    g = t1.gradient(y, x)

Hexact = t2.jacobian(g, x)
postcov_exact = -np.linalg.inv(Hexact)

print(
    '\nRelative difference between posterior covariance matrix (sandwich) and '
    'covariance matrix derived from exact Hessian'
)
print((postcov - postcov_exact) / np.abs(postcov))
print('\nRelative difference in uncertainties: ')
print((np.sqrt(np.diag(postcov)) - np.sqrt(np.diag(postcov_exact))) / np.sqrt(np.diag(postcov))) 

print('\nExperimental SACS data: ')
print(exptable_sacs)

print('\nPure SACS evaluation result: ')
print(priortable_sacs)

postvals_sacs = optres.position.numpy()
postcov_sacs = postcov_exact 
postcov_rel_sacs = postcov_exact / np.outer(postvals_sacs, postvals_sacs)

exptable_sacs2 = pd.DataFrame({
    'NODE': ['exp_7000', 'exp_7001', 'exp_7002'],
    'REAC': ['MT:6-R1:8', 'MT:6-R1:9', 'MT:6-R1:10'],
    'ENERGY': np.nan,
    'DATA': postvals_sacs,
    'DB_IDX': np.max(exptable_nonsacs.DB_IDX) + 1,
    'DS_IDX': [0, 1, 2],
    'UNC': np.sqrt(np.diag(postcov_rel_sacs)),
})

exptable = pd.concat([exptable_nonsacs, exptable_sacs2], ignore_index=True)
expcov_nonsacs = expcov[np.ix_(~sacs_mask, ~sacs_mask)]
expcov = block_diag([expcov_nonsacs, 1e-10 * np.identity(3)]).toarray()
expcov_cut = block_diag([expcov_nonsacs, postcov_rel_sacs]).toarray()
exptable.UNC = np.sqrt(np.diag(expcov))


save_objects('output/00_sacs_eval.pkl', locals(),
    'exptable_sacs', 'priortable_sacs', 'expcov_sacs',
    'postvals_sacs', 'postcov_rel_sacs', 'exptable', 'expcov', 'expcov_cut'
)
