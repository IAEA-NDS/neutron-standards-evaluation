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
    # MultivariateNormalLikelihoodWithCovParams,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)

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
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

# variation-01: remove specific experimental datasets after visual inspection
exp_remove_mask = (exptable.NODE == 'exp_722') & (exptable.ENERGY > 23)  # Ponkratov U5(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_8008')  # removal of Nolte abs. U8(n,f) measurement (34 - 200 MeV)
exp_remove_mask |= (exptable.NODE == 'exp_874') & (exptable.ENERGY > 23)  # Ponkratov U8(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_524') & (exptable.ENERGY > 27)  # A.D. Carlson PU5(n,f) above 27 MeV
# remove due to recommendation in excel sheet
exp_remove_mask |= (exptable.NODE == 'exp_8029')
# remove Maslov's patch
exp_remove_mask |= (exptable.NODE == 'exp_1003')
# remove Scherbakov PU9/U5 n,f data
exp_remove_mask |= (exptable.NODE == 'exp_1012')
# remove everything except datapoints at 29 and 30 MeV
exp_remove_mask |= (exptable.ENERGY < 29) | (exptable.ENERGY > 30)

exp_keep_idcs = np.where(~exp_remove_mask)[0]
exptable = exptable.loc[exp_keep_idcs].reset_index(drop=True)
expcov = csr_matrix(expcov.toarray()[np.ix_(exp_keep_idcs, exp_keep_idcs)])
# variation-01 end

# implement the recommendations of the excel sheet,
# except the recommendation to convert the
# Cance 1978 Pu9, U8, U5 absolute cross sections to ratios
def replace_mt(node, old_mt, new_mt):
    """Change the data type (given by MT) of a dataset."""
    t = exptable.loc[exptable.NODE == node, 'REAC']
    t = t.str.replace(rf'^MT:{old_mt}', f'MT:{new_mt}', regex=True)
    exptable.loc[exptable.NODE == node, 'REAC'] = t

replace_mt('exp_602', 3, 4)
replace_mt('exp_685', 3, 4)
replace_mt('exp_605', 3, 4)
replace_mt('exp_666', 3, 4)
replace_mt('exp_600', 3, 4)
replace_mt('exp_608', 3, 4)
replace_mt('exp_631', 3, 4)
replace_mt('exp_1012', 3, 4)
replace_mt('exp_6001', 4, 3)


# initialize the normalization errors
priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

# some convenient shortcuts
priorvals = priortable.PRIOR.to_numpy()
expvals = exptable.DATA.to_numpy()

# speed up the pdf log_prob calculations exploiting the block diagonal structure
expcov_list, idcs_tuples = create_datablock_covmat_list(db['datablock_list'], relative=True)
# variation-01: remove certain points in datablocks
for i in range(len(expcov_list)):
    cur_idcs = np.arange(idcs_tuples[i][0], idcs_tuples[i][1]+1)
    cur_idcs = cur_idcs[np.isin(cur_idcs, exp_keep_idcs)] - idcs_tuples[i][0]
    expcov_list[i] = csr_matrix(expcov_list[i].toarray()[np.ix_(cur_idcs, cur_idcs)])

expcov_list = [x for x in expcov_list if x.shape != (0, 0)]

# variation-01 end
expchol_list = [tf.linalg.cholesky(x.toarray()) for x in expcov_list]
expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
        x, is_non_singular=True, is_square=True
    ) for x in expchol_list]

# generate a restricted mapping blending out the fixed parameters
is_adj = priorcov.diagonal() != 0.
adj_idcs = np.where(is_adj)[0]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
)
propfun = tf.function(restrimap.propagate)
jacfun = tf.function(restrimap.jacobian)

# generate the experimental covariance matrix
expcov_chol = tf.linalg.LinearOperatorBlockDiag(
    expchol_op_list, is_non_singular=True, is_square=True)
expcov_linop = tf.linalg.LinearOperatorComposition(
    [expcov_chol, expcov_chol.adjoint()],
    is_self_adjoint=True, is_positive_definite=True
)

# generate the prior distribution
is_adj_constr = is_adj & np.isfinite(priorcov.diagonal())
is_adj_constr_idcs = np.where(is_adj_constr)[0]
priorcov_chol = tf.linalg.LinearOperatorDiag(np.sqrt(priorcov.diagonal()[is_adj_constr]))
prior_red = MultivariateNormal(priorvals[is_adj_constr], priorcov_chol)
prior_red.log_prob_hessian(priorvals[is_adj_constr])

prior = DistributionForParameterSubset(
    prior_red, len(adj_idcs), is_adj_constr_idcs
)

likelihood = MultivariateNormalLikelihood(
    len(adj_idcs), propfun, jacfun, expvals, expcov_chol, approximate_hessian=True, relative=True
)

# combine prior and likelihood into posterior
post = UnnormalizedDistributionProduct([prior, likelihood])

usu_df = []  # quick and dirty

save_objects('output/01_model_preparation_output.pkl', locals(),
             'post', 'likelihood', 'priorvals', 'is_adj',
             'priortable', 'exptable', 'expcov', 'compmap', 'restrimap')
