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
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)

import sys
sys.path.append('../gmapy/tnc_axton_eval')
import tnc_inference_prep as prep

from gmapy.tf_uq.inference import (
    determine_MAP_estimate,
    generate_MCMC_chain,
)
from gmapy.mcmc_inference import compute_effective_sample_size

# do a MAP
# optres = determine_MAP_estimate(prep.startvals_tf, prep.func_and_grad_tf, prep.func_hessian_tf, ret_optres=True)
# opt_params = (prep.trafo(optres.position)).numpy()

# do MCMC
# log_prob = lambda x: (-prep.chisquare(x))
# chain, _ = generate_MCMC_chain(prep.startvals_tf, log_prob, prep.chisquare_hessian, num_leapfrog_steps=3, step_size=0.1, num_results=1000)
# opt_params_mcmc = np.mean(prep.trafo(chain).numpy(), axis=0)


tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = '../data/data.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

gma_priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

# TODO: remove the thermal neutron constants
exptable.loc[exptable.ENERGY == 2.530000e-08]
# find TNC values with cross correlations to higher energies
ds_ids = exptable.loc[exptable.ENERGY == 2.530000e-08, 'NODE']
tmp = exptable[exptable.NODE.isin(ds_ids)]


# variation-01: remove specific experimental datasets after visual inspection
exp_remove_mask = (exptable.NODE == 'exp_722') & (exptable.ENERGY > 23)  # Ponkratov U5(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_874') & (exptable.ENERGY > 23)  # Ponkratov U8(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_524') & (exptable.ENERGY > 27)  # A.D. Carlson PU5(n,f) above 27 MeV
# remove due to recommendation in excel sheet
exp_remove_mask |= (exptable.NODE == 'exp_8029')
# remove Maslov's patch
exp_remove_mask |= (exptable.NODE == 'exp_1003')

# remove Lounsbury
exp_remove_mask |= (exptable.NODE == 'exp_8099')
exp_remove_mask |= (exptable.NODE == 'exp_8098')
exp_remove_mask |= (exptable.NODE == 'exp_8097')

# remove Axton points
for i in  range(910, 935):
    exp_remove_mask |= (exptable.NODE == f'exp_{i}')




exp_keep_idcs = np.where(~exp_remove_mask)[0]
exptable = exptable.loc[exp_keep_idcs].reset_index(drop=True)
expcov = csr_matrix(expcov.toarray()[np.ix_(exp_keep_idcs, exp_keep_idcs)])

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
gma_priortable, priorcov = attach_shape_prior((gma_priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((gma_priortable, exptable), reduce=True)
initialize_shape_prior((gma_priortable, exptable), compmap)

# some convenient shortcuts
gma_priorvals = gma_priortable.PRIOR.to_numpy()
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
    len(gma_priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=gma_priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
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
prior_red = MultivariateNormal(gma_priorvals[is_adj_constr], priorcov_chol)
prior_red.log_prob_hessian(gma_priorvals[is_adj_constr])
prior = DistributionForParameterSubset(
    prior_red, len(adj_idcs), is_adj_constr_idcs
)

likelihood = MultivariateNormalLikelihood(
    len(adj_idcs), propfun, jacfun, expvals, expcov_chol, approximate_hessian=True, relative=True
)

# combine prior and likelihood into posterior

# Replace the reaction notation in GMA reactions by the
# notation of the Axtion reactions. Remove the GMA reactions
# which are associated with unadjustable parameters.
# Also construct a flag that indicates whether a GMA reactions
# is also given in the Axton reactions.
gma_energies = gma_priortable.ENERGY.tolist()
orig_gma_reacs = gma_priortable.REAC.tolist()
gma_reacs = [prep.gma_to_axt_map.get(r,r) for r in orig_gma_reacs]
axt_reacs = prep.axt_prior_reacs
ext_gma_reacs = [f'{r};E={e}' for r, e, a in zip(gma_reacs, gma_energies, is_adj) if a]
ext_axt_reacs = [f'{r};E={2.53e-8}' for r in axt_reacs]
is_axt = np.array([r in ext_axt_reacs for r in ext_gma_reacs])

# Generate separate posterior distributions for the Axton database and the GMA database
gma_post = UnnormalizedDistributionProduct([prior, likelihood])
axt_post = prep.AxtonChiSquareDist()

# Check if GMA posterior distribution is okay
gma_priorvals = gma_priortable.loc[is_adj, 'PRIOR'].to_numpy()
# gma_post.log_prob(gma_priorvals)
# assert len(ext_gma_reacs) == len(gma_priorvals)

# Check if Axton posterior distribution is okay
axt_priorvals = prep.startvals
axt_post.log_prob(axt_priorvals)
assert len(axt_priorvals) == len(ext_axt_reacs)

# For calculating the deduplicated input prior, merge GMA and Axton
# reactions, but skip GMA reactoins that also occur in Axton
red_ext_gma_reacs = [r for r, s in zip(ext_gma_reacs, is_axt) if not s]
ext_reacs = red_ext_gma_reacs + ext_axt_reacs

priorvals = np.sqrt(np.concatenate([gma_priorvals[~is_axt], axt_priorvals]))

# Construct the posterior compound object
post = prep.GmaAxtDist(ext_gma_reacs, gma_post, ext_axt_reacs, axt_post, ext_reacs)

priortable = pd.concat([gma_priortable[is_adj][~is_axt], prep.axt_priortable], ignore_index=True)

# post.log_prob(priortable.PRIOR.to_numpy())
# post.log_prob_hessian(priorvals)

save_objects('output/01_model_preparation_output.pkl', locals(),
             'post', 'priorvals', 'priortable', 'exptable', 'expcov', 'compmap', 'restrimap')
