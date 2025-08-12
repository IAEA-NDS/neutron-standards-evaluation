import re
import pandas as pd
from scipy.sparse import block_diag, csr_matrix, diags
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
    MultivariateNormalLikelihoodWithCovParams,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)

tfd = tfp.distributions
tfb = tfp.bijectors

priortable = pd.DataFrame({
    'NODE': 'xsid_1',
    'REAC': 'MT:1-R1:1',
    'ENERGY': np.arange(1, 10),
    'PRIOR': 17.0,
    'UNC': np.inf,
    'DESCR': 'channel'
})
priorcov = diags(priortable['UNC'])


# prepare experimental quantities
rng = np.random.default_rng()

energies = np.arange(1, 10)
true_values = np.array([20] * len(energies), dtype=float)
sys_unc = 0.1
stat_unc = 0.01
num_datasets = 2 
sys_unc = rng.standard_normal(num_datasets) * sys_unc
sys_err = [rng.standard_normal(1)*s for s in sys_unc] 
sys_err = [-0.1, 0.1]

expdata_list = []
for expid in range(num_datasets):
    expvalues = true_values.copy()
    expvalues += true_values * rng.standard_normal(len(true_values)) * stat_unc
    expvalues += true_values * sys_err[expid]
    expdata = pd.DataFrame({
        'NODE': f'exp_{expid}',
        'REAC': 'MT:1-R1:1',
        'ENERGY': energies,
        'DATA': expvalues,
        # correct: 'UNC': np.sqrt(stat_unc**2 + sys_unc**2), 
        'UNC': stat_unc,
        'DESCR': f'experiment {expid}'
    })
    expdata_list.append(expdata)


exptable = pd.concat(expdata_list)
expcov = diags(np.square(exptable['UNC'].tolist()))

# initialize the normalization errors
priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

# some convenient shortcuts
priorvals = priortable.PRIOR.to_numpy()
expvals = exptable.DATA.to_numpy()

expcov_list = [expcov]

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

# relevant USU error contributions
# abs U5(n,f) at 1, 5, 15 MeV (clear USU around 2 MeV region)
# abs PU9(n,f) at 1, 5 MeV (likely no USU but to be conservative)
# shape U5(n,f) at 0, 1, 5, 15 MeV (likely USU in the low energy range (not thermal), at about 2 MeV nd at 15 MeV)
# shape PU9(n,f) at 1, 5 MeV (likely USU at about 2 MeV)
# MT:3-R1:10-R2:8 at 0, 1, 5, 15, 30, 100, 200
# MT:3-R1:9-R2:8 at 0, 1, 5, 15, 30, 60
# MT:4-R1:10-R2:8 at 1, 5 MeV (likely USU in 1-5 MeV range)
# MT:4-R1:9-R2:8 at 0, 1, 5, 15 (likely USU at

usu_dfs = []
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:1-R1:1',), (1., 9.), (1e-2,)*2))
usu_df = pd.concat(usu_dfs, ignore_index=True)

usu_map = EnergyDependentAbsoluteUSUMap((usu_df, exptable), reduce=True)
usu_jac = tf.sparse.to_dense(usu_map.jacobian(usu_df.PRIOR.to_numpy()))

def create_like_cov_fun(usu_df, expcov_linop, Smat):

    def map_uncertainties(u):
        ids = np.zeros((len(usu_df),), dtype=np.int32)
        for index, row in red_usu_df.iterrows():
            reac = row.REAC
            energy = row.ENERGY
            cur_idcs = usu_df.index[
                (usu_df.REAC == reac) & (usu_df.ENERGY == energy)
            ].to_numpy()
            ids[cur_idcs] = index
        # scatter the uncertainties to the appropriate places
        tf_ids = tf.constant(ids, dtype=tf.int32)
        uncs = tf.nn.embedding_lookup(u, tf_ids)
        return uncs

    def like_cov_fun(u):
        uncs = map_uncertainties(u)
        # covop = tf.linalg.LinearOperatorLowRankUpdate(
        covop = tf.linalg.LinearOperatorLowRankUpdate(
            expcov_linop, Smat, tf.square(uncs) + 1e-7,
            is_self_adjoint=True, is_positive_definite=True,
            is_diag_update_positive=True
        )
        return covop
    red_usu_df = usu_df[['REAC', 'ENERGY']].drop_duplicates()
    red_usu_df.sort_values(
        ['REAC', 'ENERGY'], ascending=True, ignore_index=True, inplace=True
    )
    return like_cov_fun, red_usu_df


like_cov_fun, red_usu_df = create_like_cov_fun(usu_df, expcov_linop, usu_jac)
num_covpars = len(red_usu_df)

# generate the prior distribution
is_adj_constr = is_adj & np.isfinite(priorcov.diagonal())
is_adj_constr_idcs = np.where(is_adj_constr)[0]
priorcov_chol = tf.linalg.LinearOperatorDiag(np.sqrt(priorcov.diagonal()[is_adj_constr]))
prior_red = MultivariateNormal(priorvals[is_adj_constr], priorcov_chol)
prior_red.log_prob_hessian(priorvals[is_adj_constr])
prior = DistributionForParameterSubset(
    prior_red, len(adj_idcs) + num_covpars, is_adj_constr_idcs
)

# generate the likelihood
likelihood = MultivariateNormalLikelihoodWithCovParams(
    len(adj_idcs), num_covpars, propfun, jacfun, expvals, like_cov_fun,
    approximate_hessian=True, relative=True
)

# combine prior and likelihood into posterior
# post = UnnormalizedDistributionProduct([prior, likelihood])
post = likelihood

save_objects('output/01_model_preparation_output.pkl', locals(),
             'post', 'likelihood', 'priorvals', 'is_adj', 'usu_df', 'red_usu_df',
             'num_covpars', 'priortable', 'exptable', 'expcov', 'like_cov_fun', 'compmap', 'restrimap')
