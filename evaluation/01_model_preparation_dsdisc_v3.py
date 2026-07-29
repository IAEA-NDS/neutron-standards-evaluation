import sys
import pathlib
# resolve gmapy to the submodule of this repository, which pins the
# version with the vectorized compound map and exact Hessian support
sys.path.insert(
    0, (pathlib.Path(__file__).resolve().parents[1] / 'gmapy').as_posix()
)
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
from gmapy.mappings.tf.vectorized_compound_map_tf \
    import VectorizedCompoundMap as CompoundMapTF
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
    UnnormalizedDistributionProduct,
    LogBarrier,
    SoftplusHalfNormal,
    SoftplusHalfCauchy
)
from gmapy.tf_uq.covariance_models import LowRankCovarianceModel

tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
# (path and output directory overridable via environment variables
# for side-by-side runs with modified databases)
import os
db_path = os.environ.get('EVAL_DB_PATH', '../data/data.json')
outdir = os.environ.get('EVAL_OUTPUT_DIR', 'output')
os.makedirs(outdir, exist_ok=True)
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

# EXPERIMENTAL, disabled: de-spike the third uncertainty component,
# which the DATP preprocessing enlarged on points deviating more than
# ULI=3 sigma from the a-priori. Both wholesale removal and capping
# at the dataset median proved statistically invalid: component 3
# carries most of the per-point uncorrelated variance, and shrinking
# it drives datablock covariance blocks towards singularity (chi2 and
# hat-matrix traces explode). The clean approach is to regenerate the
# database from the pre-DATP crd files with ULI=0 in datpy.
DESPIKE_COMP3 = False
if DESPIKE_COMP3:
    from gmapy.data_management import datablock_api as _dbapi
    for _cur_db in db['datablock_list']:
        for _ds in _dbapi.dataset_iterator(_cur_db):
            if 'CO' in _ds:
                _co = np.array(_ds['CO'], dtype=float)
                if _co.ndim == 2 and _co.shape[1] >= 12:
                    _med = np.median(_co[:, 2])
                    _co[:, 2] = np.minimum(_co[:, 2], _med)
                    _ds['CO'] = _co.tolist()

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

# variation-01: remove specific experimental datasets after visual inspection
exp_remove_mask = (exptable.NODE == 'exp_722') & (exptable.ENERGY > 23)  # Ponkratov U5(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_874') & (exptable.ENERGY > 23)  # Ponkratov U8(n,f) shape beyond 23 MeV
exp_remove_mask |= (exptable.NODE == 'exp_524') & (exptable.ENERGY > 27)  # A.D. Carlson PU5(n,f) above 27 MeV
# remove due to recommendation in excel sheet
exp_remove_mask |= (exptable.NODE == 'exp_8029')
# remove Maslov's patch
exp_remove_mask |= (exptable.NODE == 'exp_1003')

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
    fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs,
    whessfun=compmap.weighted_row_hessian
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

# ---------------------------------------------------------------
# combined USU model (v2):
#   (a) shared per-channel error bands on piecewise-linear energy
#       knots -- the uncertainty floor that cannot be averaged away
#   (b) per-dataset discrepancy components with smooth Legendre
#       energy shapes -- arbitration between conflicting datasets;
#       shape datasets (fitted normalization) get no constant mode
# both parameterized via softplus magnitudes; the per-dataset part
# additionally carries a weak half-normal restraint (see below)
# ---------------------------------------------------------------
usu_dfs = []
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:1-R1:8',), (5e-3, 1e-1, 1., 5., 15., 30.), (1e-2,)*6))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:1-R1:9',), (5e-3, 1e-1, 1., 5., 15., 30.), (1e-2,)*6))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:2-R1:8',), (1e-3, 1e-2, 1e-1, 1., 5., 15., 30.), (1e-2,)*7))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:2-R1:9',), (1e-3, 1e-2, 1e-1, 1., 5., 15., 30.), (1e-2,)*7))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:3-R1:10-R2:8',), (0.1, 1., 5., 15., 30., 100., 200.), (1e-2,)*7))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:3-R1:9-R2:8',), (1e-3, 1e-2, 0.1, 1., 5., 15., 30., 60.), (1e-2,)*8))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:4-R1:10-R2:8',), (0.1, 1., 5., 15., 30.), (1e-2,)*5))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:4-R1:9-R2:8',), (1e-3, 1e-2, 1e-1, 1., 5., 15., 30.), (1e-2,)*7))
usu_df_sh = pd.concat(usu_dfs, ignore_index=True)
# NOTE: no per-dataset exemptions here (variation-01 removed some
# datasets from the shared bands) -- dataset-specific behavior is the
# job of the per-dataset components now

usu_map = EnergyDependentAbsoluteUSUMap((usu_df_sh, exptable), reduce=True)
usu_jac = tf.sparse.to_dense(usu_map.jacobian(usu_df_sh.PRIOR.to_numpy()))

red_sh = usu_df_sh[['REAC', 'ENERGY']].drop_duplicates()
red_sh.sort_values(['REAC', 'ENERGY'], ascending=True,
                   ignore_index=True, inplace=True)
sh_ids = np.zeros((len(usu_df_sh),), dtype=np.int32)
for index, row in red_sh.iterrows():
    cur_idcs = usu_df_sh.index[
        (usu_df_sh.REAC == row.REAC) & (usu_df_sh.ENERGY == row.ENERGY)
    ].to_numpy()
    sh_ids[cur_idcs] = index
onehot = np.zeros((len(usu_df_sh), len(red_sh)))
onehot[np.arange(len(usu_df_sh)), sh_ids] = 1.
smat_shared_np = np.array(usu_jac) @ onehot
n_shared = smat_shared_np.shape[1]

# per-dataset Legendre discrepancy columns
SHAPE_PREFIXES = ('MT:2-', 'MT:4-', 'MT:8-', 'MT:9-')
node_arr = exptable.NODE.to_numpy().astype(str)
reac_arr = exptable.REAC.to_numpy().astype(str)
energy_arr = exptable.ENERGY.to_numpy().astype(float)

smat_cols = []
col2ds = []
ds_records = []
for nd in pd.unique(node_arr):
    ridcs = np.where(node_arr == nd)[0]
    reac = reac_arr[ridcs[0]]
    is_shape = reac.startswith(SHAPE_PREFIXES)
    en = energy_arr[ridcs]
    loge = np.log10(np.maximum(en, 1e-12))
    span = loge.max() - loge.min()
    ndistinct = len(np.unique(loge))
    if span > 1e-8 and ndistinct >= 2:
        t = 2. * (loge - loge.min()) / span - 1.
    else:
        t = np.zeros_like(loge)
    basis = [] if is_shape else [np.ones_like(t)]
    if ndistinct >= 2 and span > 1e-8:
        basis.append(t)
    if ndistinct >= 3:
        basis.append(0.5 * (3. * t**2 - 1.))
    if not basis:
        continue
    for b in basis:
        col = np.zeros(len(exptable))
        col[ridcs] = b
        smat_cols.append(col)
        col2ds.append(len(ds_records))
    ds_records.append({
        'TYPE': 'dataset', 'NODE': nd, 'REAC': reac,
        'ENERGY': np.nan, 'NPTS': len(ridcs), 'NCOLS': len(basis)
    })

smat_ds_np = np.stack(smat_cols, axis=1)
col2ds = np.array(col2ds, dtype=np.int32)
smat_np = np.hstack([smat_shared_np, smat_ds_np])
col2par = np.concatenate(
    [np.arange(n_shared, dtype=np.int32), n_shared + col2ds]
)
num_covpars = n_shared + len(ds_records) + 1  # +1: global scale tau

sh_records = [
    {'TYPE': 'shared', 'NODE': '', 'REAC': row.REAC,
     'ENERGY': row.ENERGY, 'NPTS': 0, 'NCOLS': 1}
    for _, row in red_sh.iterrows()
]
glob_records = [{'TYPE': 'global', 'NODE': '', 'REAC': '',
                 'ENERGY': np.nan, 'NPTS': 0, 'NCOLS': 0}]
red_usu_df = pd.DataFrame(sh_records + ds_records + glob_records)
usu_df = red_usu_df
print(f'horseshoe USU model: {n_shared} shared knots + '
      f'{len(ds_records)} per-dataset bands, '
      f'{smat_np.shape[1]} columns', flush=True)


def create_like_cov_fun(smat, colmap, n_shared, expcov_linop):
    # regularized-horseshoe scales for the per-dataset columns:
    #   sigma~^2 = c^2 tau^2 lambda^2 / (c^2 + tau^2 lambda^2)
    # with local scales lambda_d = softplus(u_d), one global scale
    # tau = softplus(u[-1]) and slab width c capping the tails
    # (bounded soft rejection); shared floor columns keep plain
    # softplus magnitudes
    HS_SLAB = 0.3
    shared_mask = colmap < n_shared

    def diag_fun(u):
        tau = tf.math.softplus(u[-1])
        sp = tf.math.softplus(tf.gather(u, colmap))
        tl2 = tf.square(tau * sp)
        sig2_hs = HS_SLAB**2 * tl2 / (HS_SLAB**2 + tl2)
        d = tf.where(shared_mask, tf.square(sp), sig2_hs)
        return d + 1e-7
    return LowRankCovarianceModel(
        expcov_linop, tf.constant(smat, dtype=tf.float64), diag_fun
    )


like_cov_fun = create_like_cov_fun(smat_np, col2par, n_shared, expcov_linop)

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
    approximate_hessian=False, relative=True,
    whessfun=restrimap.weighted_hessian
)

# log-barrier enforcing positivity of the physics parameters
# (cross sections, normalization factors); prevents descent into the
# unphysical funnel region of the relative-covariance posterior
# (predictions -> 0 inflate the log-determinant reward)
barrier_qmat = np.hstack(
    [np.eye(len(adj_idcs)), np.zeros((len(adj_idcs), num_covpars))]
)
posbarrier = LogBarrier(barrier_qmat, strength=1e-4)

# combine prior, likelihood and barrier into posterior
# regularized-horseshoe priors: half-Cauchy(1) on the per-dataset
# local scales, half-Cauchy(tau_0) on the global scale; the shared
# floor bands are left unrestrained
lambda_idcs = len(adj_idcs) + n_shared \
    + np.arange(num_covpars - n_shared - 1)
tau_idx = np.array([len(adj_idcs) + num_covpars - 1])
lambda_prior = DistributionForParameterSubset(
    SoftplusHalfCauchy(1.0), len(adj_idcs) + num_covpars, lambda_idcs
)
tau_prior = DistributionForParameterSubset(
    SoftplusHalfCauchy(0.03), len(adj_idcs) + num_covpars, tau_idx
)
post = UnnormalizedDistributionProduct(
    [prior, likelihood, posbarrier, lambda_prior, tau_prior]
)

save_objects(f'{outdir}/01_model_preparation_output.pkl', locals(),
             'post', 'likelihood', 'priorvals', 'is_adj', 'usu_df', 'red_usu_df',
             'num_covpars', 'priortable', 'exptable', 'expcov', 'like_cov_fun', 'compmap', 'restrimap', 'smat_np', 'col2par', 'n_shared')
