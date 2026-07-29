"""Softplus per-dataset discrepancy fit with moment-based warm start.

Reference mode for the moment estimator: physics parameters of the
well-behaved uli3 fit (consensus not yet dragged by the unclamped
datasets). For each dataset the GLS coefficient estimate of its
Legendre discrepancy modes is compared with the sampling covariance
implied by the stated uncertainties; the positive excess gives the
initial band magnitude.
"""
import sys
sys.path.insert(0, '/home/gschnabel/Seafile/OmegaSpace/gmapy')
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_factor, cho_solve
from gmapy.tf_uq.inference import determine_MAP_estimate_precond_lbfgs
from gmapy.data_management.object_utils import load_objects, save_objects

OUTDIR = 'output_dsdisc_sp_uli0'

post, likelihood, exptable, red_usu_df, num_covpars, smat_np, col2ds = \
    load_objects(f'{OUTDIR}/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'exptable', 'red_usu_df',
                 'num_covpars', 'smat_np', 'col2ds')
ref_optres, = load_objects(
    'output_newds_uli3/02_parameter_optimization_output.pkl', 'optres')

num_params = likelihood._num_params
pars_ref = np.array(ref_optres.position)[:num_params]
p = np.reshape(np.array(likelihood._propfun(
    tf.constant(pars_ref, dtype=tf.float64))), -1)
y = np.reshape(np.array(likelihood._like_data, dtype=float), -1)
z = (y - p) / p

# experimental covariance blocks (relative units)
base = likelihood._covmodel._base_operator
op = base
while not isinstance(op, tf.linalg.LinearOperatorBlockDiag):
    op = op.operators[0]
sizes = [int(o.domain_dimension) for o in op.operators]
edges = np.concatenate([[0], np.cumsum(sizes)])
cfull = np.array(base.to_dense())

# moment estimate of the discrepancy magnitude per dataset
u_init = np.full(num_covpars, 5e-3)
for b in range(len(sizes)):
    sl = slice(edges[b], edges[b+1])
    cb = cfull[sl, sl]
    cf = cho_factor(0.5 * (cb + cb.T))
    zb = z[sl]
    smat_b = smat_np[sl, :]
    active_cols = np.where(np.any(smat_b != 0., axis=0))[0]
    for ids in np.unique(col2ds[active_cols]):
        cols = active_cols[col2ds[active_cols] == ids]
        sb = smat_b[:, cols]
        w = cho_solve(cf, sb)
        gmat = sb.T @ w
        ginv = np.linalg.inv(gmat)
        ahat = ginv @ (w.T @ zb)
        u2 = (ahat @ ahat - np.trace(ginv)) / len(cols)
        u_init[ids] = np.clip(np.sqrt(max(u2, 0.)), 5e-3, 2.0)

theta0 = np.log(np.expm1(u_init))
print(f'[SP] moment warm start: {np.sum(u_init > 0.02)} datasets '
      f'initialized above 2%, max {u_init.max()*100:.1f}%', flush=True)
top = np.argsort(-u_init)[:10]
for i in top:
    print(f'[SP]   init {u_init[i]*100:6.1f}%  {red_usu_df.NODE.iloc[i]}'
          f'  ({red_usu_df.REAC.iloc[i]})', flush=True)

refvals = tf.constant(np.concatenate([pars_ref, theta0]), dtype=tf.float64)
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian
batched_neg_log_prob = tf.function(post.neg_log_prob_batch)

optres = determine_MAP_estimate_precond_lbfgs(
    refvals, neg_log_prob_and_gradient, neg_log_post_hessian,
    max_iters=10000, nugget=1e-3, hessian_refresh_interval=100,
    batch_neg_log_prob=batched_neg_log_prob, saddle_free='auto',
    checkpoint_file=f'{OUTDIR}/02_optimizer_checkpoint.npz',
    ret_optres=True, must_converge=True
)
theta_fit = optres.position.numpy()[-num_covpars:]
red_usu_df['USU'] = np.log1p(np.exp(-np.abs(theta_fit))) \
    + np.maximum(theta_fit, 0.)  # numerically stable softplus
save_objects(f'{OUTDIR}/02_parameter_optimization_output.pkl',
             locals(), 'optres', 'red_usu_df')
print('[SP] final objective:', float(np.array(optres.objective_value)),
      flush=True)
usu = red_usu_df.sort_values('USU', ascending=False)
print('[SP] top fitted discrepancy bands:', flush=True)
for _, row in usu.head(15).iterrows():
    print(f'[SP]   {row.USU*100:6.1f}%  {row.NODE}  ({row.REAC}, '
          f'{row.NPTS} pts)', flush=True)
