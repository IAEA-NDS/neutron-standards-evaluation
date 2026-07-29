"""Regularized-horseshoe USU fit, warm-started from the v2
(half-normal) solution: tau from the median nonzero v2 band, local
scales lambda_d = sigma_d^v2 / tau inverted through the slab formula,
physics parameters from the v2 mode."""
import sys
sys.path.insert(0, '/home/gschnabel/Seafile/OmegaSpace/gmapy')
import numpy as np
import tensorflow as tf
from gmapy.tf_uq.inference import determine_MAP_estimate_precond_lbfgs
from gmapy.data_management.object_utils import load_objects, save_objects

OUTDIR = 'output_dsdisc_v3_uli0'
HS_SLAB = 0.3


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.)


def inv_softplus(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 30., y, np.log(np.expm1(np.clip(y, 1e-10, None))))


post, likelihood, red_usu_df, num_covpars, n_shared = load_objects(
    f'{OUTDIR}/01_model_preparation_output.pkl',
    'post', 'likelihood', 'red_usu_df', 'num_covpars', 'n_shared')
v2_optres, v2_red = load_objects(
    'output_dsdisc_v2_uli0/02_parameter_optimization_output.pkl',
    'optres', 'red_usu_df')

num_params = likelihood._num_params
x_v2 = np.array(v2_optres.position)
pars_ref = x_v2[:num_params]
theta_v2 = x_v2[num_params:]
n_ds = num_covpars - n_shared - 1
assert len(theta_v2) == n_shared + n_ds

sigma_ds_v2 = softplus(theta_v2[n_shared:])
active = sigma_ds_v2[sigma_ds_v2 > 0.01]
tau_init = float(np.median(active)) if len(active) else 0.03
# invert the slab formula so the effective bands start at their v2
# values: tau*lambda = c*sigma/sqrt(c^2 - sigma^2)
sig = np.minimum(sigma_ds_v2, 0.97 * HS_SLAB)
tl = HS_SLAB * sig / np.sqrt(HS_SLAB**2 - sig**2)
lam_init = np.clip(tl / tau_init, 1e-3, None)

theta0 = np.concatenate([
    theta_v2[:n_shared],
    inv_softplus(lam_init),
    inv_softplus([tau_init])
])
print(f'[V3] warm start: tau = {tau_init*100:.2f}%, '
      f'{np.sum(sigma_ds_v2 > 0.02)} v2 bands above 2%', flush=True)

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
tau_fit = float(softplus(theta_fit[-1]))
lam_fit = softplus(theta_fit[n_shared:-1])
tl2 = (tau_fit * lam_fit)**2
sig_eff = np.sqrt(HS_SLAB**2 * tl2 / (HS_SLAB**2 + tl2))
usu = np.concatenate([softplus(theta_fit[:n_shared]), sig_eff, [tau_fit]])
red_usu_df['USU'] = usu
save_objects(f'{OUTDIR}/02_parameter_optimization_output.pkl',
             locals(), 'optres', 'red_usu_df')
print('[V3] final objective:', float(np.array(optres.objective_value)),
      flush=True)
print(f'[V3] fitted global scale tau = {tau_fit*100:.2f}%', flush=True)
print(f'[V3] per-dataset bands > 2%: {np.sum(sig_eff > 0.02)}  '
      f'(v2: {np.sum(sigma_ds_v2 > 0.02)});  > 10%: '
      f'{np.sum(sig_eff > 0.10)}', flush=True)

sh = red_usu_df[red_usu_df.TYPE == 'shared'].sort_values(
    'USU', ascending=False)
print('[V3] top shared floor bands:', flush=True)
for _, row in sh.head(10).iterrows():
    print(f'[V3]   {row.USU*100:6.1f}%  {row.REAC} @ {row.ENERGY:g} MeV',
          flush=True)
ds = red_usu_df[red_usu_df.TYPE == 'dataset'].sort_values(
    'USU', ascending=False)
print('[V3] top per-dataset discrepancy bands:', flush=True)
for _, row in ds.head(15).iterrows():
    print(f'[V3]   {row.USU*100:6.1f}%  {row.NODE}  ({row.REAC}, '
          f'{row.NPTS} pts)', flush=True)
