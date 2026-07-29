import os
import sys
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_factor, cho_solve
from gmapy.data_management.object_utils import load_objects

outdir = sys.argv[1] if len(sys.argv) > 1 else 'output'

post, likelihood, exptable = load_objects(
    f'{outdir}/01_model_preparation_output.pkl',
    'post', 'likelihood', 'exptable'
)
optres, = load_objects(
    f'{outdir}/02_parameter_optimization_output.pkl', 'optres'
)
x_mode = np.array(optres.position)
num_params = likelihood._num_params
pars_mode = x_mode[:num_params]
covpars_mode = x_mode[num_params:]
m = len(exptable)

pars_t = tf.constant(pars_mode, dtype=tf.float64)
y = np.reshape(np.array(likelihood._like_data, dtype=float), -1)
p = np.reshape(np.array(likelihood._propfun(pars_t)), -1)
z = (y - p) / p
jac = np.array(tf.sparse.to_dense(likelihood._jacfun(pars_t)))
kmat = jac * (y / p**2)[:, None]

covmodel = likelihood._covmodel
base_op = covmodel._base_operator
blockdiag = base_op
while not isinstance(blockdiag, tf.linalg.LinearOperatorBlockDiag):
    blockdiag = blockdiag.operators[0]
block_sizes = [int(op.domain_dimension) for op in blockdiag.operators]
assert sum(block_sizes) == m
edges = np.concatenate([[0], np.cumsum(block_sizes)])
cexp = np.array(base_op.to_dense())
smat = np.array(covmodel._smat)
dvals = np.array(covmodel._diag_fun(
    tf.constant(covpars_mode, dtype=tf.float64)
))
print('[CHI2] residuals and covariance assembled', flush=True)

# posterior precision (pars block, conditional on fitted u)
lam = np.array(post.neg_log_prob_hessian(
    tf.constant(x_mode, dtype=tf.float64)
))
lam_pars = 0.5 * (lam[:num_params, :num_params]
                  + lam[:num_params, :num_params].T)
cf = cho_factor(lam_pars)
wmat = cho_solve(cf, kmat.T)          # Lambda^-1 K^T  (n x m)
print('[CHI2] hat-matrix solve done', flush=True)

# global chi-square without and with the USU contribution
zsol = np.zeros(m)
chi2_blocks = np.zeros(len(block_sizes))
hat_blocks = np.zeros(len(block_sizes))
for b in range(len(block_sizes)):
    sl = slice(edges[b], edges[b+1])
    cb = cexp[sl, sl]
    zb = z[sl]
    sol = np.linalg.solve(cb, zb)
    zsol[sl] = sol
    chi2_blocks[b] = zb @ sol
    mb = kmat[sl] @ wmat[:, sl]
    hat_blocks[b] = np.trace(np.linalg.solve(cb, mb))
chi2_exp = float(np.sum(chi2_blocks))
c0 = cexp + (smat * dvals[None, :]) @ smat.T
chi2_c0 = float(z @ np.linalg.solve(0.5*(c0+c0.T), z))
p_eff = float(np.sum(hat_blocks))
print(f'[CHI2] global: chi2(no USU) = {chi2_exp:.1f}   '
      f'chi2(with fitted USU) = {chi2_c0:.1f}   points = {m}', flush=True)
print(f'[CHI2] effective params absorbed by data: {p_eff:.1f}  '
      f'-> dof ~ {m - p_eff:.1f}', flush=True)
print(f'[CHI2] chi2/dof (no USU)  : {chi2_exp/(m - p_eff):.3f}', flush=True)
print(f'[CHI2] chi2/dof (with USU): {chi2_c0/(m - p_eff):.3f}', flush=True)

# per reaction class: observed vs expected chi-square
reacs = exptable.REAC.to_numpy().astype(str)
point_chi2 = z * zsol
point_reac = reacs
rows = []
for r in np.unique(point_reac):
    sel = point_reac == r
    npts = int(np.sum(sel))
    chi2_r = float(np.sum(point_chi2[sel]))
    # expected value: points minus hat-trace share (approximated by
    # distributing block hat over blocks fully inside the class)
    hat_r = 0.
    for b in range(len(block_sizes)):
        sl = slice(edges[b], edges[b+1])
        inblk = np.sum(sel[sl])
        if inblk > 0:
            hat_r += hat_blocks[b] * inblk / block_sizes[b]
    exp_r = npts - hat_r
    rows.append((chi2_r / max(exp_r, 1e-6), chi2_r, exp_r, npts, r))
rows.sort(reverse=True)
print('[CHI2] === reaction classes ranked by chi2/expected ===', flush=True)
print('[CHI2]  ratio    chi2   expected  points  reaction', flush=True)
for ratio, chi2_r, exp_r, npts, r in rows:
    if npts >= 10:
        print(f'[CHI2] {ratio:6.2f} {chi2_r:8.1f} {exp_r:9.1f} '
              f'{npts:7d}  {r}', flush=True)
