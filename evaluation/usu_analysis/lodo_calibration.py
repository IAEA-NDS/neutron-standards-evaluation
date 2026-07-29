"""Leave-one-dataset-out predictive calibration on the v3b model.

For every dataset d: predict its residuals from all other data using
the linearized model around the v3b mode (shared bands treated as
explicit low-rank Gaussian effects so the data covariance stays
block-diagonal), and score the predictive chi-square in two variants:
  'claimed'       - predictive covariance includes d's own fitted band
  'out-of-sample' - excludes d's own band (it was fitted on d itself)
"""
import sys
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_factor, cho_solve
from gmapy.data_management.object_utils import load_objects

outdir = sys.argv[1] if len(sys.argv) > 1 else 'output_dsdisc_v3b_uli0'

post, likelihood, exptable, smat_np, col2par, n_shared = load_objects(
    f'{outdir}/01_model_preparation_output.pkl',
    'post', 'likelihood', 'exptable', 'smat_np', 'col2par', 'n_shared')
optres, = load_objects(
    f'{outdir}/02_parameter_optimization_output.pkl', 'optres')

x_mode = np.array(optres.position)
xt = tf.constant(x_mode, dtype=tf.float64)
num_params = likelihood._num_params
pars_t = tf.constant(x_mode[:num_params], dtype=tf.float64)
covpars = tf.constant(x_mode[num_params:], dtype=tf.float64)

y = np.reshape(np.array(likelihood._like_data, dtype=float), -1)
p = np.reshape(np.array(likelihood._propfun(pars_t)), -1)
z = (y - p) / p
jac = np.array(tf.sparse.to_dense(likelihood._jacfun(pars_t)))
kmat = jac * (y / p**2)[:, None]
m = len(exptable)

comps = list(post._distributions)
nonlik = [comps[0], comps[2]]  # prior, barrier
lam_pb = -sum(np.array(c.log_prob_hessian(xt)) for c in nonlik)
lam_pb = 0.5 * (lam_pb + lam_pb.T)[:num_params, :num_params]
g_pb = -sum(np.array(c.log_prob_and_gradient(xt)[1])
            for c in nonlik)[:num_params]

dvals = np.array(likelihood._covmodel._diag_fun(covpars))
sh_cols = np.where(col2par < n_shared)[0]
ds_cols = np.where(col2par >= n_shared)[0]
s_sh = smat_np[:, sh_cols]
d_sh = dvals[sh_cols]
s_ds = smat_np[:, ds_cols]
d_ds = dvals[ds_cols]
col2ds_par = col2par[ds_cols]

base_op = likelihood._covmodel._base_operator
op = base_op
while not isinstance(op, tf.linalg.LinearOperatorBlockDiag):
    op = op.operators[0]
sizes = [int(o.domain_dimension) for o in op.operators]
edges = np.concatenate([[0], np.cumsum(sizes)])
cexp = np.array(base_op.to_dense())

# augmented parameters (physics, shared-band effects a)
naug = num_params + len(sh_cols)
k_aug = np.hstack([kmat, s_sh])
lam_prior = np.zeros((naug, naug))
lam_prior[:num_params, :num_params] = lam_pb
lam_prior[num_params:, num_params:] = np.diag(1. / d_sh)
rhs_prior = np.concatenate([-g_pb, np.zeros(len(sh_cols))])

node_arr = exptable.NODE.to_numpy().astype(str)
reac_arr = exptable.REAC.to_numpy().astype(str)

lam_full = lam_prior.copy()
rhs_full = rhs_prior.copy()
block_data = []
for b in range(len(sizes)):
    sl = slice(edges[b], edges[b+1])
    cb = cexp[sl, sl].copy()
    scols = s_ds[sl, :]
    active = np.where(np.any(scols != 0., axis=0))[0]
    cb += (scols[:, active] * d_ds[active][None, :]) @ scols[:, active].T
    cb = 0.5 * (cb + cb.T)
    cf = cho_factor(cb)
    kb = k_aug[sl]
    zb = z[sl]
    w = cho_solve(cf, kb)
    qb = kb.T @ w
    rb = w.T @ zb
    lam_full += qb
    rhs_full += rb
    block_data.append((sl, cb, qb, rb))
print('[LODO] full augmented system assembled', flush=True)

results = []
for b, (sl, cb, qb, rb) in enumerate(block_data):
    rows = np.arange(sl.start, sl.stop)
    for nd in np.unique(node_arr[sl]):
        dmask = node_arr[rows] == nd
        di = rows[dmask]
        oi = rows[~dmask]
        kd = k_aug[di]
        zd = z[di]
        cdd = cb[np.ix_(dmask, ~dmask)] if False else None
        c_dd = cb[np.ix_(dmask, dmask)]
        if len(oi):
            c_oo = cb[np.ix_(~dmask, ~dmask)]
            c_do = cb[np.ix_(dmask, ~dmask)]
            ko = k_aug[oi]
            zo = z[oi]
            cfo = cho_factor(0.5 * (c_oo + c_oo.T))
            wo = cho_solve(cfo, ko)
            qo = ko.T @ wo
            ro = wo.T @ zo
            a_do = cho_solve(cfo, c_do.T).T   # C_do C_oo^-1
            k_tilde = kd - a_do @ ko
            c_cond = c_dd - a_do @ c_do.T
        else:
            qo = np.zeros_like(qb)
            ro = np.zeros_like(rb)
            a_do = None
            k_tilde = kd
            c_cond = c_dd
        lam_rest = lam_full - qb + qo
        rhs_rest = rhs_full - rb + ro
        cfr = cho_factor(
            lam_rest + 1e-10 * np.max(np.diag(lam_rest)) * np.eye(naug))
        theta = cho_solve(cfr, rhs_rest)
        hk = cho_solve(cfr, k_tilde.T)
        s_par = k_tilde @ hk
        if len(oi):
            pred = kd @ theta + a_do @ (zo - ko @ theta)
        else:
            pred = kd @ theta
        r = zd - pred
        # own-band block of dataset d (columns tied to its covpar)
        own_cols = np.where(np.any(s_ds[di][:, :] != 0., axis=0))[0]
        own = (s_ds[np.ix_(di, own_cols)] * d_ds[own_cols][None, :]) \
            @ s_ds[np.ix_(di, own_cols)].T
        s_claim = 0.5 * (c_cond + c_cond.T) + s_par
        s_oos = s_claim - own
        out = []
        for smat in (s_claim, s_oos):
            try:
                cfs = cho_factor(0.5 * (smat + smat.T))
                out.append(float(r @ cho_solve(cfs, r)))
            except Exception:
                out.append(np.nan)
        results.append((nd, reac_arr[di[0]], len(di), out[0], out[1]))
    if b % 40 == 0:
        print(f'[LODO] block {b}/{len(block_data)}', flush=True)

nds = np.array([r[2] for r in results])
c_cl = np.array([r[3] for r in results])
c_os = np.array([r[4] for r in results])
ok = np.isfinite(c_cl) & np.isfinite(c_os)
print(f'[LODO] datasets scored: {ok.sum()}/{len(results)}', flush=True)
print(f'[LODO] GLOBAL predictive chi2/n  claimed: '
      f'{c_cl[ok].sum()/nds[ok].sum():.3f}   out-of-sample: '
      f'{c_os[ok].sum()/nds[ok].sum():.3f}', flush=True)
print(f'[LODO] implied calibration factors (sqrt): claimed '
      f'{np.sqrt(c_cl[ok].sum()/nds[ok].sum()):.3f}  oos '
      f'{np.sqrt(c_os[ok].sum()/nds[ok].sum()):.3f}', flush=True)
frac = np.mean(c_cl[ok]/nds[ok] > 2.)
print(f'[LODO] fraction of datasets with claimed chi2/n > 2: {frac:.2f}',
      flush=True)

# channel aggregation (out-of-sample)
agg = {}
for nd, reac, n, ccl, cos_ in results:
    if not np.isfinite(cos_):
        continue
    a = agg.setdefault(reac, [0., 0., 0.])
    a[0] += cos_; a[1] += n; a[2] += ccl
rows = sorted(((v[0]/v[1], v[2]/v[1], int(v[1]), k) for k, v in agg.items()
               if v[1] >= 20), reverse=True)
print('[LODO] channels by out-of-sample predictive chi2/n '
      '(claimed in parens):', flush=True)
for oosr, clr, n, k in rows[:14]:
    print(f'[LODO]   {oosr:8.2f} ({clr:6.2f})  n={n:5d}  {k}', flush=True)
print('[LODO] U5 direct channels:', flush=True)
for k in ('MT:1-R1:8', 'MT:2-R1:8', 'MT:6-R1:8'):
    if k in agg:
        v = agg[k]
        print(f'[LODO]   {v[0]/v[1]:8.2f} ({v[2]/v[1]:6.2f})  '
              f'n={int(v[1]):5d}  {k}', flush=True)
top = sorted((r for r in results if np.isfinite(r[4])),
             key=lambda r: -r[4]/r[2])[:12]
print('[LODO] worst datasets (out-of-sample chi2/n):', flush=True)
for nd, reac, n, ccl, cos_ in top:
    print(f'[LODO]   {cos_/n:9.1f}  {nd}  ({reac}, {n} pts)', flush=True)
