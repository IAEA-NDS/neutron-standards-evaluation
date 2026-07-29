import sys
import numpy as np
import tensorflow as tf
from gmapy.data_management.object_utils import load_objects

outdir = sys.argv[1] if len(sys.argv) > 1 else 'output'

post, likelihood, priortable, exptable, is_adj = load_objects(
    f'{outdir}/01_model_preparation_output.pkl',
    'post', 'likelihood', 'priortable', 'exptable', 'is_adj'
)
optres, = load_objects(
    f'{outdir}/02_parameter_optimization_output.pkl', 'optres'
)
x_mode = np.array(optres.position)
xt = tf.constant(x_mode, dtype=tf.float64)
num_params = likelihood._num_params
pars_mode = x_mode[:num_params]
covpars_mode = x_mode[num_params:]
adj_idcs = np.where(np.array(is_adj))[0]
m = len(exptable)

comps = list(post._distributions)
prior = comps[0]
nonlik = [comps[0]] + comps[2:]  # prior, barrier, band priors

# full posterior precision and its prior/barrier parts at the mode
lam_full = np.array(post.neg_log_prob_hessian(xt))
lam_full = 0.5 * (lam_full + lam_full.T)
lam_prior = -sum(np.array(c.log_prob_hessian(xt)) for c in nonlik)
print('[DEC] posterior Hessian computed', flush=True)

# GLS quantities in the C0 space (relative covariance)
pars_t = tf.constant(pars_mode, dtype=tf.float64)
y = np.reshape(np.array(likelihood._like_data, dtype=float), -1)
p = np.reshape(np.array(likelihood._propfun(pars_t)), -1)
jac = np.array(tf.sparse.to_dense(likelihood._jacfun(pars_t)))
kmat = jac * (y / p**2)[:, None]

covmodel = likelihood._covmodel
base_op = covmodel._base_operator
smat = np.array(covmodel._smat)
dvals = np.array(covmodel._diag_fun(
    tf.constant(covpars_mode, dtype=tf.float64)
))
blockdiag = base_op
while not isinstance(blockdiag, tf.linalg.LinearOperatorBlockDiag):
    blockdiag = blockdiag.operators[0]
block_sizes = [int(op.domain_dimension) for op in blockdiag.operators]
assert sum(block_sizes) == m
block_edges = np.concatenate([[0], np.cumsum(block_sizes)])
cexp = np.array(base_op.to_dense())
c0 = cexp + (smat * dvals[None, :]) @ smat.T
c0 = 0.5 * (c0 + c0.T)
c0_chol = np.linalg.cholesky(c0)
print(f'[DEC] dense C0 built ({len(block_sizes)} blocks) and factorized',
      flush=True)


def solve_c0(rhs):
    tmp = np.linalg.solve(c0_chol, rhs)
    return np.linalg.solve(c0_chol.T, tmp)


def decompose(tag, reac, energy):
    mask = (priortable.REAC.str.fullmatch(reac, na=False)
            & priortable.NODE.str.match('xsid_', na=False)
            & (np.abs(priortable.ENERGY - energy) < 1e-9))
    glob = np.array(priortable.index[mask])
    k = int(np.searchsorted(adj_idcs, glob[0]))
    assert adj_idcs[k] == glob[0]
    ek = np.zeros(len(x_mode)); ek[k] = 1.
    # least-squares solve: the posterior precision is singular along
    # flat covpar directions, which are hence treated as conditioned
    w = np.linalg.lstsq(lam_full, ek, rcond=1e-10)[0]
    var_k = w[k]
    sd_rel = 100. * np.sqrt(var_k) / abs(pars_mode[k])
    prior_term = float(w @ (lam_prior @ w))
    u = kmat @ w[:num_params]
    ut = solve_c0(u)
    gls_total = float(u @ ut)
    usu_bands = dvals * (smat.T @ ut)**2
    usu_total = float(np.sum(usu_bands))
    cexp_pt = ut * (cexp @ ut)
    cexp_total = float(np.sum(cexp_pt))
    remainder = var_k - prior_term - gls_total
    print(f'[DEC] ===== {tag} ({reac} @ {energy} MeV) =====', flush=True)
    print(f'[DEC] posterior sd (marginal): {sd_rel:.3f} %', flush=True)
    print(f'[DEC] variance split: prior+barrier {100*prior_term/var_k:6.2f} %'
          f'  exp-data {100*cexp_total/var_k:6.2f} %'
          f'  usu-bands {100*usu_total/var_k:6.2f} %'
          f'  higher-order/covpar {100*remainder/var_k:6.2f} %', flush=True)
    # aggregate experimental contributions by reaction class
    reacs = exptable.REAC.to_numpy()
    contrib = {}
    for r in np.unique(reacs):
        sel = reacs == r
        contrib[r] = float(np.sum(cexp_pt[sel]))
    top = sorted(contrib.items(), key=lambda t: -abs(t[1]))[:10]
    for r, c in top:
        print(f'[DEC]   {100*c/var_k:7.2f} %  {r}', flush=True)
    # energy attribution within the strongest classes
    ens = exptable.ENERGY.to_numpy()
    bins = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 5., 10., 30., 1000.])
    hist = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        sel = (ens >= bins[i]) & (ens < bins[i+1])
        hist[i] = np.sum(cexp_pt[sel])
    print('[DEC]   energy bins (MeV) of exp contributions:', flush=True)
    for i in range(len(bins) - 1):
        if abs(hist[i]) > 0.005 * var_k:
            print(f'[DEC]   {bins[i]:7.1f}-{bins[i+1]:6.1f}: '
                  f'{100*hist[i]/var_k:6.2f} %', flush=True)
    return k, w, var_k


def logo(tag, k, var_k, groups):
    # GLS-level leave-group-out: replace the full-data GLS term by
    # the complement's and invert the pars block (conditional on u)
    lam_pars = lam_full[:num_params, :num_params]
    gls_full = kmat.T @ solve_c0(kmat)
    print(f'[LOGO] ===== {tag}: sd ratio without group =====', flush=True)
    for gname, sel in groups.items():
        comp = ~sel
        kc = kmat[comp]
        cexp_cc = cexp[np.ix_(comp, comp)]
        s_c = smat[comp]
        c0_cc = cexp_cc + (s_c * dvals[None, :]) @ s_c.T
        gls_comp = kc.T @ np.linalg.solve(
            0.5 * (c0_cc + c0_cc.T), kc
        )
        lam_new = lam_pars - gls_full + gls_comp
        lam_new = 0.5 * (lam_new + lam_new.T)
        ekp = np.zeros(num_params); ekp[k] = 1.
        var_new = float(np.linalg.solve(lam_new, ekp)[k])
        var_base = float(np.linalg.solve(lam_pars, ekp)[k])
        print(f'[LOGO]   {np.sqrt(var_new/var_base):6.3f}  x sd without: '
              f'{gname}  ({int(np.sum(sel))} points)', flush=True)


reacs = exptable.REAC.to_numpy()
groups = {
    'abs Pu9 (MT:1-R1:9)': reacs == 'MT:1-R1:9',
    'shape Pu9 (MT:2-R1:9)': reacs == 'MT:2-R1:9',
    'ratio Pu9/U5 (MT:3+4-R1:9-R2:8)': (
        (reacs == 'MT:3-R1:9-R2:8') | (reacs == 'MT:4-R1:9-R2:8')
    ),
    'abs U5 (MT:1-R1:8)': reacs == 'MT:1-R1:8',
    'shape U5 (MT:2-R1:8)': reacs == 'MT:2-R1:8',
    'SACS (MT:6, MT:10)': (
        np.char.startswith(reacs.astype(str), 'MT:6')
        | np.char.startswith(reacs.astype(str), 'MT:10')
    ),
}

for tag, reac in (('Pu9(n,f)', 'MT:1-R1:9'), ('U5(n,f)', 'MT:1-R1:8')):
    k, w, var_k = decompose(tag, reac, 2.0)
    logo(tag, k, var_k, groups)
