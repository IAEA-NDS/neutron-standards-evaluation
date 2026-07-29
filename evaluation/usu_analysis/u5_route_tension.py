"""Two-route consistency test for U5(n,f): direct measurements
(datasets whose only reaction is 8) vs the U5 curve implied by all
other data through ratios/sums, linearized around the v3b mode."""
import sys
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_factor, cho_solve
from gmapy.data_management.object_utils import load_objects

outdir = sys.argv[1] if len(sys.argv) > 1 else 'output_dsdisc_v3b_uli0'

post, likelihood, priortable, exptable, is_adj, smat_np, col2par, \
    n_shared = load_objects(
        f'{outdir}/01_model_preparation_output.pkl',
        'post', 'likelihood', 'priortable', 'exptable', 'is_adj',
        'smat_np', 'col2par', 'n_shared')
optres, = load_objects(
    f'{outdir}/02_parameter_optimization_output.pkl', 'optres')

x_mode = np.array(optres.position)
xt = tf.constant(x_mode, dtype=tf.float64)
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

# prior + barrier precision (physics block) and gradient at the mode
comps = list(post._distributions)
nonlik = [comps[0], comps[2]]  # prior, barrier (band priors act on covpars)
lam_pb = -sum(np.array(c.log_prob_hessian(xt)) for c in nonlik)
lam_pb = 0.5 * (lam_pb + lam_pb.T)
g_pb = -sum(np.array(c.log_prob_and_gradient(xt)[1]) for c in nonlik)
lam_pb = lam_pb[:num_params, :num_params]
g_pb = g_pb[:num_params]

# experimental covariance blocks; per-dataset USU bands (arbitration)
# can be added within blocks, shared floors are deliberately excluded
base_op = likelihood._covmodel._base_operator
op = base_op
while not isinstance(op, tf.linalg.LinearOperatorBlockDiag):
    op = op.operators[0]
sizes = [int(o.domain_dimension) for o in op.operators]
edges = np.concatenate([[0], np.cumsum(sizes)])
cexp = np.array(base_op.to_dense())
dvals = np.array(likelihood._covmodel._diag_fun(
    tf.constant(covpars_mode, dtype=tf.float64)))
ds_cols = np.where(col2par >= n_shared)[0]
cband = (smat_np[:, ds_cols] * dvals[ds_cols][None, :]) \
    @ smat_np[:, ds_cols].T

# direct-U5 rows: reaction string contains R1:8 and no other reaction
reacs = exptable.REAC.to_numpy().astype(str)
direct = np.isin(reacs, ('MT:1-R1:8', 'MT:2-R1:8', 'MT:6-R1:8'))
print(f'direct-U5 rows: {direct.sum()}   other rows: {(~direct).sum()}',
      flush=True)


def route_solutions(with_bands):
    cfull = cexp + cband if with_bands else cexp
    lam_d = lam_pb.copy(); rhs_d = -g_pb.copy()
    lam_i = lam_pb.copy(); rhs_i = -g_pb.copy()
    for b in range(len(sizes)):
        sl = slice(edges[b], edges[b+1])
        msk = direct[sl]
        cb = cfull[sl, sl]
        kb = kmat[sl]
        zb = z[sl]
        for sel, lam, rhs in (((msk), lam_d, rhs_d),
                              ((~msk), lam_i, rhs_i)):
            if not np.any(sel):
                continue
            cs = cb[np.ix_(sel, sel)]
            cf = cho_factor(0.5 * (cs + cs.T))
            w = cho_solve(cf, kb[sel])
            lam += kb[sel].T @ w
            rhs += w.T @ zb[sel]
    return (lam_d, rhs_d), (lam_i, rhs_i)


def posterior_at_nodes(lam, rhs, node_idcs):
    ridge = 1e-10 * np.max(np.diag(lam))
    lam = lam + ridge * np.eye(len(lam))
    cf = cho_factor(lam)
    dx = cho_solve(cf, rhs)
    cols = cho_solve(cf, np.eye(len(lam))[:, node_idcs])
    var = np.array([cols[k, i] for i, k in enumerate(node_idcs)])
    return dx[node_idcs], var


# U5 cross-section nodes on the evaluation grid
mask = (priortable.REAC.str.fullmatch('MT:1-R1:8', na=False)
        & priortable.NODE.str.match('xsid_', na=False))
u5_glob = np.array(priortable.index[mask])
u5_en = priortable.ENERGY.to_numpy()[u5_glob]
adj_idcs = np.where(np.array(is_adj))[0]
pos = {g: i for i, g in enumerate(adj_idcs)}
keep = [(g, e) for g, e in zip(u5_glob, u5_en) if g in pos]
show_en = (0.1, 0.5, 1.0, 2.0, 2.5, 5.0, 14.0, 20.0)

for with_bands in (False, True):
    tag = 'C_exp + per-dataset bands' if with_bands else 'C_exp only'
    (lam_d, rhs_d), (lam_i, rhs_i) = route_solutions(with_bands)
    node_idcs = [pos[g] for g, e in keep]
    md, vd = posterior_at_nodes(lam_d, rhs_d, node_idcs)
    mi, vi = posterior_at_nodes(lam_i, rhs_i, node_idcs)
    print(f'\n[U5T] ===== {tag} =====', flush=True)
    print(f'[U5T] {"E(MeV)":>8} {"direct%":>8} {"sd_d%":>6} '
          f'{"indirect%":>9} {"sd_i%":>6} {"tension":>8}', flush=True)
    tmax, emax = 0., 0.
    for i, (g, e) in enumerate(keep):
        x0 = pars_mode[pos[g]]
        srel_d = 100 * np.sqrt(vd[i]) / abs(x0)
        srel_i = 100 * np.sqrt(vi[i]) / abs(x0)
        dd = 100 * md[i] / abs(x0)
        di = 100 * mi[i] / abs(x0)
        t = (md[i] - mi[i]) / np.sqrt(vd[i] + vi[i])
        if abs(t) > abs(tmax) and 0.05 <= e <= 30.:
            tmax, emax = t, e
        if any(abs(e - s) < 1e-6 for s in show_en):
            print(f'[U5T] {e:>8g} {dd:>8.2f} {srel_d:>6.2f} '
                  f'{di:>9.2f} {srel_i:>6.2f} {t:>8.2f}', flush=True)
    print(f'[U5T] max |tension| in 0.05-30 MeV: {tmax:.2f} at '
          f'{emax:g} MeV', flush=True)
