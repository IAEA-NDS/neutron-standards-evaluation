import numpy as np
from scipy.sparse import block_diag, identity, vstack, issparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def get_exptable_by_blocks(exptable, block_idcs, block_starts, block_stops):
    sta = block_starts
    sto = block_stops
    idcs = sum((list(range(sta[i], sto[i])) for i in block_idcs), start=[])
    return exptable.loc[idcs]


def get_expcov_by_blocks(expcov, block_idcs, block_starts, block_stops):
    sta = block_starts
    sto = block_stops
    idcs = sum((list(range(sta[i], sto[i])) for i in block_idcs), start=[])
    return expcov[np.ix_(idcs, idcs)]


def get_S_by_blocks(S, block_idcs, block_starts, block_stops):
    Slist = []
    for idx in block_idcs:
        sta = block_starts[idx]
        sto = block_stops[idx]
        curS = _to_dense_array(S[sta:sto,:])
        Slist.append(curS)
    return np.vstack(Slist)


def get_idcs_by_blocks(block_idcs, block_starts, block_stops):
    sta = block_starts
    sto = block_stops
    idcs = sum((list(range(sta[i], sto[i])) for i in block_idcs), start=[])
    return idcs


def _to_dense_array(x):
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


def calc_postmean(
    predvals, expvals, S_blocks, inv_covmat_blocks, block_idcs=None,
    invert=False, idx=None, scale=None
): 
    """Get posterior mean values."""
    predvals = np.array(predvals).reshape(-1, 1)
    expvals = np.array(expvals).reshape(-1, 1)
    block_sizes = [c.shape[0] for c in inv_covmat_blocks]
    block_stops = np.cumsum(block_sizes)
    block_starts = np.concatenate([[0], block_stops[:-1]])
    if not invert:
        block_mask = np.ones(len(inv_covmat_blocks), dtype=bool)
        point_mask = np.ones(block_stops[-1]-block_starts[0], dtype=bool) 
    else:
        block_mask = np.zeros(len(inv_covmat_blocks), dtype=bool)
        point_mask = np.zeros(block_stops[-1]-block_starts[0], dtype=bool) 
    if block_idcs:
        for i in block_idcs:
            block_mask[i] = invert
            point_mask[block_starts[i]:block_stops[i]] = invert
    # select corresponding values 
    expvals = expvals[point_mask,:]
    predvals = predvals[point_mask,:]

    blocks = [inv_covmat_blocks[i] for i, p in enumerate(block_mask) if p]
    Sb = [S_blocks[i] for i, p in enumerate(block_mask) if p] 
    inv_covmat = block_diag(blocks, format='csc')
    S = vstack(Sb, format='csc')
    regmat = identity(S.shape[1]) * 1e-10
    postcov = np.linalg.inv(_to_dense_array(S.T @ inv_covmat @ S + regmat))
    postvals = postcov @ S.T @ inv_covmat @ (expvals - predvals)
    if scale is not None:
        postvals = postvals / np.array(scale).reshape(-1, 1)  
    if idx is not None:
        postvals = postvals[idx]
    return postvals


def add_normunc(covmat, idcs, unc): 
    covmat = covmat.copy()
    covmat[np.ix_(idcs, idcs)] += unc*unc
    return covmat


def calc_postcov2(
    S, expcov_rel, pred_exp, pred_prior,
    block_starts, block_stops, reg=1e-8,
    idcs=None, exploit_structure=False
):
    assert S.shape[0] == pred_exp.size
    assert S.shape[1] == pred_prior.size
    assert expcov_rel.shape[0] == expcov_rel.shape[1]
    assert expcov_rel.shape[0] == S.shape[0]
    if idcs is None:
        idcs = np.arange(expcov_rel.shape[0])
    # filter input according to idcs 
    exp_mask = np.zeros(pred_exp.size, dtype=bool) 
    exp_mask[idcs] = True 
    pred_exp = np.array(pred_exp)
    pred_prior = np.array(pred_prior)
    S = _to_dense_array(S)
    expcov_rel = _to_dense_array(expcov_rel)
    # do the inference 
    expcov = expcov_rel * pred_exp.reshape(-1, 1) * pred_exp.reshape(1,-1)
    regmat = reg * np.identity(S.shape[1], dtype=float)
    if exploit_structure:
        A = 0.0
        for bsta, bsto in zip(block_starts, block_stops):
            cur_exp_mask = exp_mask[bsta:bsto]
            curcov = expcov[bsta:bsto, bsta:bsto][np.ix_(cur_exp_mask, cur_exp_mask)]
            curinv = np.linalg.inv(curcov)
            curS = S[bsta:bsto,:][cur_exp_mask,:]
            A += curS.T @ curinv @ curS 
    else:
        curS = S[exp_mask, :]
        curcov = expcov[np.ix_(exp_mask, exp_mask)]
        curinv = np.linalg.inv(curcov)
        A = curS.T @ curinv @ curS
    postcov = np.linalg.inv(A + regmat)
    postcov /= pred_prior.reshape(-1,1) * pred_prior.reshape(1,-1) 
    return postcov


def calc_postcov(
    S_blocks, inv_covmat_blocks, block_idcs=None, invert=False, idx=None, scale=None
): 
    """Get posterior covariance matrix."""
    scale = np.array(scale)
    if not invert:
        block_mask = np.ones(len(inv_covmat_blocks), dtype=bool)
    else:
        block_mask = np.zeros(len(inv_covmat_blocks), dtype=bool)
    if block_idcs:
        for i in block_idcs:
            block_mask[i] = invert
    blocks = [inv_covmat_blocks[i] for i, p in enumerate(block_mask) if p]
    Sb = [S_blocks[i] for i, p in enumerate(block_mask) if p] 
    inv_covmat = block_diag(blocks, format='csc')
    S = vstack(Sb, format='csc')
    regmat = identity(S.shape[1]) * 1e-10
    postcov = np.linalg.inv(_to_dense_array(S.T @ inv_covmat @ S + regmat))
    if scale is not None:
        postcov = postcov / scale.reshape(-1, 1) / scale.reshape(1, -1)
    if idx is not None:
        postcov = postcov[np.ix_(idx, idx)]
    return postcov


def calc_target_unc(
    projvec, S_blocks, inv_covmat_blocks, block_idcs=None, invert=False
):
    """Evaluate using all datablocks except the ones in block_idcs."""
    k = projvec.reshape(-1, 1)
    if not invert:
        block_mask = np.ones(len(inv_covmat_blocks), dtype=bool)
    else:
        block_mask = np.zeros(len(inv_covmat_blocks), dtype=bool)
    if block_idcs:
        for i in block_idcs:
            block_mask[i] = invert
    blocks = [inv_covmat_blocks[i] for i, p in enumerate(block_mask) if p]
    Sb = [S_blocks[i] for i, p in enumerate(block_mask) if p] 
    inv_covmat = block_diag(blocks, format='csc')
    S = vstack(Sb, format='csc')
    regmat = identity(projvec.size) * 1e-10
    res = np.sqrt(k.T @ spsolve(S.T @ inv_covmat @ S + regmat, k))
    return res.item() 


def calc_rel_target_unc(
    projvec, S_blocks, inv_covmat_blocks, pred, block_idcs=None, invert=False
):
    return calc_target_unc(
        projvec, S_blocks, inv_covmat_blocks, block_idcs, invert
    ) / pred


def get_normalization_unc(covmat):
    """Extract normalization uncertainty component from a covariance matrix."""
    numpts = covmat.shape[0] 
    uk = np.ones(numpts, dtype=float)
    k = uk / np.sqrt(np.sum(uk*uk))
    normunc = np.sqrt(k.T @ covmat @ k) / np.sqrt(numpts)
    return normunc


def get_constraint_reduction(covmat, constr):
    covmat = _to_dense_array(covmat)
    constr = constr.reshape(-1, 1)
    Ainv = np.linalg.inv(covmat)
    num = Ainv @ constr @ constr.T @ Ainv
    den = 1 + constr.T @ Ainv @ constr
    return Ainv - num / den


def cov2cor(covmat):
    uncs = np.sqrt(np.diag(covmat))
    return covmat / uncs.reshape(-1, 1) / uncs.reshape(1, -1) 


def plot_cormat(x, y, cormat):
    fig, ax = plt.subplots()
    cax = plt.imshow(cormat, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label='Correlation Coefficient')
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title('labeled cormat')
    plt.tight_layout()
    plt.show()

