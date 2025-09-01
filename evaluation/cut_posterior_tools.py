import numpy as np


def cut_inference(refvals, S, predvals, expvals, expcov, cut_idcs=None, cut_unc=1e-5):
    cut_idcs = [] if cut_idcs is None else cut_idcs
    refvals = refvals.reshape(-1, 1).copy()
    predvals = predvals.reshape(-1, 1).copy()
    expvals = expvals.reshape(-1, 1).copy()
    num_cuts = np.sum(cut_idcs) if isinstance(cut_idcs, bool) else len(cut_idcs)
    cutcov = expcov.copy()
    cut_uncs = np.abs(expvals.flatten()[cut_idcs]) * cut_unc
    cut_vars = np.square(cut_uncs)
    cutcov[np.ix_(cut_idcs, cut_idcs)] = np.diag(cut_vars)
    cutcov_inv = np.linalg.inv(cutcov)
    postcov_cond = np.linalg.inv(S.T @ cutcov_inv @ S)
    postcov_cond2 = np.linalg.inv(S.T @ np.linalg.solve(cutcov, S))
    postvals_cond = refvals.copy()  # sneaky reference Python!
    postvals_cond += postcov_cond @ (S.T @ (cutcov_inv @ (expvals - predvals)))
    # now update the covariance matrix
    X = postcov_cond @ S[cut_idcs,:].T @ cutcov_inv[np.ix_(cut_idcs, cut_idcs)]  # @ sacs_diff
    postcov_cut = X @ expcov[np.ix_(cut_idcs, cut_idcs)] @ X.T
    return postvals_cond.flatten(), postcov_cut + postcov_cond
