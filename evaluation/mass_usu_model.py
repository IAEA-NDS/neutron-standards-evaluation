import re
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix, csr_matrix, diags


def get_mass_usu_sens(exp_reacs):
    row_idcs = []
    col_idcs = []
    vals = []
    for i, reac in enumerate(exp_reacs): 
        if re.match('MT:([136]|10)-', reac):
            if 'R1:8' in reac:
                row_idcs.append(i)
                col_idcs.append(0)
                vals.append(1.)
            elif 'R1:9' in reac:
                row_idcs.append(i)
                col_idcs.append(1)
                vals.append(1.)
            elif 'R1:10' in reac: 
                row_idcs.append(i)
                col_idcs.append(2)
                vals.append(1.)
        if reac.startswith('MT:3-') or reac.startswith('MT:10-'):
            if 'R2:8' in reac:
                row_idcs.append(i)
                col_idcs.append(0)
                vals.append(-1.)
            elif 'R2:9' in reac:
                row_idcs.append(i)
                col_idcs.append(1)
                vals.append(-1.)
            elif 'R2:10' in reac: 
                row_idcs.append(i)
                col_idcs.append(2)
                vals.append(-1.)
    return coo_matrix((vals, (row_idcs, col_idcs))).toarray()


def create_like_cov_fun(usu_df, expcov_linop, Smat):

    def map_uncertainties(u):
        ids = np.zeros((len(usu_df),), dtype=np.int32)
        for index, row in red_usu_df.iterrows():
            material = row.MATERIAL
            cur_idcs = usu_df.index[
                (usu_df.MATERIAL == material)
            ].to_numpy()
            ids[cur_idcs] = index
        # scatter the uncertainties to the appropriate places
        tf_ids = tf.constant(ids, dtype=tf.int32)
        uncs = tf.nn.embedding_lookup(u, tf_ids)
        return uncs

    def like_cov_fun(u):
        # ad-hoc hack to see if we observe the expected uncertainty inflation
        u = tf.constant([0.005, 0.005, 0.005], dtype=tf.float64)
        uncs = map_uncertainties(u)
        # covop = tf.linalg.LinearOperatorLowRankUpdate(
        covop = tf.linalg.LinearOperatorLowRankUpdate(
            expcov_linop, Smat, tf.square(uncs) + 1e-7,
            is_self_adjoint=True, is_positive_definite=True,
            is_diag_update_positive=True
        )
        return covop
    red_usu_df = usu_df[['MATERIAL']].drop_duplicates()
    red_usu_df.sort_values(
        ['MATERIAL'], ascending=True, ignore_index=True, inplace=True
    )
    return like_cov_fun, red_usu_df
