import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
import matplotlib.pyplot as plt


priortable,  exptable, like_cov_fun, compmap, restrimap, num_covpars, red_usu_df = \
    load_objects('../../output/084a569/evaluation/output/01_model_preparation_output.pkl', 
            'priortable', 'exptable', 'like_cov_fun',
            'compmap', 'restrimap', 'num_covpars', 'red_usu_df')
optres, = \
    load_objects('../../output/084a569/evaluation/output/02_parameter_optimization_output.pkl', 
            'optres')

chain, = \
    load_objects('../../output/084a569/evaluation/output/03_mcmc_sampling_output.pkl', 
            'chain')

all_postvals_mcmc = np.mean(np.abs(chain), axis=0)

plt.hist(np.abs(chain[:, -3]), bins=20)
plt.show()


rpriortable = priortable[priortable.UNC != 0].reset_index()

postvals = optres.position[:-num_covpars]
usuvals = optres.position[-num_covpars:]
usuvals_mcmc = all_postvals_mcmc[-num_covpars:]

red_usu_df['POST'] = np.abs(usuvals)
red_usu_df['POST_MCMC'] = np.abs(usuvals_mcmc)

expcov_usu = like_cov_fun(usuvals_mcmc).to_dense()
expcov = like_cov_fun(tf.constant([1e-4]*num_covpars, dtype=tf.float64)).to_dense()

predvals = restrimap.propagate(postvals)
expcov_usu_abs = expcov_usu * np.reshape(predvals, (-1,1)) * np.reshape(predvals, (1,-1))
expcov_abs = expcov * np.reshape(predvals, (-1,1)) * np.reshape(predvals, (1,-1))

npinv = np.linalg.inv
S = tf.sparse.to_dense(restrimap.jacobian(postvals)).numpy()
postcov_usu = npinv(S.T @ npinv(expcov_usu_abs.numpy()) @ S)
postcov = npinv(S.T @ npinv(expcov_abs.numpy()) @ S)
postuncs_abs = np.sqrt(np.diag(postcov))
postuncs_usu_abs = np.sqrt(np.diag(postcov_usu))


def plot_eval(priortable, exptable, postuncs_abs, expcov_abs):
    expids = pd.unique(exptable.NODE)
    expuncs_abs = np.sqrt(np.diag(expcov_abs))
    for expid in expids:
        curdt = exptable[exptable.NODE == expid]
        curuncs = expuncs_abs[curdt.index]
        plt.errorbar(curdt.ENERGY, curdt.DATA, curuncs)
        plt.plot(curdt.ENERGY, curdt.DATA)
    plt.errorbar(priortable.ENERGY, postvals, postuncs_abs)
    plt.plot(priortable.ENERGY, postvals)
    plt.show()



expcor_usu = expcov_usu / np.sqrt((np.reshape(np.diag(expcov_usu), (-1,1)) * np.reshape(np.diag(expcov_usu), (1,-1))))
expcor = expcov / np.sqrt((np.reshape(np.diag(expcov), (-1,1)) * np.reshape(np.diag(expcov), (1,-1))))


postcor = postcov / np.sqrt((np.reshape(np.diag(postcov), (-1,1)) * np.reshape(np.diag(postcov), (1,-1))))
postcor_usu = postcov_usu / np.sqrt((np.reshape(np.diag(postcov_usu), (-1,1)) * np.reshape(np.diag(postcov_usu), (1,-1))))

sel_pu9 = rpriortable[
    (rpriortable.REAC == 'MT:1-R1:9')
    & (rpriortable.ENERGY > 0.8)
    & (rpriortable.ENERGY < 6)
].index
curmat = postcor_usu[np.ix_(sel_pu9, sel_pu9)]

fig, ax = plt.subplots()
cax = plt.imshow(curmat, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax, label='Correlation Coefficient')
ax.set_xticks(np.arange(len(rpriortable.loc[sel_pu9, 'ENERGY'])))
ax.set_yticks(np.arange(len(rpriortable.loc[sel_pu9, 'ENERGY'])))
ax.set_xticklabels(rpriortable.loc[sel_pu9, 'ENERGY'])
ax.set_yticklabels(rpriortable.loc[sel_pu9, 'ENERGY'])
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax.set_title('labeled cormat')
plt.tight_layout()
plt.show()
