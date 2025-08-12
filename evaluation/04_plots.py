import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
import matplotlib.pyplot as plt


priortable, exptable, like_cov_fun, compmap = \
    load_objects('output/01_model_preparation_output.pkl', 
            'priortable', 'exptable', 'like_cov_fun', 'compmap')
optres, = \
    load_objects('output/02_parameter_optimization_output.pkl', 
            'optres')

postvals = optres.position[:-2]
usuvals = optres.position[-2:]

expcov_usu = like_cov_fun(usuvals).to_dense()
expcov = like_cov_fun(tf.constant([1e-4, 1e-4], dtype=tf.float64)).to_dense()

predvals = compmap.propagate(postvals)
expcov_usu_abs = expcov_usu * np.reshape(predvals, (-1,1)) * np.reshape(predvals, (-1,1))
expcov_abs = expcov * np.reshape(predvals, (-1,1)) * np.reshape(predvals, (-1,1))

npinv = np.linalg.inv
S = tf.sparse.to_dense(compmap.jacobian(postvals)).numpy()
postcov_usu = npinv(S.T @ npinv(expcov_usu_abs.numpy()) @ S)
postcov = npinv(S.T @ npinv(expcov_abs.numpy()) @ S)
postuncs_abs = np.sqrt(np.diag(postcov))
postuncs_usu_abs = np.sqrt(np.diag(postcov_usu))


def plot_eval(priortable, exptable, postuncs_abs, expcov_abs):
    expids = pd.unique(exptable.REAL_NODE)
    expuncs_abs = np.sqrt(np.diag(expcov_abs))
    for expid in expids:
        curdt = exptable[exptable.REAL_NODE == expid]
        curuncs = expuncs_abs[curdt.index]
        plt.errorbar(curdt.ENERGY, curdt.DATA, curuncs)
        plt.plot(curdt.ENERGY, curdt.DATA)
    plt.errorbar(priortable.ENERGY, postvals, postuncs_abs)
    plt.plot(priortable.ENERGY, postvals)
    plt.show()


plot_eval(priortable, exptable, postuncs_usu_abs, expcov_usu)

expcor_usu = expcov_usu / np.sqrt((np.reshape(np.diag(expcov_usu), (-1,1)) * np.reshape(np.diag(expcov_usu), (1,-1))))
expcor = expcov / np.sqrt((np.reshape(np.diag(expcov), (-1,1)) * np.reshape(np.diag(expcov), (1,-1))))


postcor = postcov / np.sqrt((np.reshape(np.diag(postcov), (-1,1)) * np.reshape(np.diag(postcov), (1,-1))))
postcor_usu = postcov_usu / np.sqrt((np.reshape(np.diag(postcov_usu), (-1,1)) * np.reshape(np.diag(postcov_usu), (1,-1))))

plt.imshow(postcor_usu, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
