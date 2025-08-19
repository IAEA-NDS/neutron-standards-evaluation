import sys
sys.path.append('../data')
from data_utils import  load_std2017_data
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table,
)
from gmapy.mappings.priortools import (
    attach_shape_prior,
    initialize_shape_prior,
    remove_dummy_datasets
)
import numpy as np
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
import tensorflow as tf
import matplotlib.pyplot as plt
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)


std2017_data, std2017_reacs = load_std2017_data()

db_path = '../data/data.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

priorvals = priortable.PRIOR.to_numpy()

# generate a restricted mapping blending out the fixed parameters
is_adj = priorcov.diagonal() != 0.
adj_idcs = np.where(is_adj)[0]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
)
propfun = tf.function(restrimap.propagate)
jacfun = tf.function(restrimap.jacobian)

red_priortable = priortable.loc[is_adj, :].reset_index(drop=True)

# predict the evaluated cross sections
chain, = load_objects('../output/0f2311f/output/03_mcmc_sampling_output.pkl', 'chain')
propfun(chain[0,:len(red_priortable)])

# predict the evaluated cross sections using optimization
optres, = load_objects(
    '../evaluation/output_ea40e40/02_parameter_optimization_output.pkl',
    'optres'
)
eval_xs_opt = optres.position
pred_xs_opt = propfun(eval_xs_opt[:len(red_priortable)])

# calculate evaluated values
eval_xs = np.mean(chain.numpy(), axis=0) 
eval_unc = np.std(chain.numpy(), axis=0)
red_priortable['EVAL'] = eval_xs[:len(red_priortable)]
red_priortable['EVAL_UNC'] = eval_unc[:len(red_priortable)]
red_priortable['EVAL_OPT'] = eval_xs_opt[:len(red_priortable)]

# calculate derived sacs values
pred_xs = propfun(eval_xs[:len(red_priortable)])
exptable['PRED'] = pred_xs
exptable['PRED_OPT'] = pred_xs_opt
exptable[exptable['REAC'].str.startswith('MT:6-')]

exptable[exptable['REAC'].str.startswith('MT:10-')]

# plot the PU9(n,f) cross section 


def plot_data(reac, xlim, ylim, with_exp=True, file=None):
    plot_title = {'MT:1-R1:8': 'U5(n,f)', 'MT:1-R1:9': 'PU9(n,f)', 'MT:1-R1:10': 'U8(n,f)'}[reac]
    dt = std2017_data.query(f'REAC == "{reac}"')
    evaldt = red_priortable.query(f'REAC == "{reac}"')
    expdt = exptable.query(f'REAC == "{reac}"')
    if with_exp:
        plt.errorbar(expdt['ENERGY'], expdt['DATA'], yerr=expdt['UNC'], fmt='o')
    plt.title(plot_title)
    plt.plot(dt['ENERGY'], dt['DATA'], color='blue', label='std2017')
    plt.plot(evaldt['ENERGY'], evaldt['EVAL'], color='red', label='current')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('energy [MeV]')
    plt.ylabel('cross section [b]')
    plt.legend(loc='lower right')
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()


plot_data('MT:1-R1:8', [0.15, 100], [1, 2.2], False, file='u5_n_f_150-100000.png')
plot_data('MT:1-R1:8', [0.15, 100], [1, 2.2], True, file='u5_n_f_150-100000_exp.png')
plot_data('MT:1-R1:9', [0.15, 100], [1, 2.5], False, file='pu9_n_f_150-100000.png')
plot_data('MT:1-R1:9', [0.15, 100], [1, 2.5], True, file='pu9_n_f_150-100000_exp.png')

plot_data('MT:1-R1:8', [0.15, 30], [1, 2.2], False, file='u5_n_f_150-30000.png')
plot_data('MT:1-R1:8', [0.15, 30], [1, 2.2], True, file='u5_n_f_150-30000_exp.png')

plot_data('MT:1-R1:9', [0.1, 7], [1.4, 2.12], False, file='pu9_n_f_100-7000.png')
plot_data('MT:1-R1:9', [0.1, 7], [1.4, 2.12], True,  file='pu9_n_f_100-7000_exp.png')
plot_data('MT:1-R1:9', [0.1, 30], [1, 2.6], False,   file='pu9_n_f_100-30000.png')
plot_data('MT:1-R1:9', [0.1, 30], [1, 2.6], True,    file='pu9_n_f_100-30000_exp.png')

plot_data('MT:1-R1:10', [0.1, 30], [0, 2], False, file='u8_n_f_100-30000.png')
plot_data('MT:1-R1:10', [0.1, 30], [0, 2], True, file='u8_n_f_100-30000_exp.png')
