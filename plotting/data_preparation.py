import sys
sys.path.append('../data')

import joblib  # for caching

import pandas as pd
from gmapy.data_management.object_utils import load_objects
import numpy as np
import matplotlib.pyplot as plt
from gmapy.mappings.tf.compound_map_tf import CompoundMap
from gmapy.mappings.tf.restricted_map import RestrictedMap
import tensorflow as tf
from data_utils import load_std2017_data
from utils import renormalize_data, translate_to_absreacs
from gmapy.mappings.priortools import attach_shape_prior
from gmapy.data_management.quantity_types import SHAPE_MT_IDS


mem = joblib.Memory('/tmp')


def load_evaluation(git_hash, label, color, style):
    dfs = prepare_result_data(git_hash)
    return {
        'git_hash': git_hash,
        'pred_dt': dfs['pred_dt'], 
        'label': label ,
        'color': color,
        'style': style,
    }


@mem.cache
def prepare_result_data(git_hash):
    curcalc = f'../output/{git_hash}/output'
    priortable, red_usu_df, is_adj, exptable, restrimap, num_covpars, like_cov_fun = \
        load_objects(f'{curcalc}/01_model_preparation_output.pkl',
                     'priortable', 'red_usu_df', 'is_adj',
                     'exptable', 'restrimap', 'num_covpars', 'like_cov_fun')
    chain, = load_objects(f'{curcalc}/03_mcmc_sampling_output.pkl', 'chain')
    optres, = load_objects(f'{curcalc}/02_parameter_optimization_output.pkl', 'optres')
    eval_maxlike_raw = optres.position.numpy()

    red_priortable = priortable.loc[is_adj, :].reset_index(drop=True)

    # augment priortable with results
    red_priortable['POST'] = np.mean(chain[:, :len(red_priortable)], axis=0)
    red_priortable['POSTUNC'] = np.std(chain[:, :len(red_priortable)], axis=0)

    # add column where uncertainties are inflated by USU components
    # as they are used/determined in the evaluation
    inflated_cov = like_cov_fun(np.mean(np.abs(chain[:, -num_covpars:]), axis=0))
    inflated_uncs = np.sqrt(inflated_cov.diag_part())
    exptable['UNC_USU'] = inflated_uncs * exptable['DATA']

    red_priortable['MAXLIKE'] = eval_maxlike_raw[:len(red_priortable)]
    postvals = red_priortable['POST'].to_numpy(copy=True)
    expvals = exptable['DATA'].to_numpy(copy=True)

    std2017_dt, descr_list = load_std2017_data()

    # create the mapping object
    compmap = CompoundMap((priortable, std2017_dt), reduce=True)
    restrmap = RestrictedMap(
        len(is_adj), compmap.propagate, compmap.jacobian,
        fixed_params=priortable.loc[~is_adj, 'PRIOR'].to_numpy(copy=True),
        fixed_params_idcs=np.where(~is_adj)[0]
    )
    restrmap_prop = tf.function(restrmap.propagate)


    # propagate the mcmc estimates
    prop_chain = np.zeros((chain.shape[0], len(std2017_dt)), dtype=np.float64)
    for idx in range(chain.shape[0]):
        curchain = chain[idx, :len(red_priortable)]
        prop_chain[idx, :] = restrmap_prop(curchain)

    # construct one datatable with std2017 data and newest
    eval_mcmc = np.mean(prop_chain, axis=0)
    eval_mcmc_unc = np.std(prop_chain, axis=0)
    std2017_dt['POST'] = eval_mcmc
    std2017_dt['POSTUNC'] = eval_mcmc_unc
    eval_maxlike = restrmap_prop(optres.position[:len(red_priortable)])
    std2017_dt['MAXLIKE'] = eval_maxlike

    # =================================================
    # PREPARE FOR EXTENDED COMPARISON
    # =================================================

    exptable2 = exptable.copy()
    exptable2['ORIG_REAC'] = exptable['REAC']
    exptable2['ORIG_DATA'] = exptable['DATA']

    # renormalize shape data to match the fit
    renorm_vals2 = renormalize_data(red_priortable, exptable, postvals, expvals)
    exptable2['DATA'] = renorm_vals2
    translated_reacs = translate_to_absreacs(exptable2.REAC.to_numpy())
    exptable2['REAC'] = translated_reacs

    base_energies = np.sort(std2017_dt.ENERGY.unique())
    grouped = exptable2.groupby('REAC')
    exp_index = 1000
    pred_dt_table = []
    for group, curdt in grouped:
        Emin = np.min(curdt['ENERGY'])
        Emax = np.max(curdt['ENERGY'])
        tmp = base_energies
        energies = tmp[(tmp >= Emin) & (tmp <= Emax)]
        tmpdt = pd.DataFrame({
            'NODE': 'exp_' + str(exp_index),
            'REAC': group,
            'ENERGY': energies,
        })
        exp_index += 1
        pred_dt_table.append(tmpdt)

    pred_dt = pd.concat(pred_dt_table, ignore_index=True)
    pred_dt = pred_dt.sort_values(['REAC', 'ENERGY']).reset_index(drop=True)

    # create the mapping object
    priortable2 = priortable.copy()
    priortable2 = priortable2[~priortable2.NODE.str.match('norm_')]
    is_adj2 = priortable2.NODE != 'fis'
    red_priortable2 = priortable2.loc[is_adj2, :].reset_index(drop=True)

    compmap2 = CompoundMap((priortable, pred_dt), reduce=True)
    restrmap2 = RestrictedMap(
        len(is_adj), compmap2.propagate, compmap2.jacobian,
        fixed_params=priortable.loc[~is_adj, 'PRIOR'].to_numpy(copy=True),
        fixed_params_idcs=np.where(~is_adj)[0]
    )
    restrmap_prop2 = tf.function(restrmap2.propagate)
    eval_maxlike2 = restrmap_prop2(optres.position[:len(red_priortable)])

    # propagate the mcmc estimates
    prop_chain2 = np.zeros((chain.shape[0], len(pred_dt)), dtype=np.float64)
    for idx in range(chain.shape[0]):
        curchain = chain[idx, :len(red_priortable)]
        prop_chain2[idx, :] = restrmap_prop2(curchain)

    # construct a datatable with all propagated values
    eval_mcmc2 = np.mean(prop_chain2, axis=0)
    eval_mcmc_unc2 = np.std(prop_chain2, axis=0)
    pred_dt['PRED'] = eval_mcmc2
    pred_dt['PREDUNC'] = eval_mcmc_unc2
    pred_dt['MAXLIKE'] = eval_maxlike2

    return {
        'pred_dt': pred_dt,
        'exptable': exptable2,
        'std2017_dt': std2017_dt,
    }
