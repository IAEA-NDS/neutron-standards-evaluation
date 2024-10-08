import numpy as np
import pandas as pd


def translate_to_absreacs(reacs):
    absreacs = []
    for reac in reacs:
        if reac.startswith('MT:2-'):
            absreac = 'MT:1-' + reac[5:]
        elif reac.startswith('MT:4-'):
            absreac = 'MT:3-' + reac[5:]
        elif reac.startswith('MT:8-'):
            absreac = 'MT:5-' + reac[5:]
        elif reac.startswith('MT:9-'):
            absreac = 'MT:7-' + reac[5:]
        else:
            absreac = reac
        absreacs.append(absreac)
    return np.array(absreacs)


def renormalize_data(priortable, exptable, priorvals, expvals):
    renorm_vals = np.array(expvals)
    expids = exptable.loc[exptable.REAC.str.match('MT:[2489]-'), 'NODE'].unique()
    for expid in expids:
        norm_node = 'norm_' + expid[4:]
        norm_index = priortable.index[priortable.NODE == norm_node]
        if len(norm_index) != 1:
            raise IndexError(
                f'exactly one normalization error must be present for {expid}'
            )
        norm_index = norm_index[0]
        norm_fact = priorvals[norm_index]
        exp_idcs = exptable.index[exptable.NODE == expid]
        renorm_vals[exp_idcs] /= norm_fact
    return renorm_vals
