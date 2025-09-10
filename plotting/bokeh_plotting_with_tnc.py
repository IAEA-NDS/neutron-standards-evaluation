import re
import os
from bokeh.palettes import Category20_20
from bokeh.plotting import figure, show, save, curdoc
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Row,
    Column,
    Select,
    CustomJS,
    Dropdown,
    NumberFormatter,
    Div,
)
from bokeh.models.widgets import TableColumn, DataTable
from bokeh.layouts import column, row
from bokeh.models import TabPanel, Tabs
from bokeh.layouts import gridplot

import pandas as pd
from gmapy.data_management.object_utils import load_objects
import numpy as np
import matplotlib.pyplot as plt
from gmapy.mappings.tf.compound_map_tf import CompoundMap
from gmapy.mappings.tf.restricted_map import RestrictedMap
import tensorflow as tf
from gmapy.mappings.priortools import attach_shape_prior
from gmapy.data_management.quantity_types import SHAPE_MT_IDS

from data_preparation import (
    prepare_result_data,
    load_evaluation,
    get_human_readable_reaction_string,
    parse_reaction_string,
    load_endf_evaluation
)

# only used for renormalization
dfs = prepare_result_data('ea40e40')
exptable = dfs['exptable']

# reference cross section
dfs = prepare_result_data('ea40e40')
ref_priortable = dfs['priortable'].copy()
std2017 = dfs['std2017_dt']


pred_list = []
cols = []
pred_list.append(load_evaluation('ea40e40', 'endfb81 submission (MCMC)', 'brown', 'dotdash'))
cols.append("PRED")


endfb81_path = '/home/gschnabel/bigdata/nuclibs/endfb8.1/neutrons-version.VIII.1'
endfb81_pu9_file = os.path.join(endfb81_path, 'n-094_Pu_239.endf')
pu9_nf_dt = load_endf_evaluation(endfb81_pu9_file, 18, 9)
pred_list.append(
    {
        'git_hash': None,
        'pred_dt': pu9_nf_dt,
        'label': 'b81',
        'color': 'blue',
        'style': 'solid',
    }
)
cols.append("PRED")


endfb81_u5_file = os.path.join(endfb81_path, 'n-092_U_235.endf')
u5_nf_dt = load_endf_evaluation(endfb81_u5_file, 18, 8)
pred_list.append(
    {
        'git_hash': None,
        'pred_dt': u5_nf_dt,
        'label': 'b81',
        'color': 'blue',
        'style': 'solid',
    }
)
cols.append("PRED")

endfb81_u8_file = os.path.join(endfb81_path, 'n-092_U_238.endf')
u8_nf_dt = load_endf_evaluation(endfb81_u8_file, 18, 10)
pred_list.append(
    {
        'git_hash': None,
        'pred_dt': u8_nf_dt,
        'label': 'b81',
        'color': 'blue',
        'style': 'solid',
    }
)
cols.append("PRED")

# interpolate STD2017 to energies of experiments and predictions
dt_list = [v['pred_dt'] for v in pred_list]

# add the experimental data
dt_list.append(exptable)
cols.append('DATA')


for col, dt in zip(cols, dt_list):
    grouped = dt.groupby('REAC')
    for reac, curdt in grouped:
        is_mt1 = reac.startswith('MT:1-')
        is_mt3 = reac.startswith('MT:3-')
        if not is_mt1 and not is_mt3:
            continue
        if is_mt1:
            if not np.isin(reac, std2017.REAC):
                continue
            red_std2017 = std2017[std2017.REAC == reac]
            interp_vals = np.interp(curdt.ENERGY, red_std2017.ENERGY, red_std2017.DATA, left=np.nan, right=np.nan)
            dt.loc[curdt.index, 'STD2017'] = interp_vals
            dt.loc[curdt.index, 'RATIO'] = dt.loc[curdt.index, col] / dt.loc[curdt.index, 'STD2017']
        if is_mt3:
            reac_comps = reac.split('-')
            r1 = reac_comps[1].split(':')[1]
            r2 = reac_comps[2].split(':')[1]
            reac1 = f'MT:1-R1:{r1}'
            reac2 = f'MT:1-R1:{r2}'
            if not np.isin(reac1, std2017.REAC) or not np.isin(reac2, std2017.REAC):
                continue
            red_std2017_upper = std2017[std2017.REAC == reac1]
            red_std2017_lower = std2017[std2017.REAC == reac2]
            interp_vals_upper = np.interp(curdt.ENERGY, red_std2017_upper.ENERGY, red_std2017_upper.DATA, left=np.nan, right=np.nan)
            interp_vals_lower = np.interp(curdt.ENERGY, red_std2017_lower.ENERGY, red_std2017_lower.DATA, left=np.nan, right=np.nan)
            interp_vals = interp_vals_upper / interp_vals_lower
            dt.loc[curdt.index, 'STD2017'] = interp_vals
            dt.loc[curdt.index, 'RATIO'] = dt.loc[curdt.index, col] / dt.loc[curdt.index, 'STD2017']


cur_eval = pred_list[0]
assert cur_eval['label'] == 'endfb81 submission (MCMC)'
assert cur_eval['git_hash'] == 'ea40e40'

pred_dt = cur_eval['pred_dt']
pu9_eval = pred_dt.query('REAC == "MT:1-R1:9"')

plt.plot(pu9_eval['ENERGY'], pu9_eval['RATIO']-1, c='g')
plt.xlim(0.1, 16)
plt.ylim(-0.03, 0.03)
plt.xscale('log')
plt.title('PU9(n,f) cross section: NEW versus STD2017')
plt.xlabel('energy [MeV]')
plt.ylabel('(NEW - STD2017)/STD2017')
# plt.show()
plt.savefig('pu9_nf_endfb81_paper_plot.png')
