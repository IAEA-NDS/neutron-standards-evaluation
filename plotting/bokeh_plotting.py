import re
import os
from bokeh.palettes import Category20_20
from bokeh.plotting import figure, show, save, curdoc
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Row,
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
dfs = prepare_result_data('08923b6')
exptable = dfs['exptable']

# reference cross section
dfs = prepare_result_data('ea40e40')
ref_priortable = dfs['priortable'].copy()
std2017 = dfs['std2017_dt']


pred_list = []
cols = []
# pred_list.append(load_evaluation('01a02a0', '8007 removed', 'green', 'dotdash'))
# pred_list.append(load_evaluation('f42e55d', '1013 to shape', 'blue', 'dotdash'))
# pred_list.append(load_evaluation('1e8ce5e', 'recommend_new MCMC', 'orange', 'dashed'))
# cols.append("PRED")
pred_list.append(load_evaluation('1e8ce5e', 'latest evaluation', 'green', 'solid'))
cols.append("MAXLIKE")
# pred_list.append(load_evaluation('ea40e40', 'liso_rel_low_unc', 'brown', 'dotdash'))
# cols.append("PRED")
# pred_list.append(load_evaluation('003a588', 'liso_abs_low_unc', 'blue', 'dotdash'))
# cols.append("PRED")
# pred_list.append(load_evaluation('55c975c', 'liso_abs', 'black', 'solid'))
# cols.append("PRED")
# pred_list.append(load_evaluation('fc8634c', 'no TPC>7 MeV', 'cyan', 'dashed'))
# cols.append("PRED")
# pred_list.append(load_evaluation('04cc6d2', 'drop Sherbakov exp (1012) OPT', 'red', 'solid'))
# cols.append("MAXLIKE")


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


##################################################
#            PLOTTING
##################################################


# helper function to plot experimental data in current figure


def get_expdata_for_reaction(reac, expdata, datacol=None):
    expdata = expdata[expdata.REAC == reac].copy()
    expdata = expdata[expdata.ENERGY > 2.58e-8].copy()
    is_okay = (~expdata[datacol].isna()) & (expdata[datacol] > 0.5) & (expdata[datacol] < 1.5)
    expdata = expdata[is_okay].copy()
    return expdata


def plot_expdata(figure, reac, expdata, datacol=None, include_usu=False):
    expdata = get_expdata_for_reaction(reac, expdata, datacol)
    if len(expdata) == 0:
        return

    grouped = expdata.groupby('NODE')
    numgroups = len(grouped)
    colpal = Category20_20[:numgroups]
    while len(colpal) < numgroups:
        colpal = colpal + colpal
    coldict = {k: colpal[i] for i, k in enumerate(grouped.groups.keys())}
    for node, curdt in grouped:
        curdt = curdt.copy()
        curlabel = node
        if 'ORIG_REAC' in curdt.columns:
            orig_reac = curdt['ORIG_REAC'].iloc[0]
            m = re.match(r'MT:(\d+)-', orig_reac)
            is_shape = int(m.group(1)) in SHAPE_MT_IDS
            qstr = 's' if is_shape else 'a'
            curlabel += qstr
        curdt['label'] = curlabel
        curdt['color'] = coldict[node]
        cursource = ColumnDataSource(data=curdt)
        figure.scatter('ENERGY', datacol, size=10, source=cursource, color='color',
                       legend_label=curlabel, level='underlay')
        err_xs = []
        err_ys = []
        uncvals = curdt['UNC_USU'] if include_usu else curdt['UNC']
        uncvals /= curdt['STD2017']  # uncertainties relative to std2017
        for x, y, yerr in zip(curdt['ENERGY'], curdt[datacol], uncvals):
            err_xs.append((x, x))
            err_ys.append((y-yerr, y+yerr))
        figure.multi_line(err_xs, err_ys, color=coldict[node])
        # hover = HoverTool(tooltips=[('Label', '@label')])
        # figure.add_tools(hover)
        # plt.errorbar(curdt.ENERGY, curdt.RENORM_DATA,
        #              yerr=curdt.UNC, fmt='o', label=curlabel)


# helper function to plot evaluations

def plot_evaluation(figure, reac, pred_dt, datacol, Emin, Emax, label, color, style):
    cdt = pred_dt.query(f'REAC == "{curreac}" & ENERGY >= {Emin} & ENERGY <= {Emax}')
    cdt = cdt.copy()
    cdt = cdt[(cdt['RATIO'] > 0.5) & (cdt['RATIO'] < 1.5)].copy()
    if len(cdt) == 0:
        return

    cursource = ColumnDataSource(data=cdt)
    figure.line(
        'ENERGY', datacol, source=cursource,
        color=color, line_dash=style, legend_label=label, line_width=4)

# plot comparing absolute cross sections

figures = {}
allreacs = {}
is_all_empty = True
for curreac in pred_list[0]['pred_dt'].REAC.unique():
    mtnum, *rnums = parse_reaction_string(curreac)
    m = re.findall(r'R\d+:(\d+)', curreac)
    if any(int(x) > 10 for x in m):
        continue
    if curreac.startswith('MT:6') or curreac.startswith('MT:10'):
        continue
    subfigures = []
    # first with RENORM_ML data
    curtitle = get_human_readable_reaction_string(curreac, ref_priortable)
    curfigure = figure(title=curtitle, width=1500, height=800, toolbar_location='above', name=curreac, x_axis_type='log')
    curexptable = get_expdata_for_reaction(curreac, exptable, datacol='RATIO')
    if len(curexptable) > 0:
        Emin = curexptable.ENERGY.min()
        Emax = curexptable.ENERGY.max()
        for pred in pred_list:
            plot_evaluation(
                curfigure, curreac, pred['pred_dt'], 'RATIO',
                Emin*0.9, Emax*1.1, label=pred['label'],
                color=pred['color'], style=pred['style'],
            )
        plot_expdata(curfigure, curreac, curexptable, datacol='RATIO', include_usu=False)
        subfigures.append(curfigure)
        curfigure.title.text_font_size = '20pt'
        curfigure.xaxis.axis_label = 'energy [MeV]'
        curfigure.xaxis.axis_label_text_font_size = '20pt'
        curfigure.yaxis.axis_label_text_font_size = '20pt'
        curfigure.xaxis.major_label_text_font_size = '20pt'
        curfigure.yaxis.major_label_text_font_size = '20pt'
        curfigure.legend.label_text_font_size = '15pt'
        if mtnum in (1, 5):
            curfigure.yaxis.axis_label = 'xs relative to std2017'
        elif mtnum in (3,):
            curfigure.yaxis.axis_label = 'ratio relative to std2017'
        # save everything
        figures[curreac] = subfigures


# create a panel for the sacs data


sacs_tables = []
for cur_pred_info in pred_list:
    if not 'pred_sacs_dt' in cur_pred_info:
        continue
    cur_sacs_dt = cur_pred_info['pred_sacs_dt'][["REAC", "OPT", "MCMC"]].copy()
    cur_sacs_dt['REAC'] = cur_sacs_dt['REAC'].apply(lambda x: get_human_readable_reaction_string(x, ref_priortable)) 

    cursource = ColumnDataSource(cur_sacs_dt)
    curcolumns = [
        TableColumn(field="REAC", title="Reaction"),
        TableColumn(field="OPT", title="Optim", formatter=NumberFormatter(format='0.0000')),
        TableColumn(field="MCMC", title="MCMC", formatter=NumberFormatter(format='0.0000')),
    ]
    sacs_datatable = DataTable(source=cursource, columns=curcolumns, width=700, height=200)
    sacs_datatable_title = Div(text=f"<h2>{cur_pred_info['label']} (git: {cur_pred_info['git_hash']})</h2>")

    sacs_tables.append(sacs_datatable_title)
    sacs_tables.append(sacs_datatable)

# auxiliary function

def get_groupname(key):
    s1 = key.split('-')
    s2 = s1[0].split(':')
    mtnum = int(s2[1])
    if mtnum != 3:
        return s1[0]
    else:
        return '-'.join(s1[:2])

# prepare the layout

panel_groups = {}
for k, p in figures.items():
    groupname = get_groupname(k)
    curgroup = panel_groups.setdefault(groupname, [])
    if len(p) > 0:
        currow = row(p[0])  # if several panels: row(p[0], p[1], ...)
        curgroup.append(TabPanel(child=currow, title=k))

super_panel_groups = []
for k, p in panel_groups.items():
    curtabs = TabPanel(child=Tabs(tabs=p), title=k)
    super_panel_groups.append(curtabs)

# add the SACS data Panel

curtab = TabPanel(child=column(sacs_tables), title="SACS")
super_panel_groups.append(curtab)


# create the layout

layout = Tabs(tabs=super_panel_groups)


# save to ile

save(layout, filename='testplot.html', title='Plots', template="basic.html", resources="inline")

