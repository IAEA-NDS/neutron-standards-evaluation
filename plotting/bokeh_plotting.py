import re
import os
from bokeh.palettes import Category20_20
from bokeh.plotting import figure, show, save, curdoc
from bokeh.models import ColumnDataSource, HoverTool, Row, Select, CustomJS, Dropdown
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
)


dfs = prepare_result_data('ea40e40')
exptable = dfs['exptable']
std2017 = dfs['std2017_dt']


pred_list = []
cols = []
# pred_list.append(load_evaluation('01a02a0', '8007 removed', 'green', 'dotdash'))
# pred_list.append(load_evaluation('f42e55d', '1013 to shape', 'blue', 'dotdash'))
pred_list.append(load_evaluation('1e8ce5e', 'recommend_new MCMC', 'orange', 'dashed'))
cols.append("PRED")
pred_list.append(load_evaluation('1e8ce5e', 'recommend_new OPT', 'green', 'dashed'))
cols.append("MAXLIKE")
pred_list.append(load_evaluation('649a9e8', 'recommend_base OPT', 'red', 'solid'))
cols.append("MAXLIKE")
pred_list.append(load_evaluation('649a9e8', 'recommend_base MCMC', 'blue', 'dotdash'))
cols.append("PRED")
pred_list.append(load_evaluation('ea40e40', 'latest eval', 'black', 'solid'))
cols.append("PRED")
pred_list.append(load_evaluation('89dc6bf', 'drop TPC exp (6001) MCMC', 'cyan', 'dashed'))
cols.append("PRED")
pred_list.append(load_evaluation('89dc6bf', 'drop TPC exp (6001) OPT', 'brown', 'dotdash'))
cols.append("MAXLIKE")



# TODO: adhoc addition of Pu9(n,f) cross section
# needs to be done in a cleaner way in the future
from endf_parserpy import EndfParserCpp

parser = EndfParserCpp()

endfb81_path = '/home/gschnabel/bigdata/nuclibs/endfb8.1/neutrons-version.VIII.1'
endfb81_pu9_file = 'n-094_Pu_239.endf'
pu9 = parser.parsefile(os.path.join(endfb81_path, endfb81_pu9_file))

pu9_nf = pu9[3][18]['xstable']
pu9_nf_dt = pd.DataFrame({'REAC': 'MT:1-R1:9', 'ENERGY': pu9_nf['E'], 'PRED': pu9_nf['xs']})
pu9_nf_dt['ENERGY'] = pu9_nf_dt['ENERGY'] / 1e6

pred_list.append(
    {
        'git_hash': None,
        'pred_dt': pu9_nf_dt,
        'label': 'b81',
        'color': 'black',
        'style': 'dashed',
    }
)

cols.append("PRED")

# interpolate STD2017 to energies of experiments and predictions
dt_list = [v['pred_dt'] for v in pred_list]

# add the experimental data
dt_list.append(exptable)
cols.append('DATA')

##################################################
#            PLOTTING
##################################################


# helper function to plot experimental data in current figure

def plot_expdata(figure, reac, expdata, datacol=None, include_usu=False):
    expdata = expdata[expdata.REAC == reac].copy()
    expdata = expdata[expdata.ENERGY > 2.58e-8].copy()
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
                      legend_label=curlabel)
        err_xs = []
        err_ys = []
        uncvals = curdt['UNC_USU'] if include_usu else curdt['UNC']
        for x, y, yerr in zip(curdt['ENERGY'], curdt[datacol], uncvals):
            err_xs.append((x, x))
            err_ys.append((y-yerr, y+yerr))
        figure.multi_line(err_xs, err_ys, color=coldict[node])
        # hover = HoverTool(tooltips=[('Label', '@label')])
        # figure.add_tools(hover)
        # plt.errorbar(curdt.ENERGY, curdt.RENORM_DATA,
        #              yerr=curdt.UNC, fmt='o', label=curlabel)
    Emin = expdata.ENERGY.min()
    Emax = expdata.ENERGY.max()
    return Emin, Emax


# helper function to plot evaluations

def plot_evaluation(figure, reac, pred_dt, datacol, Emin, Emax, label, color, style):
    cdt = pred_dt.query(f'REAC == "{curreac}" & ENERGY >= {Emin} & ENERGY <= {Emax}')
    cdt = cdt.copy()
    cursource = ColumnDataSource(data=cdt)
    figure.line(
        'ENERGY', datacol, source=cursource,
        color=color, line_dash=style, legend_label=label)

# plot comparing absolute cross sections

figures = {}
allreacs = {}
for curreac in pred_list[0]['pred_dt'].REAC.unique():
    m = re.findall(r'R\d+:(\d+)', curreac)
    if any(int(x) > 10 for x in m):
        continue
    if curreac.startswith('MT:6') or curreac.startswith('MT:10'):
        continue
    subfigures = []
    # first with RENORM_ML data
    curfigure = figure(title=f'{curreac}', width=1000, height=800, toolbar_location='above', name=curreac)
    Emin, Emax = plot_expdata(curfigure, curreac, exptable, datacol='DATA')
    for pred in pred_list:
        plot_evaluation(
            curfigure, curreac, pred['pred_dt'], 'PRED',
            Emin*0.9, Emax*1.1, label=pred['label'],
            color=pred['color'], style=pred['style'],
        )
    subfigures.append(curfigure)
    # save everything
    figures[curreac] = subfigures


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
    currow = row(p[0])  # if several panels: row(p[0], p[1], ...)
    curgroup.append(TabPanel(child=currow, title=k))

super_panel_groups = []
for k, p in panel_groups.items():
    curtabs = TabPanel(child=Tabs(tabs=p), title=k)
    super_panel_groups.append(curtabs)

layout = Tabs(tabs=super_panel_groups)

# save to ile

save(layout, filename='testplot.html', title='Plots', template="basic.html", resources="inline")

