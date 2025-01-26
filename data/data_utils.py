import pandas as pd
import numpy as np
import os
import json


def load_std2017_data():
    """Load STD2017 data and create experimental dataframe."""
    data_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(data_path, 'data.json'), 'r') as f:
        prior = json.load(f)['prior']

    prior = [p for p in prior if 'CLAB' in p]
    mtdic = {r['CLAB'].strip(): f'MT:1-R1:{r["ID"]}' for r in prior}

    descr_list = []
    outdt_list = []

    datadir = os.path.join(data_path, 'std2017')

    tmp = pd.read_csv(
        os.path.join(datadir, 'std17-003_Li_006.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = '6Li(n,a)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1000', 'REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp.CS.to_numpy(), 'UNC': tmp.DCS.to_numpy(), 'DESCR': descr}))

    tmp = pd.read_csv(
        os.path.join(datadir, 'std17-005_B_010.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = '10B(n,a1)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1001','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp.CS.to_numpy(), 'UNC': tmp.DCS.to_numpy(), 'DESCR': descr}))
    descr = '10B(n,a)'
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1002','REAC': 'MT:5-R1:3-R2:4', 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS.1'].to_numpy(), 'UNC': tmp['DCS.1'].to_numpy(), 'DESCR': descr}))

    tmp = pd.read_csv(
        os.path.join(datadir, 'std17-079_Au_197.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = 'Au(n,g)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1003','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))

    tmp = pd.read_csv(
        os.path.join(datadir, 'std17-092_U_235.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = 'U5(n,f)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1004','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))

    tmp = pd.read_csv(
        os.path.join(datadir, 'std17-092_U_238.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = 'U8(n,f)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1005','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))

    tmp = pd.read_csv(
        os.path.join(datadir, 'rec17-094_Pu_239.txt'), comment='#', index_col=None, sep=r'\s+'
    )
    descr = 'PU9(n,f)'
    descr_list.append(descr)
    outdt_list.append(pd.DataFrame({'NODE': 'exp_1006','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))


    # now for the thermal neutron constants
    tmp = pd.read_csv(
        os.path.join(datadir, 'Standards2017_TNC.txt'), comment='#', index_col=0, sep=r'\s+'
    )

    isos = tmp.columns
    isos = isos[~isos.str.endswith('UNC')]
    quants = tmp.index
    exp_cnt = 1005
    for iso in isos:
        for q in quants:
            exp_cnt += 1
            descr = f'{q}-{iso}'
            if descr == 'SF-U5':
                descr = 'U5(n,f)'
            elif descr == 'SF-PU9':
                descr = 'PU9(n,f)'
            # create dataframe
            outdt_list.append(pd.DataFrame({
                'NODE': 'exp_' + str(exp_cnt), 'REAC': [mtdic[descr]],
                'ENERGY': 2.53e-8, 'DATA': tmp.loc[q, iso],
                'UNC': tmp.loc[q, iso + '-UNC'], 'DESCR': descr
            }))

    outdt = pd.concat(outdt_list, ignore_index=True)
    outdt = outdt.sort_values(['REAC', 'ENERGY'], ignore_index=True)
    return outdt, descr_list
