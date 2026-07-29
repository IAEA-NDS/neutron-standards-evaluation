import io
import sys
import numpy as np

sys.path.insert(0, '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/datpy')
sys.path.insert(0, '/home/gschnabel/Seafile/OmegaSpace/gmapy')
from types import SimpleNamespace
from datpy.data_io.legacy import data_input
data_input.Dataset = lambda **kw: SimpleNamespace(**kw)
read_datablocks = data_input.read_datablocks
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management import datablock_api as dbapi

CANDS = [
    '/home/gschnabel/Seafile/OmegaSpace/release_preparation/tasks/task_5i/GMDATA26-4.CRD',
    '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-pipeline-private/input/input_std2017.crd',
    '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/datpy/legacy-tests/test_003/input/GMDATA.CRD',
    '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-pipeline/fortran_repro_study/input_std2017.crd',
    '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/data/orig_crd_files/GMDATA.CRD',
]

db = read_gma_database(
    '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/data/data.json')
json_ds = {}
for cur_db in db['datablock_list']:
    for ds in dbapi.dataset_iterator(cur_db):
        if 'CO' not in ds:
            continue
        co = np.array(ds['CO'], dtype=float)
        if co.ndim != 2 or co.shape[1] < 12:
            continue
        ens = np.array(ds['E'], dtype=float)
        json_ds[ds['NS']] = (ens, co)
print(f'[CRD] data.json datasets with CO: {len(json_ds)}', flush=True)

for path in CANDS:
    try:
        with open(path, 'r') as f:
            content = f.read()
        import logging
        logging.disable(logging.CRITICAL)
        blocks = read_datablocks(io.StringIO(content))
    except Exception as exc:
        print(f'[CRD] {path}: PARSE FAILED: {type(exc).__name__}: {exc}',
              flush=True)
        continue
    crd_ds = {}
    for blk in blocks:
        for d in blk:
            crd_ds[d.dataset_id] = d
    common = sorted(set(crd_ds) & set(json_ds))
    n_pt = n_pt_match = n_infl = n_defl = 0
    infl_list = []
    for ns in common:
        d = crd_ds[ns]
        ens_j, co_j = json_ds[ns]
        pos = {round(float(e), 6): i for i, e in enumerate(d.energies)}
        for i, e in enumerate(ens_j):
            n_pt += 1
            k = pos.get(round(float(e), 6))
            if k is None:
                continue
            n_pt_match += 1
            c3_crd = float(np.asarray(d.uncertainties)[2, k])
            c3_json = float(co_j[i, 2])
            if c3_json > c3_crd + 0.05:
                n_infl += 1
                infl_list.append((ns, float(e), c3_crd, c3_json))
            elif c3_json < c3_crd - 0.05:
                n_defl += 1
    print(f'[CRD] {path.split("/OmegaSpace/",1)[1]}', flush=True)
    print(f'[CRD]   crd datasets: {len(crd_ds)}  common w/ json: '
          f'{len(common)}  json pts in common: {n_pt}  '
          f'energy-matched: {n_pt_match}', flush=True)
    print(f'[CRD]   comp3 INFLATED (json>crd): {n_infl}   '
          f'deflated: {n_defl}', flush=True)
    if infl_list:
        by_ds = {}
        for ns, e, a, b in infl_list:
            by_ds.setdefault(ns, []).append((e, a, b))
        top = sorted(by_ds.items(), key=lambda t: -len(t[1]))[:8]
        for ns, pts in top:
            mx = max(b - a for _, a, b in pts)
            print(f'[CRD]     exp_{ns}: {len(pts)} inflated pts, '
                  f'max increase {mx:.2f}%', flush=True)
