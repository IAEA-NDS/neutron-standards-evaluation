import numpy as np
import tensorflow as tf
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management import datablock_api as dbapi
from gmapy.data_management.object_utils import load_objects

db = read_gma_database('../data/data.json')
likelihood, exptable = load_objects(
    'output/01_model_preparation_output.pkl', 'likelihood', 'exptable'
)
optres, = load_objects(
    'output/02_parameter_optimization_output.pkl', 'optres'
)
x_mode = np.array(optres.position)
pars_t = tf.constant(x_mode[:likelihood._num_params], dtype=tf.float64)
p_all = np.reshape(np.array(likelihood._propfun(pars_t)), -1)
y_all = np.reshape(np.array(likelihood._like_data, dtype=float), -1)
node_all = exptable.NODE.to_numpy().astype(str)
en_all = exptable.ENERGY.to_numpy()
reac_all = exptable.REAC.to_numpy().astype(str)

n_pts = n_c3 = n_spike = n_ident_viol = 0
z_spike, z_plain = [], []
spike_channels = {}
c3_share_num = c3_share_den = 0.
spike_datasets = {}

for cur_db in db['datablock_list']:
    for ds in dbapi.dataset_iterator(cur_db):
        if 'CO' not in ds:
            continue
        co = np.array(ds['CO'], dtype=float)
        if co.ndim != 2 or co.shape[1] < 12:
            continue
        if ds.get('NNCOX', 0) != 0:
            co = co / 10.
        ens = np.array(ds['E'], dtype=float)
        ns = ds['NS']
        sel = node_all == f'exp_{ns}'
        if not np.any(sel):
            continue
        # align datablock points with (possibly filtered) exptable rows
        pos = {round(float(e), 8): i for i, e in enumerate(ens)}
        rows = np.where(sel)[0]
        keep = [pos.get(round(float(en_all[r]), 8), -1) for r in rows]
        rows = rows[[k >= 0 for k in keep]]
        keep = [k for k in keep if k >= 0]
        if len(keep) == 0:
            continue
        co = co[keep]
        comp3 = co[:, 2]
        relu = np.sqrt(np.sum(co[:, 2:11]**2, axis=1))
        enff = np.array(ds.get('ENFF', []), dtype=float)
        xnoru = np.sum(enff**2) if enff.size else 0.
        tot = np.sqrt(relu**2 + xnoru)
        tot = np.where(tot > 0, tot, np.inf)
        dev = 100. * np.abs(y_all[rows] - p_all[rows]) / np.abs(p_all[rows])
        z = dev / tot
        # 12th-column identity check (F12^2 - sum(F3..F11)^2 = stat part)
        ses = co[:, 11]**2 - relu**2
        n_ident_viol += int(np.sum(ses < -1e-6))
        # spike detection: comp3 much larger than dataset-typical comp3
        med = np.median(comp3[comp3 > 0]) if np.any(comp3 > 0) else 0.
        spike = (comp3 > 0) & (comp3 >= max(2. * med, 1e-6)) \
            if med > 0 else (comp3 > 0)
        strict_spike = spike & (comp3 > 1.5 * np.median(tot))
        n_pts += len(z)
        n_c3 += int(np.sum(comp3 > 0))
        n_spike += int(np.sum(strict_spike))
        c3_share_num += float(np.sum(comp3**2))
        c3_share_den += float(np.sum(tot[np.isfinite(tot)]**2))
        z_spike.extend(z[strict_spike].tolist())
        z_plain.extend(z[~strict_spike].tolist())
        if np.any(strict_spike):
            r = reac_all[rows][0]
            spike_channels[r] = spike_channels.get(r, 0) \
                + int(np.sum(strict_spike))
            spike_datasets[ns] = int(np.sum(strict_spike))

z_spike = np.array(z_spike); z_plain = np.array(z_plain)
print(f'[FOR] points matched: {n_pts}   with comp3>0: {n_c3} '
      f'({100.*n_c3/n_pts:.1f} %)', flush=True)
print(f'[FOR] comp3 share of total variance (global): '
      f'{100.*c3_share_num/c3_share_den:.1f} %', flush=True)
print(f'[FOR] strong comp3 spikes (comp3 >= 2x dataset median and '
      f'> 1.5x total-unc median): {n_spike}', flush=True)
print(f'[FOR] 12th-column identity violations: {n_ident_viol}', flush=True)
if len(z_spike):
    q = np.percentile(z_spike, [25, 50, 75, 90])
    print(f'[FOR] deviation/unc of SPIKE points: quartiles '
          f'{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}  90%: {q[3]:.2f}  '
          f'frac in [2,4]: {np.mean((z_spike > 2) & (z_spike < 4)):.2f}',
          flush=True)
q = np.percentile(z_plain, [25, 50, 75, 90])
print(f'[FOR] deviation/unc of other points: quartiles '
      f'{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}  90%: {q[3]:.2f}', flush=True)
top = sorted(spike_channels.items(), key=lambda t: -t[1])[:10]
print('[FOR] spike counts by reaction:', flush=True)
for r, c in top:
    print(f'[FOR]   {c:5d}  {r}', flush=True)
top = sorted(spike_datasets.items(), key=lambda t: -t[1])[:10]
print('[FOR] top datasets by spikes:', flush=True)
for ns, c in top:
    print(f'[FOR]   {c:5d}  exp_{ns}', flush=True)
