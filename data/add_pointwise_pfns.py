# Replace the legacy binned Cf-252 fission spectrum in the GMA JSON
# database by the new point-wise evaluated PFNS given as a text file
# in `other_data_sources`. The spectrum is stored as a
# `pointwise-fission-spectrum` prior block (introduced in gmapy for
# this purpose), which interprets the values as a density (1/MeV)
# interpolated according to the declared interpolation law, in
# contrast to the bin-integrated probabilities of the legacy
# representation. Provenance information (source file, sha256 hash,
# date) is embedded in the block itself so that the database remains
# self-documenting.
#
# This script is meant to be run after `merge_crd_datasets.py` as part
# of the database preparation chain:
#   data_commit_4610454.json --(merge_crd_datasets.py)--> data.json
#   data.json --(add_pointwise_pfns.py)--> data.json
# Rerunning it is harmless as the fission spectrum block is replaced
# by type.
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.json.encoder import JSONEncoder
import numpy as np
import hashlib
import datetime
import json
import os

# filepaths
dbpath = 'data.json'
pfns_path = os.path.join(
    'other_data_sources',
    'Eval_MvCf-2520,fPFNSSmoothed_Maerten60degonly' +
    'Chalupka1datapointsrejectLA-U...ANDARD.txt'
)

# the low-energy part of the spectrum follows a power law
# (proportional to sqrt(E)), which is represented exactly by
# log-log interpolation whereas lin-lin interpolation would
# introduce significant errors on the sparse low-energy mesh
interpolation = 'log-log'


def integrate_loglog(energies, values):
    """Exact integral of a log-log interpolated function."""
    e1 = energies[:-1]
    e2 = energies[1:]
    y1 = values[:-1]
    y2 = values[1:]
    p = np.log(y2/y1) / np.log(e2/e1)
    segints = np.where(
        np.abs(p + 1.) > 1e-10,
        y1 * e1 * (np.power(e2/e1, p+1.) - 1.) / (p+1.),
        y1 * e1 * np.log(e2/e1)
    )
    return np.sum(segints)


# load the point-wise PFNS evaluation
pfns = np.loadtxt(pfns_path)
energies = pfns[:, 0]
values = pfns[:, 1]

# sanity checks: mesh ordering and positivity (the latter being
# a requirement for log-log interpolation)
assert np.all(np.diff(energies) > 0.), 'energies not strictly increasing'
assert np.all(values > 0.), 'spectrum values not all positive'

# report normalization under different interpretations; gmapy
# normalizes the spectrum internally so deviations from one are
# inconsequential for the evaluation but are reported here for
# future reference
linlin_int = np.trapz(values, energies)
loglog_int = integrate_loglog(energies, values)
gma_range = (energies >= 2.53e-8) & (energies <= 20.)
loglog_int_gma_range = integrate_loglog(
    energies[gma_range], values[gma_range]
)
print(f'number of mesh points: {len(energies)}')
print(f'energy range: {energies[0]:.3e} to {energies[-1]:.3e} MeV')
print(f'integral (lin-lin interpretation): {linlin_int:.8f}')
print(f'integral (log-log interpretation): {loglog_int:.8f}')
print('integral (log-log) within GMA energy range ' +
      f'2.53e-8 to 20 MeV: {loglog_int_gma_range:.8f}')

# compute hash of the source file for provenance tracking
with open(pfns_path, 'rb') as f:
    pfns_sha256 = hashlib.sha256(f.read()).hexdigest()

# construct the new prior block; NOTE: no `uncertainties` field is
# provided so the spectrum is treated as fixed in the evaluation;
# uncertainties (and a `correlations` matrix) can be added later to
# let the spectrum vary
new_fisblock = {
    'type': 'pointwise-fission-spectrum',
    'description': ('Cf-252(sf) PFNS, smoothed evaluation, ' +
                    'by AIACHNE project, main evaluator Denise Neudecker, ' +
                    '(Maerten 60 deg only, Chalupka 1 data point ' +
                    'rejected), point-wise density in 1/MeV'),
    'source_file': os.path.basename(pfns_path),
    'source_sha256': pfns_sha256,
    'date_added': datetime.date.today().isoformat(),
    'energies': energies.tolist(),
    'values': values.tolist(),
    'interpolation': interpolation
}

# load GMA JSON database
db = read_gma_database(dbpath)
db['datablocks'] = db.pop('datablock_list')  # for consistent naming
db['prior'] = db.pop('prior_list')  # for consistent naming

# replace the fission spectrum block
fis_idcs = [i for i, b in enumerate(db['prior'])
            if b['type'] in ('legacy-fission-spectrum',
                             'pointwise-fission-spectrum')]
if len(fis_idcs) != 1:
    raise IndexError('expected exactly one fission spectrum block ' +
                     f'in the database but found {len(fis_idcs)}')
old_type = db['prior'][fis_idcs[0]]['type']
db['prior'][fis_idcs[0]] = new_fisblock
print(f'replaced `{old_type}` block by `pointwise-fission-spectrum` ' +
      f'block with {interpolation} interpolation')

# write to disk
with open(dbpath, 'w') as f:
    json.dump(db, f, cls=JSONEncoder, values_per_line=10,
              min_list_len=15, indent=2)
