# from gmapy.legacy.database_reading import read_gma_database  # only reads legacy GMA
from gmapy.data_management.database_IO import read_gma_database  # reads both legacy and JSON GMA database
from gmapy.legacy.conversion_utils import convert_GMA_database_to_JSON
import json
from gmapy.data_management.json.encoder import JSONEncoder
import os

# filepaths
json_dbpath = 'data_commit_4610454.json'
modified_dbpath = 'data.json'

# load GMA JSON database
db = read_gma_database(json_dbpath)
db['datablocks'] = db.pop('datablock_list')  # for consistent naming
db['prior'] = db.pop('prior_list')  # for consistent naming

# load reduced CRD file
red_crd_path = os.path.join('reduced_crd_files', 'LA-UR-25-32229-1_textfile.json')
with open(red_crd_path, 'r') as f:
    crd_red = json.load(f)

# get datablock/dataset mapping for datasets in GMA database
gma_dataset_map = {}
for blck_idx, blck in enumerate(db['datablocks']):
    for ds_idx, ds in enumerate(blck['datasets']):
        if 'NS' in ds:
            ds_id = ds['NS']
        else:
            ds_id = ds['identifier']
        curlist = gma_dataset_map.setdefault(ds_id, [])
        curlist.append((blck_idx, ds_idx))

# get datablock/dataset mapping for datasets in GMA database
crd_dataset_map = {}
for blck_idx, blck in enumerate(crd_red['datablocks']):
    for ds_idx, ds in enumerate(blck['datasets']):
        if 'NS' in ds:
            ds_id = ds['NS']
        else:
            ds_id = ds['identifier']
        curlist = crd_dataset_map.setdefault(ds_id, [])
        curlist.append((blck_idx, ds_idx))

# merge CRD file into GMA
new_blocks = {}
for key, lst in crd_dataset_map.items():
    crd_blck_idx, crd_ds_idx = lst[0]
    new_dataset = crd_red['datablocks'][crd_blck_idx]['datasets'][crd_ds_idx]
    # if dataset id already exists in GMA, replace dataset
    if key in gma_dataset_map:
        gma_lst = gma_dataset_map[key]
        if len(gma_lst) > 1:
            raise IndexError(f'Ambiguous dataset id for {key}')
        gma_blck_idx, gma_ds_idx = gma_lst[0]
        db['datablocks'][gma_blck_idx]['datasets'][gma_ds_idx] = new_dataset
        print(f'replaced {key} in GMA database by dataset in CRD file')
    # otherwise, add the dataset
    else:
        new_lst = new_blocks.setdefault(key, [])
        new_lst.append(new_dataset)
        print(f'added new dataset {key} to GMA database')

db['datablocks'] += [
    {'type': 'legacy-experiment-datablock', 'datasets': lst}
    for lst in new_blocks.values()
]


# write to disk
with open(modified_dbpath, 'w') as f:
    json.dump(db, f, cls=JSONEncoder,values_per_line=10, min_list_len=15, indent=2)


