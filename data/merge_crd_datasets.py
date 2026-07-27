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
red_crd_path = os.path.join('reduced_crd', 'file')
with open('red_crd_path', 'r') as f:
    crd_red = json.load(f)

# merge CRD file into GMA
db['datablocks'].insert(0, crd_red['datablocks'][0])

# write to disk
with open(modified_dbpath, 'w') as f:
    json.dump(db, f, cls=JSONEncoder,values_per_line=10, min_list_len=15, indent=2)


# check duplicates
dataset_ids = [ ds['NS']  for dblock in db['datablocks'] for ds in dblock['datasets']]
seen = set()
duplicates = set()
for item in dataset_ids:
    if item in seen:
        duplicates.add(item)
    else:
        seen.add(item)

print(f'duplicates: {duplicates}')
