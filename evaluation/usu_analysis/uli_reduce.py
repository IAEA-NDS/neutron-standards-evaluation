"""Step 2 of the manual reduction workflow (unreduced json -> reduced
json), identical to `python -m datpy.datpy --input <in> --output <out>`
except that datpy.reduction.ULI is overridden by the first argument."""
import sys

sys.path.insert(0, '/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/datpy')
import datpy.reduction as reduction

uli = float(sys.argv[1])
reduction.ULI = uli

from datpy.datpy import run_datp
run_datp(sys.argv[2], sys.argv[3])
print(f'[ULI] reduction with ULI={uli}: {sys.argv[2]} -> {sys.argv[3]}')
