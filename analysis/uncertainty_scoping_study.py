import re
import time
import numpy as np
from data_preparation import (
    priortable,
    rpriortable,
    exptable,
    rS,
    covmat_blocks,
    inv_covmat_blocks,
    S_blocks,
    block_starts,
    block_stops,
    expcov as expcov_rel,
    expcov_abs,
)
from unctools import (
    calc_target_unc,
    calc_rel_target_unc,
    calc_postmean,
    calc_postcov,
    get_S_by_blocks,
    get_idcs_by_blocks,
    get_exptable_by_blocks,
    get_expcov_by_blocks,
    get_normalization_unc,
    add_normunc,
    calc_postcov2,
    _to_dense_array,
    cov2cor,
    plot_cormat,
)
from genetic_algo import Evolution
import pandas as pd

# First, we are using a genetic algorithm to explore
# what datablocks are responsible for explaining most
# of the observed uncertainty reduction for the
# Pu-239(n,f) cross section at 3 MeV

datapoint_index = 2905  # Pu-239(n,f) at 3 MeV 
pred = exptable.loc[datapoint_index, 'PRED']
projvec = rS[datapoint_index,:].toarray()
base_score = calc_target_unc(projvec, S_blocks, inv_covmat_blocks, []) / pred
gene_pool = [i for i in range(len(covmat_blocks))] 

def fitfunc(chromosome, projvec, scale, shift):
    return (calc_target_unc(projvec, S_blocks, inv_covmat_blocks, chromosome) * scale - shift) / len(chromosome)
    

evolution = Evolution(gene_pool, fitfunc, {'projvec': projvec, 'scale': 1/pred, 'shift': base_score})  

s1 = time.time()
evolution.init_population(10)
s2 = time.time()
evolution.average_generation_fitness()

for i in range(3):
    evolution.next_generation()
    best_fitness = max([evolution.fitness(c) for c in evolution._population])
    print(f'Best fitness: {best_fitness}')
    avg_fitness = evolution.average_generation_fitness()
    print(f'Avg fitness: {avg_fitness}')


# achieved results
# remove (0, 1, 2, 71, 78, 80, 81, 160, 183, 191) : score=0.0016 : 2.2% 
# ??? (54, 62, 65, 81, 83, 84, 85, 166, 178, 181, 191) : score=... : 4.6% 
# ??? (54, 62, 65, 78, 80, 81, 83, 84, 85, 166, 178, 181, 191)

# Retrieve the datasets involved and show impact on uncertainties 
best_chromosome = [0, 1, 2, 71, 78, 80, 81, 160, 183, 191]
get_exptable_by_blocks(exptable, best_chromosome, block_starts, block_stops)
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, best_chromosome, invert=False) / pred 
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, best_chromosome, invert=True) / pred 

# We have observed that a very well performing group of datablocks 
# results in a posterior uncertainty of Pu-9(n,f) at 3 MeV of 2.2%.
# We have also observed that the complementary set of datablocks
# leads to a similar evaluated uncertainty of about 2% as well.
# If we would only look at this marginal distributions, we would
# expect an uncertainty of about 1.4% if including both groups.
# In reality we are observing ~0.6%, meaning there are synergistic
# effects at play somehow induced by the correlations between
# other quantities.
all_datablocks = [i for i in range(len(covmat_blocks))]
important_datablocks = [0, 1, 2, 71, 78, 80, 81, 160, 183, 191]
other_datablocks = sorted(set(all_datablocks) - set(important_datablocks)) 

calc_target_unc(projvec, S_blocks, inv_covmat_blocks, important_datablocks, invert=True) / pred 
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, other_datablocks, invert=True) / pred 

# As a next step, we are removing the best performing group
# and apply the genetic algorithm another time to identify
# the datablocks in the remaining group that constrain the
# posterior uncertainties the most.

evolution2 = Evolution(other_datablocks, fitfunc, {'projvec': projvec, 'scale': 1/pred, 'shift': base_score})  
evolution2.init_population(50)

for i in range(10):
    evolution2.next_generation()
    best_fitness = max([evolution2.fitness(c) for c in evolution2._population])
    print(f'Best fitness: {best_fitness}')
    avg_fitness = evolution2.average_generation_fitness()
    print(f'Avg fitness: {avg_fitness}')

# Once this more minimal set of datablocks in the complementary
# group is identified, we include the important group and this
# complementary reduced to group to see if we still observe the
# synergistic effect for the evaluation uncertainty.

important_datablocks2 = (62, 65, 83, 84, 85, 166, 178, 181, 190)
compl_datablocks2 = sorted(set(all_datablocks) - set(important_datablocks2)) 

calc_target_unc(projvec, S_blocks, inv_covmat_blocks, important_datablocks2, invert=True) / pred
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, compl_datablocks2, invert=True) / pred

# Check assumption whether the two groups:
# important_datanblocks and important_datablocks2 almost completely
# determine the final uncertainties

all_important_datablocks = list(important_datablocks) + list(important_datablocks2)
all_important_datablocks_compl = sorted(set(all_datablocks) - set(all_important_datablocks)) 
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, all_important_datablocks, invert=True) / pred
calc_target_unc(projvec, S_blocks, inv_covmat_blocks, all_important_datablocks_compl, invert=True) / pred

# Including all datablocks in all_important_datablocks, we arrive at an
# evaluated uncertainty of 0.74%, which is only 0.1% larger than including
# all datablocks (173)

# Therefore, we can restrict our search for synergistic effects to the two
# groups important_datablocks and important_datablocks2

# How to explore?
# desiderata:
#  - we want to find some kind of minimal example (least number of datablocks involved)
#  - the synergistic effect should be clearly noticeable

base_var = np.square(calc_target_unc(projvec, S_blocks, inv_covmat_blocks, important_datablocks, invert=True))
for block in other_datablocks:
    # selblocks = list(other_datablocks)
    selblocks = [block]
    solo_var = np.square(calc_target_unc(projvec, S_blocks, inv_covmat_blocks, selblocks, invert=True)) 
    curblocks = sorted(important_datablocks + selblocks) 
    comp_var = np.square(calc_target_unc(projvec, S_blocks, inv_covmat_blocks, curblocks, invert=True))
    exp_var = 1/(1/solo_var + 1/base_var)
    if comp_var < exp_var * 0.9:
        solo_unc_rel = np.sqrt(solo_var) / pred
        base_unc_rel = np.sqrt(base_var) / pred
        exp_unc_rel = np.sqrt(exp_var) / pred
        comp_unc_rel = np.sqrt(comp_var) / pred
        # print(f'block: {block} --- base_var: {base_var} --- solo_var: {solo_var} --- exp_var: {exp_var} --- comp_var: {comp_var} --- comp_var/exp_var: {comp_var/exp_var:.3f}')
        print(f'block: {block} --- base_unc_rel: {base_unc_rel:.4f} --- solo_unc_rel: {solo_unc_rel:.4f} --- exp_unc_rel: {exp_unc_rel:.4f} --- comp_unc_rel: {comp_unc_rel:.4f} --- comp_unc_rel/exp_unc_rel: {comp_unc_rel/exp_unc_rel:.3f}')

# Output:
# block: 54 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0186 --- comp_unc_rel/exp_unc_rel: 0.857
# block: 56 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0195 --- comp_unc_rel/exp_unc_rel: 0.897
# block: 62 --- base_unc_rel: 0.0217 --- solo_unc_rel: 7308.5862 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0165 --- comp_unc_rel/exp_unc_rel: 0.761
# block: 65 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0127 --- comp_unc_rel/exp_unc_rel: 0.583
# block: 83 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0143 --- comp_unc_rel/exp_unc_rel: 0.658
# block: 84 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0116 --- comp_unc_rel/exp_unc_rel: 0.534
# block: 85 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0156 --- comp_unc_rel/exp_unc_rel: 0.719
# block: 166 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0138 --- comp_unc_rel/exp_unc_rel: 0.636
# block: 178 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0153 --- comp_unc_rel/exp_unc_rel: 0.702
# block: 181 --- base_unc_rel: 0.0217 --- solo_unc_rel: 53296.1390 --- exp_unc_rel: 0.0217 --- comp_unc_rel: 0.0113 --- comp_unc_rel/exp_unc_rel: 0.522

get_exptable_by_blocks(exptable, [84], block_starts, block_stops)

# Is there any absolute measurement with a too low systematic error component?
# Any absolute measurement without a fully correlated uncertainty component will cause a strong reduction with a shape dataset
postmean_ref = calc_postmean(exptable.PRED, exptable.DATA, S_blocks, inv_covmat_blocks, block_idcs=all_datablocks, scale=rpriortable.PRED, invert=True)

postmean = calc_postmean(exptable.PRED, exptable.DATA, S_blocks, inv_covmat_blocks, block_idcs=important_datablocks, scale=rpriortable.PRED, invert=True)
postcov = calc_postcov(S_blocks, inv_covmat_blocks, block_idcs=important_datablocks, scale=rpriortable.PRED, invert=True)

# show correlation matrix
selidx = exptable.loc[exptable.NODE == 'exp_1028'].index
selexpcov = expcov[np.ix_(selidx, selidx)].toarray()
selexpcor = selexpcov / np.sqrt(selexpcov.diagonal().reshape(-1,1) * selexpcov.diagonal().reshape(1,-1))

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(selexpcor, interpolation='none', cmap='viridis')
plt.title("Correlation Matrix")
plt.colorbar(label='Correlation')
plt.xlabel("Variables")
plt.ylabel("Variables")
# plt.xticks(range(len(selexpcor)))
# plt.yticks(range(len(selexpcor)))
plt.show()


# TODO: create function to extract systematic uncertainty from covariance matrix
# extract systematic uncertainty from posterior covariance matrix

# check normalization uncertainties of all experiments

expids = pd.unique(exptable.NODE)  
for expid in expids:
    curexptable = exptable.loc[exptable.NODE == expid]
    curreac = curexptable.REAC.iloc[0]
    if re.match('MT:[2489]', curreac):
        continue
    curidcs = curexptable.index
    curexpcov = expcov_rel[np.ix_(curidcs, curidcs)]
    curunc = get_normalization_unc(curexpcov)
    if curunc < 0.01:
        print('#########################')
        print(f'expid: {expid} --- sysunc: {curunc}')
        print(curexptable)

# Conclusion: Even if we are adding 1% to all the datasets
# with normalization uncertainty smaller than 1%, the
# issue of too small uncertainty ~0.6% in Pu-239(n,f) still remains.

red_expcov_rel = get_expcov_by_blocks(
    expcov_rel, important_datablocks, block_starts, block_stops
)

# The following analysis demonstrates that the remaining systematic
# uncertainty component between 1 and 5 MeV is about 0.86% even
# if only the most significant datablocks (important_datablocks)
# are considered in the evaluation.

# postcov = calc_postcov(S_blocks, inv_covmat_blocks, block_idcs=important_datablocks, scale=rpriortable.PRED, invert=True)
# 
# curidcs = get_idcs_by_blocks(important_datablocks, block_starts, block_stops)
# postcov2 = calc_postcov2(
#     rS, expcov_rel, exptable.PRED, rpriortable.PRED, block_starts, block_stops, reg=1e-10, idcs=curidcs,
# )

red_priortable = rpriortable[
    (rpriortable.REAC == 'MT:1-R1:9') & (rpriortable.ENERGY >= 0.9) & (rpriortable.ENERGY <= 5.0)
]
red_idcs = red_priortable.index
red_postcov = postcov[np.ix_(red_idcs, red_idcs)]
get_normalization_unc(red_postcov)

# Follow-up question: Is there a single datablock where I can introduce a significant
# normalization uncertainty that would completely change the Pu-239(n,f) uncertainty

# postcov = calc_postcov(S_blocks, inv_covmat_blocks, block_idcs=[], scale=rpriortable.PRED, invert=False)

keep_idcs = get_idcs_by_blocks(important_datablocks, block_starts, block_stops)
expids = exptable.loc[keep_idcs].DB_IDX.drop_duplicates().tolist()
quantidx = rpriortable[(rpriortable.REAC == 'MT:1-R1:9') & (rpriortable.ENERGY==3)].index
expcov_rel = _to_dense_array(expcov_rel)
for expid in expids:
    print(f'trying uncertainty inflation of {expid}') 
    curidcs = exptable[exptable.DB_IDX == expid].index
    curcov = add_normunc(expcov_rel, keep_idcs, 0.00)
    curpostcov2 = calc_postcov2(
        rS, curcov, exptable.PRED, rpriortable.PRED, block_starts, block_stops, idcs=keep_idcs
    )
    curunc = np.sqrt(curpostcov2[quantidx, quantidx])
    sysunc = get_normalization_unc(curpostcov2[np.ix_(red_idcs, red_idcs)])
    print(f'point unc: {curunc} --- sysunc: {sysunc}')

# Observation:
# even if I add a fully correlated component (1000%) over entire datablock, the systematic component remains at ~0.8-1.1%
# if I add a huge amount of normalization uncertainty (1000%) on all experiments (correlating everything), I get no more than 1.6% 

def test_scenario(quantidx, red_idcs, keep_idcs, curcov):
    # Assume a large fully correlated relative systematic component for each dataset
    curdatablocks = all_datablocks.copy()
    # curdatablocks.remove(191)
    curcov = _to_dense_array(curcov)
    curpostcov2 = calc_postcov2(
        rS, curcov, exptable.PRED, rpriortable.PRED, block_starts, block_stops, idcs=keep_idcs
    )
    curunc = np.sqrt(curpostcov2[quantidx, quantidx])
    sysunc = get_normalization_unc(curpostcov2[np.ix_(red_idcs, red_idcs)])
    print(f'point unc: {curunc} --- sysunc: {sysunc}')
    # propagate to SACS values
    prior_pred = np.array(rpriortable.PRED).reshape(-1, 1)
    curpostcov2_abs = curpostcov2 * prior_pred.T * prior_pred 
    rrS = rS[6673:6674,:]
    sacs_unc = np.sqrt(rrS @ curpostcov2_abs @ rrS.T) / exptable.PRED[6673]
    print(f'sacs Pu9(n,f) unc: {sacs_unc}')



# SCENARIO: add additional normalization uncertainty to each experiment
curdatablocks = all_datablocks.copy()
curdatablocks.remove(191)
keep_idcs = get_idcs_by_blocks(curdatablocks, block_starts, block_stops)
quantidx = rpriortable[(rpriortable.REAC == 'MT:1-R1:9') & (rpriortable.ENERGY==3)].index
expids = exptable.loc[keep_idcs].NODE.drop_duplicates().tolist()
curcov = _to_dense_array(expcov_rel)
for expid in expids:
    curidcs = exptable[(exptable.NODE == expid) & (exptable.REAC.str.match('MT:([13]|10)-R1:([89]|10)(-R2:([89]|10))?$'))].index
    if len(curidcs) > 0:
        print(f'adjusting {exptable.loc[curidcs[0]].REAC}') 
    curcov = add_normunc(curcov, curidcs, 0.03)

test_scenario(quantidx, red_idcs, keep_idcs, curcov)

# SCENARIO: add additional normalization uncertainty to each (absolute) measurement
keep_idcs = get_idcs_by_blocks(important_datablocks, block_starts, block_stops)
quantidx = rpriortable[(rpriortable.REAC == 'MT:1-R1:9') & (rpriortable.ENERGY==3)].index
reacids = exptable.loc[keep_idcs].REAC.drop_duplicates().tolist()
curcov = _to_dense_array(expcov_rel)
for reacid in reacids:
    if reacid not in (
        'MT:1-R1:9', 'MT:1-R1:8', 'MT:1-R1:10', 'MT:3-R1:9-R2:8', 'MT:3-R1:10-R2:8'
    ):
        continue
    curidcs = exptable[exptable.REAC == reacid].index
    curcov = add_normunc(curcov, curidcs, 1000)

test_scenario(quantidx, red_idcs, keep_idcs, curcov)



# Test forward propagation

postmean = calc_postmean(exptable.PRED, exptable.DATA, S_blocks, inv_covmat_blocks, block_idcs=all_datablocks, scale=rpriortable.PRED, invert=True)
postcov = calc_postcov(S_blocks, inv_covmat_blocks, block_idcs=all_datablocks, scale=rpriortable.PRED, invert=True)

sel = rpriortable.loc[(rpriortable.REAC == 'MT:1-R1:9') & (rpriortable.ENERGY >= 0.1) & (rpriortable.ENERGY <= 10.)].index
np.sqrt(np.diagonal(postcov))[754]
ens = rpriortable.loc[sel, 'ENERGY'].to_numpy()

new_postcov = add_normunc(postcov, sel, 0.01)
sacs_idx = exptable.loc[exptable.REAC == 'MT:6-R1:9'].index
rrS = rS[sacs_idx,:]

np.sqrt(rrS @ new_postcov @ rrS.T)
np.sqrt(rrS @ postcov @ rrS.T)

cormat = cov2cor(new_postcov[np.ix_(sel, sel)])
plot_cormat(ens, ens, cormat)
