import sys
import pathlib
# resolve gmapy to the submodule of this repository (must match the
# version used to create the pickles loaded below)
sys.path.insert(
    0, (pathlib.Path(__file__).resolve().parents[1] / 'gmapy').as_posix()
)
import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
from gmapy.tf_uq.inference import generate_MCMC_chain
# These imports are necessary to satisfy
# dependencies of objects loaded by dill above
import numpy as np

outdir = os.environ.get('EVAL_OUTPUT_DIR', 'output')

post, likelihood = load_objects(
    f'{outdir}/01_model_preparation_output.pkl',
    'post', 'likelihood'
)
optres, = load_objects(
    f'{outdir}/02_parameter_optimization_output.pkl',
    'optres'
)

# set seed for MCMC
tf.random.set_seed(42)

# define essential input quantities for MCMC
optvals = optres.position

s1 = time.time()
chain, tracing_info = generate_MCMC_chain(
    optvals, post.log_prob, post.neg_log_prob_hessian,
    nugget=1e-8, step_size=0.005, num_burnin_steps=int(3e3),
    num_results=int(15e3), num_leapfrog_steps=5
)
s2 = time.time()
print(f's2-s1: {s2-s1}', flush=True)

save_objects(
    f'{outdir}/03_mcmc_sampling_output.pkl', locals(),
    'chain', 'tracing_info'
)
