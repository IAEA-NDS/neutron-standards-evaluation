import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import load_objects

parser = argparse.ArgumentParser(
    prog='Convergence Checker',
    description='Check the convergence of MAP estimates by gmapy'
)

parser.add_argument(
    '-c', '--commit', type=str, required=True,  help='Commit id associated with result')
parser.add_argument('--eps', type=float, default=0.0001, help='percentual perturbation magnitude') 

args = parser.parse_args()

eps = args.eps
githash = args.commit 
artefact_dir = Path(f'../output/{githash}/evaluation/output')

post, = load_objects(artefact_dir / '01_model_preparation_output.pkl', 'post')
optres, = load_objects(artefact_dir / '02_parameter_optimization_output.pkl', 'optres')

optvec = optres.position.numpy()
log_prob = tf.function(post.log_prob)
max_log_prob = log_prob(optvec).numpy()

for i in range(optvec.size):
    print(f'checking perturbation of index {i} out of {optvec.size}...')
    curvec = optvec.copy()
    curvec[i] *= (1+eps)
    cur_log_prob1 = log_prob(curvec).numpy()
    curvec = optvec.copy()
    curvec[i] *= (1-eps) 
    cur_log_prob2 = log_prob(curvec).numpy()
    print(f'assert {max_log_prob} > {cur_log_prob1}')
    assert max_log_prob > cur_log_prob1
    print(f'assert {max_log_prob} > {cur_log_prob2}')
    assert max_log_prob > cur_log_prob2

