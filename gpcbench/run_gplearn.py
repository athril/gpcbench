import sys
import numpy as np
import pandas as pd
import re
import random
import itertools
import operator
from deap import base, tools, gp, creator
from digen import Benchmark
import argparse
from gplearn.genetic import SymbolicClassifier
import re
import pickle


# Load a package with DIGEN benchmark
benchmark=Benchmark(n_trials=200, timeout=100000)

#seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)", required=False, nargs='?')
args = parser.parse_args()

datasets=benchmark.list_datasets()
if args.dataset is not None:
    assert(args.dataset in datasets)
    datasets=args.dataset

# Create your default class here or import from the package. As an example, we re benchmarking ExtraTreesClassifier from scikit-learn:
from sklearn.ensemble import ExtraTreesClassifier
est=SymbolicClassifier()


# In order to properly benchmark a method, we need to define its parameters and their values.
# Please set the expected range of hyper parameters for your method below. For details, please refer to Optuna.
def params_myParamScope(trial):

    params={
        'tournament_size' : trial.suggest_int('tournament_size',10,25),
        'generations' : trial.suggest_int('generations', 10, 200, step=10),
        'population_size': trial.suggest_int('population_size', 50, 500, step=10),
        'p_crossover' : trial.suggest_float('p_crossover', 0.1, 0.9, step=0.1)
    }
    return params



est = SymbolicClassifier(
                        init_method='half and half',
                        function_set= ('add', 'sub', 'mul', 'div', 'log','sqrt'),
                        population_size=1000,
                        generations=100
                       )



#Perform optimization of the method on DIGEN datasets
results=benchmark.optimize(est=est,datasets=datasets, parameter_scopes=params_myParamScope, storage=None ,local_cache_dir='.')
#results[datasets]['classifier']=results[datasets]['classifier'].get_params()
pickle.dump(results, open(datasets+'-gplearn.pkl','wb'))

