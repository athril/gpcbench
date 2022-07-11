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
import re
import pickle
from m3gp.M3GP import M3GP




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
#est = M3GP(operators=['sub','add','ge','gt','mul','div','xor'])
est = M3GP()


# In order to properly benchmark a method, we need to define its parameters and their values.
# Please set the expected range of hyper parameters for your method below. For details, please refer to Optuna.
def params_myParamScope(trial):

    params={
        'max_generation' : trial.suggest_int('max_generation',10,200,step=10),
        'population_size' : trial.suggest_int('population_size',50,500,step=10),
        'tournament_size' : trial.suggest_int('tournament_size', 5, 20),
        'limit_depth' : trial.suggest_int('limit_depth', 2, 10)

    }
    return params



#Perform optimization of the method on DIGEN datasets
results=benchmark.optimize(est=est,datasets=datasets, parameter_scopes=params_myParamScope, storage='sqlite:///'+datasets+'-M3GP.db',local_cache_dir='.')
pickle.dump(results, open(datasets+'-M3GP.pkl','wb'))


