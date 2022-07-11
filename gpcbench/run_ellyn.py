#! /usr/bin/env python3

from digen import Benchmark
import argparse
from ellyn import ellyn
from sklearn.metrics import accuracy_score 

import pickle

# Load a package with DIGEN benchmark
benchmark = Benchmark(n_trials=200,timeout=100000)

# seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
    required=False,
    nargs='?')
args = parser.parse_args()

datasets = benchmark.list_datasets()
if args.dataset is not None:
    assert (args.dataset in datasets)
    datasets = args.dataset


est = ellyn(classification=True, 
            class_m4gp=True, 
            prto_arch_on=True,
            scoring_function=accuracy_score,
            shuffle_data=True)


def params_myParamScope(trial):
    rt_cross=trial.suggest_float('rt_cross',0.1,0.9, step=0.1)

    params = {
        'g': trial.suggest_int('g', 10, 200, step=10),
        'selection': trial.suggest_categorical(name='selection', choices=['lexicase', 'tournament']),
        'popsize': trial.suggest_int('popsize', 50, 500, step=10),
        'fit_type' : trial.suggest_categorical(name='fit_type', choices=['F1','F1W']),
        'max_len' : trial.suggest_int('max_len', 10, 50, step=5),
        'rt_cross' : rt_cross,
        'rt_mut' : 1-rt_cross
    }
    return params


# example dataset - digen8_4426

results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage=None, local_cache_dir='.')
results[datasets]['classifier']=results[datasets]['classifier'].get_params()
pickle.dump(results, open(datasets+'-ellyn.pkl','wb'))
