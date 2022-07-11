#! /usr/bin/env python3

from digen import Benchmark
import argparse
#from feat import feat
from sklearn.metrics import accuracy_score 
import json
import pickle

# Load a package with DIGEN benchmark
benchmark = Benchmark(n_trials=200, timeout=100000)

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



from feat import FeatClassifier

est = FeatClassifier(max_depth=6,
              max_dim = 10,
           obj='fitness,size',
           sel='lexicase',
           max_stall = 50,
           stagewise_xo = True,
           scorer='log',
           verbosity=0,
           shuffle=True,
           ml='L1_LR',
           fb=1,
           backprop=True,
           iters=1,
           normalize=False)

 
def params_myParamScope(trial):

    params = {
        'gens': trial.suggest_int('gens', 10, 200, step=10),
        'pop_size': trial.suggest_int('pop_size', 50, 500, step=10),
        'max_depth' : trial.suggest_int('max_depth', 2, 10)
    }
    return params


# example dataset - digen8_4426

#results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
#                             storage=None, local_cache_dir='.')
results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage='sqlite:///'+datasets+'-feat.db', local_cache_dir='.')

results[datasets]['classifier']=results[datasets]['classifier'].get_params()
pickle.dump(results, open(datasets+'-feat.pkl','wb'))
