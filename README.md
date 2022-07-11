# gpcbench - GP-based classifiers benchmark powered by DIGEN

gpcbench is an effort to benchmark the performance of GP-based classifiers on more challenging classification tasks.



# Benchmarked methods

Currently, gpcbench features the following GP-based classifiers:

- M3GP: https://github.com/jespb/Python-M3GP/tree/master/m3gp
- M4GP: https://github.com/cavalab/ellyn
- feat: https://github.com/cavalab/feat
- gplearn: https://github.com/trevorstephens/gplearn


# Baseline ML methods

gpcbench includes the following ML baseline models:

- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors
- LightGBM
- Logistic Regression
- Random Forest
- SVC
- XGBoost


# Benchmark suite - DIGEN

GP-based classifiers were benchmarked against baseline ML models using [DIGEN benchmark](https://github.com/epistasislab/digen).
The benchmark features 40 datasets created using genetic programming. 
