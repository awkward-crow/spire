# examples

## California housing 

### virtual env

```sh
python --version
```
=>
    Python 3.11.9

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### download data

```sh
./save_california_housing.py
```
=>
    Saved 20640 rows x 8 features to data/california_housing.csv (shuffled, seed=10331)


### lightGBM benchmark

```sh
./lightgbm_baseline.py
```
=>
    === California Housing Regression — LightGBM ===
    Samples: 20640  Features: 8
    Train: 16512  Test: 4128

      RMSE (train): 0.4154
      RMSE (test):  0.4764


### end
