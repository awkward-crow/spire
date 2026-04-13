# examples

## virtual env

```sh
pyenv local 3.11.9
python --version
```
=>
    Python 3.11.9

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Bicycle 

The UCI Bike Sharing Dataset — hourly bike rental counts in Washington D.C. from   
  2011–2012, originally from Capital Bikeshare. Features are time/calendar variables 
  (hour, weekday, season, etc.) plus weather (temp, humidity, windspeed). Target is
  cnt (total rentals per hour, typically 0–977).

### download data

```sh
mkdir -p data
./save_bicycle.py
```
=>
    Fetching Bike Sharing Demand dataset from OpenML …
    Saved 17379 rows x 12 features to data/bicycle.csv (shuffled, seed=42)

### lightGBM benchmark

```sh
./lightgbm_bicycle.py
```
=>
    === Bicycle Quantile Regression — LightGBM ===
    Samples: 17379  Features: 12
    Train: 13903  Test: 3476

    Pinball loss:
      tau=0.1  pinball (train=10.4584  test=10.6694)
      tau=0.9  pinball (train=11.3606  test=11.2987)

    80% prediction interval (q10–q90):
      coverage:     82.3%  (target 80%)
      mean width:   186.3 counts/hour

### chapel (d)gbm

see `../refactor.md`


## California housing 

The California Housing dataset comes from the 1990 U.S. Census. Each row is a      
  census block group in California. Features are things like median income, house    
  age, average rooms, population, and lat/lon. Target is median house value (in $100k
   units).

### download data

```sh
mkdir -p data
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

      RMSE (train): 0.5167
      RMSE (test):  0.5361

### chapel (d)gbm

```sh
./build/CaliforniaHousing  --nTrees=50 --maxDepth=4
```
    === California Housing Regression — Chapel GBM ===
    Locales: 1

    Samples: 20640  Features: 8
    Train: 16512  Test: 4128

      RMSE (train): 0.521041
      RMSE (test):  0.537284


## Cover Type

Binary classification on the UCI Cover Type dataset — predicting forest cover type
from cartographic variables. Class 1 (Spruce/Fir) vs class 2 (Lodgepole Pine),
~495 k rows, 54 features (10 quantitative + 44 binary wilderness-area/soil-type
indicators). Objective: LogLoss.

### download data

```sh
mkdir -p data
python save_cover_type.py
```
=>
    Saved 495141 rows x 54 features to data/cover_type.csv (shuffled, seed=10331)
    Label distribution: 211840 Spruce/Fir (1), 283301 Lodgepole Pine (0)

### lightGBM benchmark

```sh
python lightgbm_cover_type.py
```
=>
    === Cover Type Classification — LightGBM ===
    Samples: 495141  Features: 54
    Train: 396112  Test: 99029

    Log-loss:
      train: 0.435960  test: 0.436165

    Accuracy:
      train: 80.0218%  test: 80.0644%

### chapel gbm

```sh
make CoverType
./build/CoverType
```
=>
    === Cover Type Classification — Chapel GBM ===
    Locales: 1

    Samples: 495141  Features: 54  (load: 9.1s)
    Train: 396112  Test: 99029

    Log-loss:
      train: 0.433554  test: 0.433736

    Accuracy:
      train: 80.1036%  test: 80.0947%

Chapel marginally outperforms LightGBM on both metrics at these settings
(nTrees=50, maxDepth=4, eta=0.1, lambda=1.0).  Wall time ~31 s on a single
locale (8 cores); 9 s of that is CSV loading.

### end
