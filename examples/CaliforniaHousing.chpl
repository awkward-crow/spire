/*
  CaliforniaHousing.chpl
  ----------------------
  Comparison driver: trains a Chapel GBM on the California Housing
  dataset and reports RMSE on train and test sets.

  Workflow:
    1. python save_california_housing.py      # create the CSV
    2. make CaliforniaHousing                 # compile
    3. ./build/CaliforniaHousing              # run (Chapel)
    4. python lightgbm_baseline.py            # run (LightGBM)

  Config constants (override on command line, e.g. --nTrees=200):
    dataFile   path to the CSV           [data/california_housing.csv]
    nTrees     number of boosting rounds [100]
    maxDepth   maximum tree depth        [6]
    eta        learning rate             [0.1]
    lambda     L2 regularisation        [1.0]
    trainFrac  fraction used for train  [0.8]
    seed       RNG seed for binning     [42]
*/

use CSVReader;
use DataLayout;
use Binning;
use Booster;
use Objectives;
use Logger;
use Math;

config const dataFile  : string = "data/california_housing.csv";
config const nTrees    : int    = 100;
config const maxDepth  : int    = 6;
config const eta       : real   = 0.1;
config const lambda    : real   = 1.0;
config const trainFrac : real   = 0.8;

// ------------------------------------------------------------------
// rmse
// ------------------------------------------------------------------
proc rmse(preds: [] real, y: [] real): real {
  return sqrt((+ reduce [(i) in preds.domain] (preds[i] - y[i])**2)
              / preds.size: real);
}

// ------------------------------------------------------------------
// trainTestSplit
//
// Copies the first trainFrac rows into trainData and the remainder
// into testData.  Sequential split matches the Python scripts.
// ------------------------------------------------------------------
proc trainTestSplit(data: GBMData, frac: real): (GBMData, GBMData) {
  const nTrain_ = max(1, (data.numSamples * frac): int);
  const nTest_  = data.numSamples - nTrain_;

  var train = new GBMData(numSamples=nTrain_, numFeatures=data.numFeatures);
  var test  = new GBMData(numSamples=nTest_,  numFeatures=data.numFeatures);

  forall i in 0..#nTrain_ with (ref train) {
    train.y[i] = data.y[i];
    for f in 0..#data.numFeatures do train.X[i, f] = data.X[i, f];
  }
  forall i in 0..#nTest_ with (ref test) {
    test.y[i] = data.y[nTrain_ + i];
    for f in 0..#data.numFeatures do test.X[i, f] = data.X[nTrain_ + i, f];
  }

  return (train, test);
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
proc main() throws {
  writeln("=== California Housing Regression — Chapel GBM ===");
  writeln("Locales: ", numLocales);
  writeln();

  // ---- Load -------------------------------------------------------
  const all = readCSV(dataFile);
  writeln("Samples: ", all.numSamples, "  Features: ", all.numFeatures);

  // ---- Split ------------------------------------------------------
  var (train, test) = trainTestSplit(all, trainFrac);
  writeln("Train: ", train.numSamples, "  Test: ", test.numSamples);
  writeln();

  // ---- Train ------------------------------------------------------
  var cfg = new BoosterConfig(
    nTrees   = nTrees,
    maxDepth = maxDepth,
    eta      = eta,
    lambda   = lambda
  );

  const cuts     = computeBins(train);
  const ensemble = boost(train, new MSE(), cfg);

  // ---- Evaluate ---------------------------------------------------
  const trainPreds = predict(ensemble, train);
  const trainRMSE  = rmse(trainPreds, train.y);

  applyBins(test, cuts);
  const testPreds = predict(ensemble, test);
  const testRMSE  = rmse(testPreds, test.y);

  writeln("  RMSE (train): ", trainRMSE);
  writeln("  RMSE (test):  ", testRMSE);
}
