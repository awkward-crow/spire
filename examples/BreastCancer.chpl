/*
  BreastCancer.chpl
  -----------------
  Binary classification driver: trains a Chapel GBM with LogLoss on the
  Wisconsin Breast Cancer dataset and reports log-loss and accuracy on
  train and test sets.

  Workflow:
    1. python save_breast_cancer.py      # create the CSV
    2. make BreastCancer                 # compile
    3. ./build/BreastCancer              # run (Chapel)
    4. python lightgbm_breast_cancer.py  # run (LightGBM baseline)

  Config constants (override on command line, e.g. --nTrees=200):
    dataFile   path to the CSV           [data/breast_cancer.csv]
    nTrees     number of boosting rounds [100]
    maxDepth   maximum tree depth        [4]
    eta        learning rate             [0.1]
    lambda     L2 regularisation        [1.0]
    trainFrac  fraction used for train  [0.8]
*/

use CSVReader;
use DataLayout;
use Binning;
use Booster;
use Objectives;
use Logger;
use Math;
use Time;

config const dataFile  : string = "data/breast_cancer.csv";
config const nTrees    : int    = 50;
config const maxDepth  : int    = 4;
config const eta       : real   = 0.1;
config const lambda    : real   = 1.0;
config const trainFrac : real   = 0.8;
config const minHess   : real   = 1e-6;   // minimum hessian sum per leaf

// ------------------------------------------------------------------
// accuracy
//
// Fraction of samples correctly classified.
// preds are raw logits; threshold at 0 (sigmoid(x) >= 0.5 iff x >= 0).
// ------------------------------------------------------------------
proc accuracy(preds: [] real, y: [] real): real {
  var correct = 0;
  forall i in preds.domain with (+ reduce correct) {
    const yhat = if preds[i] >= 0.0 then 1.0 else 0.0;
    if yhat == y[i] then correct += 1;
  }
  return correct: real / preds.size: real * 100.0;
}

// ------------------------------------------------------------------
// trainTestSplit
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
  writeln("=== Breast Cancer Classification — Chapel GBM ===");
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
  const ensemble = boost(train, new LogLoss(minHess=minHess), cfg);

  // ---- Evaluate ---------------------------------------------------
  const trainPreds = predict(ensemble, train);
  const trainLL    = logLoss(trainPreds, train.y);
  const trainAcc   = accuracy(trainPreds, train.y);

  applyBins(test, cuts);
  const testPreds = predict(ensemble, test);
  const testLL    = logLoss(testPreds, test.y);
  const testAcc   = accuracy(testPreds, test.y);

  writeln("Log-loss:");
  writeln("  train: ", trainLL, "  test: ", testLL);
  writeln();
  writeln("Accuracy:");
  writeln("  train: ", trainAcc, "%  test: ", testAcc, "%");
}
