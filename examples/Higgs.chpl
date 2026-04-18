/*
  Higgs.chpl
  ----------
  Binary classification driver: trains a Chapel GBM with LogLoss on the
  UCI HIGGS dataset (signal vs. background, up to 11 M samples, 28 features)
  and reports log-loss and accuracy on train and test sets.

  Workflow:
    1. python save_higgs.py           # download and create HDF5 (~1 GB)
    2. make Higgs                     # compile
    3. ./build/Higgs                  # run (Chapel)
    4. python lightgbm_higgs.py       # run (LightGBM baseline)

  Config constants (override on command line, e.g. --nTrees=200):
    dataFile   path to HDF5 file         [data/higgs.h5]
    nTrees     number of boosting rounds [50]
    numLeaves  max leaves per tree       [31]
    eta        learning rate             [0.1]
    lambda     L2 regularisation        [1.0]
    trainFrac  fraction used for train  [0.909]  (~10 M train / 1 M test)
    minHess    minimum hessian per leaf  [1.0]
*/

use HDF5Reader;
use DataLayout;
use Binning;
use Booster;
use Objectives;
use Logger;
use Math;
use Time;

config const dataFile        : string = "data/higgs.h5";
config const nTrees          : int    = 50;
config const numLeaves       : int    = 31;
config const eta             : real   = 0.1;
config const lambda          : real   = 1.0;
config const trainFrac       : real   = 0.909;
config const minHess         : real   = 1.0;
config const colsampleByTree : real   = 1.0;
config const seed            : int    = 1055742;

proc accuracy(preds: [] real, y: [] real(32)): real {
  var correct = 0;
  forall i in preds.domain with (+ reduce correct) {
    const yhat = if preds[i] >= 0.0 then 1.0 else 0.0;
    if yhat == y[i] then correct += 1;
  }
  return correct: real / preds.size: real * 100.0;
}

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

proc main() throws {
  writeln("=== HIGGS Classification — Chapel GBM ===");
  writeln("Locales: ", numLocales);
  writeln();

  var loadTimer: stopwatch;
  loadTimer.start();
  const all = readHDF5(dataFile);
  loadTimer.stop();
  writeln("Samples: ", all.numSamples, "  Features: ", all.numFeatures,
          "  (load: ", loadTimer.elapsed():string, "s)");

  var (train, test) = trainTestSplit(all, trainFrac);
  writeln("Train: ", train.numSamples, "  Test: ", test.numSamples);
  writeln();

  var cfg = new BoosterConfig(
    nTrees          = nTrees,
    numLeaves       = numLeaves,
    eta             = eta,
    lambda          = lambda,
    colsampleByTree = colsampleByTree,
    seed            = seed
  );

  const cuts = computeBins(train);
  var trainTimer: stopwatch;
  trainTimer.start();
  const ensemble = boost(train, new LogLoss(minHess=minHess), cfg);
  trainTimer.stop();
  writeln("nTrees: ", nTrees, "  numLeaves: ", numLeaves,
          "  (elapsed: ", trainTimer.elapsed():string, "s)");
  writeln();

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
