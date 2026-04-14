/*
  Bicycle.chpl
  ------------
  Quantile regression driver: trains a Chapel GBM with Pinball loss on
  the UCI Bike Sharing Dataset (hourly counts) and reports:

    - Pinball loss at tau = 0.1, 0.9 on train and test sets
    - 80% interval coverage  (fraction of test rows where q10 <= y <= q90)
    - Mean interval width    (q90 - q10, averaged over test set)

  Workflow:
    1. python save_bicycle.py              # create the CSV
    2. make Bicycle                        # compile
    3. ./build/Bicycle                     # run (Chapel)
    4. python lightgbm_bicycle.py          # run (LightGBM baseline)

  Config constants (override on command line, e.g. --nTrees=200):
    dataFile   path to the CSV           [data/bicycle.csv]
    nTrees     number of boosting rounds [50]
    numLeaves  max leaves per tree       [16]
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

config const dataFile        : string = "data/bicycle.csv";
config const nTrees          : int    = 50;
config const numLeaves       : int    = 16;
config const eta             : real   = 0.1;
config const lambda          : real   = 1.0;
config const trainFrac       : real   = 0.8;
config const colsampleByTree : real   = 1.0;
config const seed            : int    = 1055742;

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
  writeln("=== Bicycle Quantile Regression — Chapel GBM ===");
  writeln("Locales: ", numLocales);
  writeln();

  // ---- Load -------------------------------------------------------
  const all = readCSV(dataFile);
  writeln("Samples: ", all.numSamples, "  Features: ", all.numFeatures);

  // ---- Split ------------------------------------------------------
  var (train, test) = trainTestSplit(all, trainFrac);
  writeln("Train: ", train.numSamples, "  Test: ", test.numSamples);
  writeln();

  // ---- Bin once; reuse cuts for both quantile models --------------
  const cuts = computeBins(train);
  applyBins(test, cuts);

  // ---- Train one model per quantile -------------------------------
  // Each Pinball instance carries its own tau; boost() calls
  // obj.initF(train) which sets train.F to the tau-quantile of y
  // before training starts.
  var cfg = new BoosterConfig(
    nTrees          = nTrees,
    numLeaves       = numLeaves,
    eta             = eta,
    lambda          = lambda,
    colsampleByTree = colsampleByTree,
    seed            = seed
  );

  writeln("Pinball loss:");

  // tau = 0.1
  const ensemble10   = boost(train, new Pinball(tau=0.1), cfg);
  const trainPreds10 = predict(ensemble10, train);
  const testPreds10  = predict(ensemble10, test);
  const trainLoss10  = pinballLoss(trainPreds10, train.y, 0.1);
  const testLoss10   = pinballLoss(testPreds10,  test.y,  0.1);
  writeln("  tau=0.1  pinball (train=", trainLoss10, "  test=", testLoss10, ")");

  // tau = 0.9
  const ensemble90   = boost(train, new Pinball(tau=0.9), cfg);
  const trainPreds90 = predict(ensemble90, train);
  const testPreds90  = predict(ensemble90, test);
  const trainLoss90  = pinballLoss(trainPreds90, train.y, 0.9);
  const testLoss90   = pinballLoss(testPreds90,  test.y,  0.9);
  writeln("  tau=0.9  pinball (train=", trainLoss90, "  test=", testLoss90, ")");

  writeln();

  // ---- 80% interval metrics on the test set -----------------------
  var covered    : int  = 0;
  var totalWidth : real = 0.0;
  forall i in test.rowDom with (+ reduce covered, + reduce totalWidth) {
    const lo = testPreds10[i];
    const hi = testPreds90[i];
    if test.y[i] >= lo && test.y[i] <= hi then covered += 1;
    totalWidth += hi - lo;
  }

  const coverage  = covered: real / test.numSamples: real * 100.0;
  const meanWidth = totalWidth / test.numSamples: real;

  writeln("80% prediction interval (q10–q90):");
  writeln("  coverage:   ", coverage, "%  (target 80%)");
  writeln("  mean width: ", meanWidth, " counts/hour");
}
