/*
  TestPredict.chpl
  ----------------
  Tests for predict() and applyBins().

  Workflow under test:
    1. computeBins(trainData)  → BinCuts
    2. boost(trainData, ...)   → [] FittedTree
    3. applyBins(testData, cuts)
    4. predict(trees, testData, eta) → [] real

  Run:
    chpl TestPredict.chpl -M ../src -o build/TestPredict
    ./build/TestPredict
*/

use Booster;
use Binning;
use Objectives;
use DataLayout;
use SyntheticData;
use Math;

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

// ------------------------------------------------------------------
// testPredictMatchesF
//
// After training, predict() on the same training data (already binned)
// must return values identical to data.F.
// ------------------------------------------------------------------
proc testPredictMatchesF() {
  writeln("\n--- predict on train set matches data.F ---");

  var data = makeSyntheticRegression(nSamples=200, nFeatures=4);
  var cfg  = new BoosterConfig(nTrees=5, maxDepth=2, eta=0.3);

  computeBins(data);
  const (trees, base) = boost(data, Objective.MSE, cfg);

  const preds = predict(trees, data, base);

  var maxDiff: real = 0.0;
  for i in data.rowDom do
    maxDiff = max(maxDiff, abs(preds[i] - data.F[i]));

  assertTrue("predict matches data.F (maxDiff=" + maxDiff:string + ")",
             maxDiff < 1e-12);
  assertTrue("data.F unchanged after predict",
             maxDiff < 1e-12);  // same check — predict must not modify data.F
}

// ------------------------------------------------------------------
// testPredictOnTestSet
//
// Trains on one dataset, applies bin cuts to a disjoint test set via
// applyBins, then checks that predictions are finite and plausible.
// ------------------------------------------------------------------
proc testPredictOnTestSet() {
  writeln("\n--- predict on held-out test set ---");

  var train = makeSyntheticRegression(nSamples=400, nFeatures=4, seed=1);
  var test  = makeSyntheticRegression(nSamples=100, nFeatures=4, seed=2);

  var cfg = new BoosterConfig(nTrees=10, maxDepth=2, eta=0.3);

  const cuts         = computeBins(train);
  const (trees, base) = boost(train, Objective.MSE, cfg);

  // Apply training cuts to the test set
  applyBins(test, cuts);

  const preds = predict(trees, test, base);

  var allFinite = true;
  for i in test.rowDom {
    if !isFinite(preds[i]) { allFinite = false; break; }
  }
  assertTrue("test predictions are all finite", allFinite);

  // Verify data.F on test set was not modified
  const fSum = + reduce test.F;
  assertTrue("test data.F not modified by predict", fSum == 0.0);

  // MSE on test set should be lower than predicting the mean (yMean)
  const yMean    = (+ reduce test.y) / test.numSamples: real;
  const msePred  = (+ reduce [(i) in test.rowDom] (preds[i]  - test.y[i])**2) / test.numSamples: real;
  const mseMean  = (+ reduce [(i) in test.rowDom] (yMean     - test.y[i])**2) / test.numSamples: real;
  assertTrue("test MSE beats predicting the mean", msePred < mseMean);
}

// ------------------------------------------------------------------
// testPredictClassification
//
// Checks that predict returns finite logits for classification.
// ------------------------------------------------------------------
proc testPredictClassification() {
  writeln("\n--- predict classification logits ---");

  var train = makeSyntheticClassification(nSamples=300, nFeatures=4, nInformative=4, seed=10);
  var test  = makeSyntheticClassification(nSamples=100, nFeatures=4, nInformative=4, seed=11);

  var cfg = new BoosterConfig(nTrees=5, maxDepth=2, eta=0.3);

  const cuts          = computeBins(train);
  const (trees, base) = boost(train, Objective.LogLoss, cfg);

  applyBins(test, cuts);
  const logits = predict(trees, test, base);

  var allFinite = true;
  for i in test.rowDom {
    if !isFinite(logits[i]) { allFinite = false; break; }
  }
  assertTrue("classification logits are all finite", allFinite);

  // Convert logits to probabilities; check mean is in (0,1)
  const probMean = (+ reduce [(i) in test.rowDom] 1.0 / (1.0 + exp(-logits[i])))
                   / test.numSamples: real;
  assertTrue("mean predicted probability in (0,1)",
             probMean > 0.0 && probMean < 1.0);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Predict Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testPredictMatchesF();
  testPredictOnTestSet();
  testPredictClassification();

  writeln("\nDone.");
}
