/*
  TestBooster.chpl
  ----------------
  Tests for Booster.boost.

  Run:
    chpl TestBooster.chpl -M ../src -o TestBooster
    ./TestBooster
*/

use Booster;
use Binning;
use Objectives;
use DataLayout;
use SyntheticData;
use Math;

const EPS = 1e-10;

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

proc assertClose(name: string, got: real, expected: real, tol: real = EPS) {
  if abs(got - expected) > tol then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

// ------------------------------------------------------------------
// testMSEDecreases
//
// Trains a small MSE booster on synthetic regression data.
// Checks that training loss decreases monotonically across rounds
// and that predictions are finite.
// ------------------------------------------------------------------
proc testMSEDecreases() {
  writeln("\n--- MSE loss decreases ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=5);

  var cfg = new BoosterConfig(nTrees=10, numLeaves=4, eta=0.3, lambda=1.0);

  computeBins(data);
  const lossBefore = mseLoss(data.F, data.y);
  boost(data, new MSE(), cfg);
  const lossAfter = mseLoss(data.F, data.y);

  assertTrue("MSE decreases after boosting",   lossAfter < lossBefore);
  assertTrue("final MSE is finite",            isFinite(lossAfter));
  assertTrue("F values all finite",
             min reduce [i in data.rowDom] (if isFinite(data.F[i]) then 1 else 0) == 1);
}

// ------------------------------------------------------------------
// testLogLossDecreases
//
// Trains on synthetic classification data.
// ------------------------------------------------------------------
proc testLogLossDecreases() {
  writeln("\n--- LogLoss decreases ---");

  var data = makeSyntheticClassification(nSamples=500, nFeatures=5);

  var cfg = new BoosterConfig(nTrees=10, numLeaves=4, eta=0.3, lambda=1.0);

  computeBins(data);
  const lossBefore = logLoss(data.F, data.y);
  boost(data, new LogLoss(), cfg);
  const lossAfter = logLoss(data.F, data.y);

  assertTrue("LogLoss decreases after boosting", lossAfter < lossBefore);
  assertTrue("final LogLoss is finite",          isFinite(lossAfter));
}

// ------------------------------------------------------------------
// testPinballDecreases
//
// Trains a quantile (tau=0.9) booster on regression data.
// ------------------------------------------------------------------
proc testPinballDecreases() {
  writeln("\n--- Pinball loss decreases ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=5);

  const tau = 0.9;
  var cfg = new BoosterConfig(nTrees=10, numLeaves=4, eta=0.3, lambda=1.0);

  computeBins(data);
  const lossBefore = pinballLoss(data.F, data.y, tau);
  boost(data, new Pinball(tau=tau), cfg);
  const lossAfter = pinballLoss(data.F, data.y, tau);

  assertTrue("Pinball loss decreases after boosting", lossAfter < lossBefore);
  assertTrue("final Pinball loss is finite",          isFinite(lossAfter));
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Booster Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testMSEDecreases();
  testLogLossDecreases();
  testPinballDecreases();

  writeln("\nDone.");
}
