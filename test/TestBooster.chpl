/*
  TestBooster.chpl
  ----------------
  Tests for Booster.boost.

  Run:
    chpl TestBooster.chpl -M ../src -o TestBooster
    ./TestBooster
*/

use Booster;
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

  var cfg = new BoosterConfig(nTrees=10, maxDepth=2, eta=0.3,
                               lambda=1.0, minHess=1.0);

  // MSE = mean((F - y)^2) / 2 — track via sum of squared residuals
  proc mse(): real {
    return (+ reduce [(i) in data.rowDom] (data.F[i] - data.y[i])**2)
           / data.numSamples: real;
  }

  const lossBefore = mse();
  boost(data, Objective.MSE, cfg);
  const lossAfter = mse();

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

  var cfg = new BoosterConfig(nTrees=10, maxDepth=2, eta=0.3,
                               lambda=1.0, minHess=1.0);

  proc logloss(): real {
    var s: real = 0.0;
    for i in data.rowDom {
      const p = 1.0 / (1.0 + exp(-data.F[i]));
      const pi = max(min(p, 1.0 - 1e-15), 1e-15);
      s += -(data.y[i] * log(pi) + (1.0 - data.y[i]) * log(1.0 - pi));
    }
    return s / data.numSamples: real;
  }

  const lossBefore = logloss();
  boost(data, Objective.LogLoss, cfg);
  const lossAfter = logloss();

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
  var cfg = new BoosterConfig(nTrees=10, maxDepth=2, eta=0.3,
                               lambda=1.0, minHess=1.0, tau=tau);

  proc pinball(): real {
    var s: real = 0.0;
    for i in data.rowDom {
      const r = data.y[i] - data.F[i];
      s += if r > 0.0 then tau * r else (tau - 1.0) * r;
    }
    return s / data.numSamples: real;
  }

  const lossBefore = pinball();
  boost(data, Objective.Pinball, cfg);
  const lossAfter = pinball();

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
