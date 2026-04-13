/*
  TestSplits.chpl
  ---------------
  Tests for Splits.leafScore, Splits.leafValue, and Splits.findBestSplits.

  Run:
    chpl TestSplits.chpl -M ../src -o TestSplits
    ./TestSplits
*/

use Splits;
use Histogram;
use Binning;   // MAX_BINS

const EPS = 1e-10;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
proc assertClose(name: string, got: real, expected: real, tol: real = EPS) {
  if abs(got - expected) > tol then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

proc assertEq(name: string, got: int, expected: int) {
  if got != expected then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

// ------------------------------------------------------------------
// testLeafFormulas
//
// Checks leafScore and leafValue against hand-computed values.
// ------------------------------------------------------------------
proc testLeafFormulas() {
  writeln("\n--- leaf formulas ---");

  // leafScore(G, H, lambda) = G² / (H + lambda)
  assertClose("leafScore G=2 H=4 lam=0",  leafScore(2.0, 4.0, 0.0),  1.0);
  assertClose("leafScore G=3 H=2 lam=1",  leafScore(3.0, 2.0, 1.0),  3.0);
  assertClose("leafScore G=0 H=5 lam=1",  leafScore(0.0, 5.0, 1.0),  0.0);
  assertClose("leafScore G=-4 H=3 lam=1", leafScore(-4.0, 3.0, 1.0), 4.0);

  // leafValue(G, H, lambda) = -G / (H + lambda)
  assertClose("leafValue G=2 H=4 lam=0",  leafValue(2.0, 4.0, 0.0), -0.5);
  assertClose("leafValue G=-3 H=2 lam=1", leafValue(-3.0, 2.0, 1.0), 1.0);
  assertClose("leafValue G=0 H=5 lam=1",  leafValue(0.0, 5.0, 1.0),  0.0);
}

// ------------------------------------------------------------------
// testFindBestSplits
//
// Hand-crafted 2-feature histogram with 3 active bins each.
// All hessians = 1.0, lambda = 0.0.
//
// Feature 0 bins: grad = [4, -2, -2]   G_P=0, H_P=3
//   after bin 0: G_L=4,H_L=1  G_R=-4,H_R=2  gain = 16/1 + 16/2 = 24.0
//   after bin 1: G_L=2,H_L=2  G_R=-2,H_R=1  gain =  4/2 +  4/1 =  6.0
//
// Feature 1 bins: grad = [3, -1, -2]   G_P=0, H_P=3
//   after bin 0: G_L=3,H_L=1  G_R=-3,H_R=2  gain =  9/1 +  9/2 = 13.5
//   after bin 1: G_L=2,H_L=2  G_R=-2,H_R=1  gain =  4/2 +  4/1 =  6.0
//
// Expected best: feature=0, bin=0, gain=24.0
// ------------------------------------------------------------------
proc testFindBestSplits() {
  writeln("\n--- findBestSplits ---");

  var hist = new HistogramData(maxNodes=1, nFeatures=2);

  // Feature 0  [f, bin, node]
  hist.grad[0, 0, 0] =  4.0;  hist.hess[0, 0, 0] = 1.0;
  hist.grad[0, 1, 0] = -2.0;  hist.hess[0, 1, 0] = 1.0;
  hist.grad[0, 2, 0] = -2.0;  hist.hess[0, 2, 0] = 1.0;

  // Feature 1  [f, bin, node]
  hist.grad[1, 0, 0] =  3.0;  hist.hess[1, 0, 0] = 1.0;
  hist.grad[1, 1, 0] = -1.0;  hist.hess[1, 1, 0] = 1.0;
  hist.grad[1, 2, 0] = -2.0;  hist.hess[1, 2, 0] = 1.0;

  const splits = findBestSplits(hist, lambda=0.0, minHess=0.5);

  assertTrue("node 0 valid",          splits[0].valid);
  assertEq  ("node 0 best feature",   splits[0].feature, 0);
  assertEq  ("node 0 best bin",       splits[0].bin,     0);
  assertClose("node 0 gain",          splits[0].gain,    24.0);
  assertClose("node 0 leftGrad",      splits[0].leftGrad, 4.0);
  assertClose("node 0 leftHess",      splits[0].leftHess, 1.0);
}

// ------------------------------------------------------------------
// testNoValidSplit
//
// All mass in bin 0 — after scanning every bin, the right child is
// always empty (H_R=0 < minHess), so no split is accepted.
// ------------------------------------------------------------------
proc testNoValidSplit() {
  writeln("\n--- no valid split ---");

  var hist = new HistogramData(maxNodes=1, nFeatures=1);
  hist.grad[0, 0, 0] = 5.0;
  hist.hess[0, 0, 0] = 5.0;

  const splits = findBestSplits(hist, lambda=1.0, minHess=1.0);
  assertTrue("valid=false when no profitable split", !splits[0].valid);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Splits Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testLeafFormulas();
  testFindBestSplits();
  testNoValidSplit();

  writeln("\nDone.");
}
