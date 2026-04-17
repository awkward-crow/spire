/*
  TestBinning.chpl
  ----------------
  Tests for Binning.findBin and Binning.computeBins.

  Run:
    chpl TestBinning.chpl -M ../src -o TestBinning
    ./TestBinning
*/

use Binning;
use DataLayout;
use SyntheticData;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
proc assertEq(name: string, got: int, expected: int) {
  if got != expected then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

// ------------------------------------------------------------------
// findBin — unit tests with a hand-crafted 3-cut / 4-bin example
//
// cuts = [1.0, 2.0, 3.0]
// bin 0: (-inf, 1.0)
// bin 1: [1.0,  2.0)
// bin 2: [2.0,  3.0)
// bin 3: [3.0,  +inf)
// ------------------------------------------------------------------
proc testFindBin() {
  writeln("\n--- findBin ---");

  const nCuts = 3;
  var cuts: [0..0, 0..#nCuts] real;
  cuts[0, 0] = 1.0;
  cuts[0, 1] = 2.0;
  cuts[0, 2] = 3.0;

  assertEq("below all cuts",   findBin(0.5,  cuts, 0, nCuts), 0);
  assertEq("on first cut",     findBin(1.0,  cuts, 0, nCuts), 1);
  assertEq("between 1 and 2",  findBin(1.5,  cuts, 0, nCuts), 1);
  assertEq("on second cut",    findBin(2.0,  cuts, 0, nCuts), 2);
  assertEq("between 2 and 3",  findBin(2.5,  cuts, 0, nCuts), 2);
  assertEq("on third cut",     findBin(3.0,  cuts, 0, nCuts), 3);
  assertEq("above all cuts",   findBin(99.0, cuts, 0, nCuts), 3);
}

// ------------------------------------------------------------------
// computeBins — integration test on a small synthetic dataset
//
// Checks:
//   1. All bin values in 0..MAX_BINS-1
//   2. Bin 0 is used (at least one value below first cut-point)
//   3. Multiple bins used per feature (not collapsed to one bin)
//   4. Monotonicity: X[i,f] < X[j,f]  =>  Xb[f,i] <= Xb[f,j]
//      (spot-checked on the first 20 samples)
// ------------------------------------------------------------------
proc testComputeBins() {
  writeln("\n--- computeBins ---");

  var data = makeSyntheticRegression(nSamples=1000, nFeatures=5);
  computeBins(data);

  // 1. Range check
  const maxBin = max reduce data.Xb;
  const minBin = min reduce data.Xb;
  assertTrue("max bin <= MAX_BINS-1", maxBin <= (MAX_BINS - 1): uint(8));
  assertTrue("min bin == 0",          minBin == 0: uint(8));

  // 2. Multiple bins used for each feature
  for f in 0..#data.numFeatures {
    var seen: [0..MAX_BINS-1] bool;
    for i in data.rowDom do seen[data.Xb[f, i]: int] = true;
    const nDistinct = + reduce seen;
    assertTrue("feature " + f:string + " uses multiple bins", nDistinct > 1);
  }

  // 3. Monotonicity spot-check: first 20 samples x first 20 samples
  const limit = min(20, data.numSamples);
  var monotone = true;
  for i in 0..#limit {
    for j in (i+1)..#limit {
      for f in 0..#data.numFeatures {
        if data.X[i, f] < data.X[j, f] && data.Xb[f, i] > data.Xb[f, j] {
          writeln("FAIL  monotonicity i=", i, " j=", j, " f=", f,
                  " X=", data.X[i,f], "/", data.X[j,f],
                  " Xb=", data.Xb[f,i], "/", data.Xb[f,j]);
          monotone = false;
        }
      }
    }
  }
  if monotone then writeln("PASS  bin monotonicity (spot check, first ", limit, " samples)");
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Binning Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testFindBin();
  testComputeBins();

  writeln("\nDone.");
}
