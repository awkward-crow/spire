/*
  TestHistogram.chpl
  ------------------
  Tests for Histogram.buildHistograms and Histogram.subtractHistograms.

  Run:
    chpl TestHistogram.chpl -M ../src -o TestHistogram
    ./TestHistogram
*/

use Histogram;
use Binning;
use DataLayout;
use SyntheticData;
use Objectives;
use Math;

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

// ------------------------------------------------------------------
// testConservation
//
// With all samples assigned to node 0, the sum of hist.grad[0, f, *]
// over all bins must equal the sum of data.grad.  Same for hess.
// ------------------------------------------------------------------
proc testConservation() {
  writeln("\n--- histogram conservation ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=4);
  computeGradients(Objective.MSE, data.F, data.y, data.grad, data.hess);
  computeBins(data);

  // All samples in node 0
  var nodeId: [data.rowDom] int = 0;

  var hist = new HistogramData(maxNodes=1, nFeatures=data.numFeatures);
  buildHistograms(data, nodeId, hist);

  const totalGrad = + reduce data.grad;
  const totalHess = + reduce data.hess;

  for f in 0..#data.numFeatures {
    const binGradSum = + reduce hist.grad[0, f, ..];
    const binHessSum = + reduce hist.hess[0, f, ..];
    assertClose("grad conservation f=" + f:string, binGradSum, totalGrad);
    assertClose("hess conservation f=" + f:string, binHessSum, totalHess);
  }
}

// ------------------------------------------------------------------
// testSubtraction
//
// Split samples into two nodes: even indices → node 0, odd → node 1.
// Build a 2-node histogram and a 1-node "parent" histogram (all samples
// in node 0).
//
// Verify: subtractHistograms(parent, node1_hist, large)
//         produces large == node0_hist for every (feature, bin).
// ------------------------------------------------------------------
proc testSubtraction() {
  writeln("\n--- subtraction trick ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=4);
  computeGradients(Objective.MSE, data.F, data.y, data.grad, data.hess);
  computeBins(data);

  // Split: even → node 0, odd → node 1
  var nodeId: [data.rowDom] int;
  forall i in data.rowDom do nodeId[i] = i % 2;

  // 2-node histogram
  var hist2 = new HistogramData(maxNodes=2, nFeatures=data.numFeatures);
  buildHistograms(data, nodeId, hist2);

  // Parent histogram: all samples in node 0
  var nodeIdAll: [data.rowDom] int = 0;
  var parent = new HistogramData(maxNodes=1, nFeatures=data.numFeatures);
  buildHistograms(data, nodeIdAll, parent);

  // Extract node-1 slice of hist2 as smallChild
  var smallChild = new HistogramData(maxNodes=1, nFeatures=data.numFeatures);
  smallChild.grad[0, .., ..] = hist2.grad[1, .., ..];
  smallChild.hess[0, .., ..] = hist2.hess[1, .., ..];

  // Derive large child via subtraction
  var large = new HistogramData(maxNodes=1, nFeatures=data.numFeatures);
  subtractHistograms(parent, smallChild, large);

  // large should match node-0 slice of hist2
  var ok = true;
  for f in 0..#data.numFeatures {
    for b in 0..#MAX_BINS {
      const diff = abs(large.grad[0, f, b] - hist2.grad[0, f, b]);
      if diff > EPS {
        writeln("FAIL  subtraction grad f=", f, " b=", b,
                " diff=", diff);
        ok = false;
      }
      const diffH = abs(large.hess[0, f, b] - hist2.hess[0, f, b]);
      if diffH > EPS {
        writeln("FAIL  subtraction hess f=", f, " b=", b,
                " diff=", diffH);
        ok = false;
      }
    }
  }
  if ok then writeln("PASS  subtraction trick (all bins, all features)");
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Histogram Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testConservation();
  testSubtraction();

  writeln("\nDone.");
}
