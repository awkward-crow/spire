/*
  DataLayout.chpl
  ---------------
  Defines the core distributed arrays used throughout the GBM prototype.

  Design decisions (documented here so they're easy to revisit):

  1. Row-partitioned X
     Each locale owns a contiguous block of rows (all features for
     those samples).  This matches Option A from the design notes and
     keeps grad/hess arrays local during histogram accumulation.

  2. 1-D arrays for targets, predictions, gradients
     Distributed identically to the row dimension of X so that
     element-wise updates (gradient computation, F += eta * tree) are
     purely local and require no cross-locale communication.

  3. Real labels for y
     Using real rather than int so the same array works for both
     classification (0.0/1.0) and regression targets without casting.

  4. Single-locale fallback
     When numLocales == 1 the blockDist degenerates to a simple local
     array — no code changes needed between single and multi-locale runs.
*/

module DataLayout {

  use BlockDist;

  // ------------------------------------------------------------------
  // GBMData record
  // Bundles all arrays that travel together through training.
  // ------------------------------------------------------------------
  record GBMData {
    var numSamples  : int;
    var numFeatures : int;

    // Feature matrix — row-major block distribution over samples
    var XDom   : domain(2) dmapped new blockDist(boundingBox={0..#numSamples, 0..#numFeatures});
    var X      : [XDom] real;
    var Xb     : [XDom] uint(8);  // bin indices — populated by Binning.computeBins()

    // 1-D arrays — same block distribution over the sample dimension
    var rowDom : domain(1) dmapped new blockDist(boundingBox={0..#numSamples});
    var y      : [rowDom] real;   // targets (0.0/1.0 for classification, real for regression)
    var F      : [rowDom] real;   // current predictions (logits for classification)
    var grad   : [rowDom] real;   // gradient per sample
    var hess   : [rowDom] real;   // hessian per sample

    proc init(numSamples: int, numFeatures: int) {
      this.numSamples  = numSamples;
      this.numFeatures = numFeatures;
      this.XDom   = {0..#numSamples, 0..#numFeatures}
                      dmapped new blockDist(boundingBox={0..#numSamples, 0..#numFeatures});
      this.rowDom = {0..#numSamples}
                      dmapped new blockDist(boundingBox={0..#numSamples});
      // Array fields (X, y, F, grad, hess) default-initialize from their domains.
    }
  }

  // ------------------------------------------------------------------
  // printDataSummary — quick sanity check at startup
  // ------------------------------------------------------------------
  proc printDataSummary(data: GBMData) {
    writeln("=== Data Summary ===");
    writeln("  Samples  : ", data.numSamples);
    writeln("  Features : ", data.numFeatures);
    writeln("  Locales  : ", numLocales);

    // Label distribution (works for both classification and regression)
    const yMean = (+ reduce data.y) / data.numSamples : real;
    const yMin  = min reduce data.y;
    const yMax  = max reduce data.y;
    writeln("  y mean   : ", yMean);
    writeln("  y range  : [", yMin, ", ", yMax, "]");
    writeln("====================");
  }

} // module DataLayout
