/*
  DataLayout.chpl
  ---------------
  Defines the core distributed arrays used throughout the GBM prototype
  and utilities for loading / generating synthetic data.

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
  use Random;
  use Math;

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
  // makeSyntheticClassification
  //
  // Generates a linearly separable binary classification dataset.
  // Each feature is drawn from N(0,1); the label is determined by
  // whether a random linear combination of the first `nInformative`
  // features exceeds zero (with a small noise term).
  //
  // Returns a GBMData record populated with X and y.  F, grad, hess
  // are zeroed and ready for the first boosting round.
  // ------------------------------------------------------------------
  proc makeSyntheticClassification(
      nSamples     : int,
      nFeatures    : int,
      nInformative : int = 5,
      seed         : int = 42
  ): GBMData {

    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    var rng = new randomStream(real, seed=seed);

    // Fill X with N(0,1) draws
    // (randomStream.fill does a parallel fill respecting the distribution)
    rng.fill(data.X);

    // Build a fixed weight vector for the informative features
    // (deterministic from seed — same result every run)
    var weights: [0..#nInformative] real;
    var wRng = new randomStream(real, seed=seed + 1);
    wRng.fill(weights);

    // Compute labels: y = 1 if dot(X[i, 0..nInformative-1], weights) + noise > 0
    var noise: [data.rowDom] real;
    var noiseRng = new randomStream(real, seed=seed + 2);
    noiseRng.fill(noise);
    forall i in data.rowDom with (ref data) {
      var score: real = 0.0;
      for f in 0..#nInformative do
        score += data.X[i, f] * weights[f];
      score += noise[i] * 0.1;
      data.y[i] = if score > 0.0 then 1.0 else 0.0;
    }

    return data;
  }

  // ------------------------------------------------------------------
  // makeSyntheticRegression
  //
  // Generates a regression dataset with a nonlinear target:
  //   y = sin(x0) + 0.5*x1^2 - x2 + noise
  //
  // Useful for testing both MSE and Pinball objectives.
  // ------------------------------------------------------------------
  proc makeSyntheticRegression(
      nSamples  : int,
      nFeatures : int,
      seed      : int = 42
  ): GBMData {

    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    var rng = new randomStream(real, seed=seed);
    rng.fill(data.X);   // features ~ N(0,1)

    var noise: [data.rowDom] real;
    var noiseRng = new randomStream(real, seed=seed + 99);
    noiseRng.fill(noise);
    forall i in data.rowDom with (ref data) {
      var signal: real = 0.0;
      if nFeatures > 0 then signal += sin(data.X[i, 0]);
      if nFeatures > 1 then signal += 0.5 * data.X[i, 1] ** 2;
      if nFeatures > 2 then signal -= data.X[i, 2];
      data.y[i] = signal + noise[i] * 0.1;
    }

    return data;
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
