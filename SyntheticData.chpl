/*
  SyntheticData.chpl
  ------------------
  Synthetic dataset generators for testing and benchmarking the GBM
  prototype.  All procedures return a GBMData record with X and y
  populated; F, grad, and hess are zeroed and ready for the first
  boosting round.
*/

module SyntheticData {

  use DataLayout;
  use Random;
  use Math;

  // ------------------------------------------------------------------
  // makeSyntheticClassification
  //
  // Generates a linearly separable binary classification dataset.
  // Each feature is drawn from N(0,1); the label is determined by
  // whether a random linear combination of the first `nInformative`
  // features exceeds zero (with a small noise term).
  // ------------------------------------------------------------------
  proc makeSyntheticClassification(
      nSamples     : int,
      nFeatures    : int,
      nInformative : int = 5,
      seed         : int = 42
  ): GBMData {

    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    var rng = new randomStream(real, seed=seed);
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
    rng.fill(data.X);

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

} // module SyntheticData
