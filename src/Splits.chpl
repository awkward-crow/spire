/*
  Splits.chpl
  -----------
  Split finding: given a completed histogram, find the best split
  for each active node.

  Algorithm: left-to-right prefix scan over bins for each (node, feature).
  Pure local arithmetic on the (non-distributed) HistogramData — call on
  locale 0.

  Gain formula (XGBoost/LightGBM convention):
    score(G, H) = G² / (H + lambda)
    gain        = score(G_L, H_L) + score(G_R, H_R) - score(G_P, H_P)

  A split is accepted only if gain > 0 and both children have
  hess_sum >= minHess.

  The inner bin scan (255 iterations) is the natural target for AVX2
  horizontal prefix sums — deferred until profiling shows it is a
  bottleneck (see notes.md).
*/

module Splits {

  use Histogram;
  use Binning;   // MAX_BINS

  // ------------------------------------------------------------------
  // SplitInfo — result of the best split found for one node.
  //
  // valid = false means no profitable split was found (leaf node).
  // bin   = the highest bin index assigned to the left child;
  //         samples with Xb[i, feature] <= bin go left.
  // ------------------------------------------------------------------
  record SplitInfo {
    var feature   : int;
    var bin       : int;
    var gain      : real;
    var leftGrad  : real;
    var leftHess  : real;
    var valid     : bool;   // false → treat node as a leaf
  }

  // ------------------------------------------------------------------
  // findBestSplits
  //
  // Returns one SplitInfo per node.  Run on locale 0.
  //
  // lambda  — L2 regularisation term (shrinks leaf values, avoids
  //           overfitting; typical range 0.1 – 10.0)
  // minHess — minimum hessian sum required in each child to accept a
  //           split (acts as a minimum-samples-per-leaf guard;
  //           typical value 1.0 for MSE/LogLoss)
  // ------------------------------------------------------------------
  proc findBestSplits(
      hist    : HistogramData,
      lambda  : real = 1.0,
      minHess : real = 1.0
  ): [] SplitInfo {

    var splits: [0..#hist.maxNodes] SplitInfo;

    for node in 0..#hist.maxNodes {

      // Node totals — sum over bins for feature 0
      // (all features give the same total by conservation invariant)
      const G_P = + reduce hist.grad[node, 0, ..];
      const H_P = + reduce hist.hess[node, 0, ..];

      if H_P < minHess {
        splits[node].valid = false;
        continue;
      }

      const scoreP    = leafScore(G_P, H_P, lambda);
      var   bestGain  = 0.0;   // only accept positive gains

      for f in 0..#hist.nFeatures {
        var G_L = 0.0;
        var H_L = 0.0;

        for b in 0..<MAX_BINS {
          G_L += hist.grad[node, f, b];
          H_L += hist.hess[node, f, b];

          const G_R = G_P - G_L;
          const H_R = H_P - H_L;

          if H_L < minHess || H_R < minHess then continue;

          const gain = leafScore(G_L, H_L, lambda)
                     + leafScore(G_R, H_R, lambda)
                     - scoreP;

          if gain > bestGain {
            bestGain              = gain;
            splits[node].feature  = f;
            splits[node].bin      = b;
            splits[node].gain     = gain;
            splits[node].leftGrad = G_L;
            splits[node].leftHess = H_L;
            splits[node].valid    = true;
          }
        }
      }
    }

    return splits;
  }

  // ------------------------------------------------------------------
  // leafScore — the regularised score for a node/leaf.
  // ------------------------------------------------------------------
  inline proc leafScore(G: real, H: real, lambda: real): real {
    return G * G / (H + lambda);
  }

  // ------------------------------------------------------------------
  // leafValue — optimal prediction for a leaf given its grad/hess sums.
  //   value = -G / (H + lambda)
  // Called by Tree.chpl when assigning leaf outputs.
  // ------------------------------------------------------------------
  inline proc leafValue(G: real, H: real, lambda: real): real {
    return -G / (H + lambda);
  }

} // module Splits
