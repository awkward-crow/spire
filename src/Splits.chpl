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
  use CTypes;    // c_array, c_ptr, c_ptrTo

  // Best-split gain kernel — separate C translation unit so LLVM vectorizes it.
  // Two passes: Chapel does the sequential prefix scan (pass 1), C does the
  // vectorizable gain computation + argmax (pass 2).
  require "splits_kernel.h", "splits_kernel.c";

  extern proc spire_findBestBin(
      prefixG  : c_ptr(real(32)),
      prefixH  : c_ptr(real(32)),
      nbins    : c_int,
      G_P      : real(32),
      H_P      : real(32),
      lambda   : real(32),
      minHess  : real(32),
      out_gain : c_ptr(real(32)),
      out_G_L  : c_ptr(real(32)),
      out_H_L  : c_ptr(real(32))
  ): c_int;

  // ------------------------------------------------------------------
  // SplitInfo — result of the best split found for one node.
  //
  // valid = false means no profitable split was found (leaf node).
  // bin   = the highest bin index assigned to the left child;
  //         samples with Xb[feature, i] <= bin go left.
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
      hist       : HistogramData,
      lambda     : real  = 1.0,
      minHess    : real  = 1.0,
      featSubset : [] int
  ): [] SplitInfo {

    var splits: [0..#hist.maxNodes] SplitInfo;

    // Anchor feature for node totals — any feature in the subset works
    // (all give the same G_P/H_P by the conservation invariant).
    const anchorFeat = featSubset[featSubset.domain.low];

    var prefixG: c_array(real(32), MAX_BINS);
    var prefixH: c_array(real(32), MAX_BINS);

    for node in 0..#hist.maxNodes {

      // Node totals — sum over bins for the anchor feature.
      // real(32) reduces; Chapel widens to real(64) when mixed with lambda.
      const G_P = + reduce hist.grad[anchorFeat, .., node];
      const H_P = + reduce hist.hess[anchorFeat, .., node];

      if H_P < minHess {
        splits[node].valid = false;
        continue;
      }

      const scoreP    = leafScore(G_P, H_P, lambda);
      var   bestGain  = 0.0;   // only accept positive gains

      for f in featSubset {
        // Pass 1 — sequential prefix scan.
        var gAcc: real(32) = 0.0: real(32);
        var hAcc: real(32) = 0.0: real(32);
        for b in 0..<MAX_BINS {
          gAcc += hist.grad[f, b, node];
          hAcc += hist.hess[f, b, node];
          prefixG[b] = gAcc;
          prefixH[b] = hAcc;
        }

        // Pass 2 — vectorized gain evaluation + argmax via C kernel.
        var out_gain, out_G_L, out_H_L: real(32);
        const bestBin = spire_findBestBin(
            c_ptrTo(prefixG[0]), c_ptrTo(prefixH[0]),
            MAX_BINS: c_int,
            G_P: real(32), H_P: real(32),
            lambda: real(32), minHess: real(32),
            c_ptrTo(out_gain), c_ptrTo(out_G_L), c_ptrTo(out_H_L)
        );
        if bestBin >= 0 && (out_gain: real) > bestGain {
          bestGain              = out_gain: real;
          splits[node].feature  = f;
          splits[node].bin      = bestBin: int;
          splits[node].gain     = out_gain: real;
          splits[node].leftGrad = out_G_L;
          splits[node].leftHess = out_H_L;
          splits[node].valid    = true;
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

  // ------------------------------------------------------------------
  // findBestSplitsNodes
  //
  // Like findBestSplits but only evaluates the nodes listed in nodeList.
  // Used in the subtraction trick: after each split only the two new
  // children need new split candidates; all other active leaves keep
  // their cached SplitInfo from the previous round.
  // ------------------------------------------------------------------
  proc findBestSplitsNodes(
      hist       : HistogramData,
      lambda     : real,
      minHess    : real,
      featSubset : [] int,
      nodeList   : [] int
  ): [] SplitInfo {

    var splits: [0..#hist.maxNodes] SplitInfo;
    const anchorFeat = featSubset[featSubset.domain.low];

    // Temp arrays reused across all (node, f) pairs — 2 × 255 × 4 B ≈ 2 KB, fits in L1.
    var prefixG: c_array(real(32), MAX_BINS);
    var prefixH: c_array(real(32), MAX_BINS);

    for node in nodeList {
      const G_P = + reduce hist.grad[anchorFeat, .., node];  // real(32)
      const H_P = + reduce hist.hess[anchorFeat, .., node];

      if H_P < minHess {
        splits[node].valid = false;
        continue;
      }

      const scoreP   = leafScore(G_P, H_P, lambda);
      var   bestGain = 0.0;

      for f in featSubset {
        // Pass 1 — sequential prefix scan (loop-carried dep on gAcc/hAcc).
        // Writes contiguous stride-1 temps so pass 2 reads are vectorizable.
        var gAcc: real(32) = 0.0: real(32);
        var hAcc: real(32) = 0.0: real(32);
        for b in 0..<MAX_BINS {
          gAcc += hist.grad[f, b, node];
          hAcc += hist.hess[f, b, node];
          prefixG[b] = gAcc;
          prefixH[b] = hAcc;
        }

        // Pass 2 — vectorized gain evaluation + argmax via C kernel.
        var out_gain, out_G_L, out_H_L: real(32);
        const bestBin = spire_findBestBin(
            c_ptrTo(prefixG[0]), c_ptrTo(prefixH[0]),
            MAX_BINS: c_int,
            G_P: real(32), H_P: real(32),
            lambda: real(32), minHess: real(32),
            c_ptrTo(out_gain), c_ptrTo(out_G_L), c_ptrTo(out_H_L)
        );
        if bestBin >= 0 && (out_gain: real) > bestGain {
          bestGain              = out_gain: real;
          splits[node].feature  = f;
          splits[node].bin      = bestBin: int;
          splits[node].gain     = out_gain: real;
          splits[node].leftGrad = out_G_L;
          splits[node].leftHess = out_H_L;
          splits[node].valid    = true;
        }
      }
    }

    return splits;
  }

  // ------------------------------------------------------------------
  // Backward-compatible overload — use all features (no subsampling).
  // ------------------------------------------------------------------
  proc findBestSplits(hist: HistogramData, lambda: real = 1.0, minHess: real = 1.0): [] SplitInfo {
    const allFeats: [0..#hist.nFeatures] int = [i in 0..#hist.nFeatures] i;
    return findBestSplits(hist, lambda, minHess, allFeats);
  }

} // module Splits
