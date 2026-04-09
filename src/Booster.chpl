/*
  Booster.chpl
  ------------
  Gradient boosting training loop.  Ties together Binning, Objectives,
  Histogram, Splits, and Tree into a complete end-to-end GBM.

  One call to boost():
    1. computeBins        — once at startup
    2. For each tree:
       a. computeGradients
       b. Level-wise tree building (0..maxDepth):
            buildHistograms → findBestSplits → recordLevel → updateNodeAssign
          then finalizeLeaves at maxDepth
       c. applyTree        — F += eta * tree predictions
    3. Returns the fitted ensemble as [] FittedTree

  The histogram is allocated once per tree (sized to tree.nNodes =
  2^(maxDepth+1)-1) and reused across depths.  nodeId uses absolute heap
  indexing throughout (see Tree.chpl).
*/

module Booster {

  use DataLayout;
  use Objectives;
  use Binning;
  use Histogram;
  use Splits;
  use Tree;

  // ------------------------------------------------------------------
  // BoosterConfig
  // ------------------------------------------------------------------
  record BoosterConfig {
    var nTrees   : int  = 100;
    var maxDepth : int  = 6;
    var eta      : real = 0.1;    // learning rate (shrinkage)
    var lambda   : real = 1.0;    // L2 regularisation on leaf weights
    var minHess  : real = 1.0;    // minimum hessian sum per leaf
    var tau      : real = 0.5;    // quantile level (Pinball loss only)
  }

  // ------------------------------------------------------------------
  // boost
  //
  // Trains a GBM ensemble on data for the given objective.
  // data.F is updated in-place as trees are added.
  // Returns the fitted trees (the full ensemble).
  // ------------------------------------------------------------------
  proc boost(
      ref data : GBMData,
      obj      : Objective,
      cfg      : BoosterConfig
  ): [] FittedTree {

    // Quantile binning — done once before any boosting rounds
    computeBins(data);

    var trees: [0..#cfg.nTrees] FittedTree;

    for t in 0..#cfg.nTrees {
      trees[t] = new FittedTree(cfg.maxDepth);

      // ----------------------------------------------------------
      // Step 1: gradients for this round
      // ----------------------------------------------------------
      computeGradients(obj, data.F, data.y, data.grad, data.hess,
                       tau=cfg.tau);

      // ----------------------------------------------------------
      // Step 2: level-wise tree building
      //
      // Histogram is sized to tree.nNodes so a single allocation
      // covers all depths.  At each depth only the active nodes
      // (those reachable from the root) have non-zero histogram
      // entries; phantom nodes return valid=false from findBestSplits.
      // ----------------------------------------------------------
      var nodeId : [data.rowDom] int = 0;
      var hist    = new HistogramData(maxNodes = trees[t].nNodes,
                                      nFeatures = data.numFeatures);

      for d in 0..cfg.maxDepth {
        buildHistograms(data, nodeId, hist);

        if d < cfg.maxDepth {
          const splits = findBestSplits(hist, cfg.lambda, cfg.minHess);
          recordLevel(trees[t], splits, hist, d, cfg.lambda);
          updateNodeAssign(data, splits, nodeId);
        } else {
          finalizeLeaves(trees[t], hist, d, cfg.lambda);
        }
      }

      // ----------------------------------------------------------
      // Step 3: update predictions
      // ----------------------------------------------------------
      applyTree(data, trees[t], cfg.eta, data.F);
    }

    return trees;
  }

} // module Booster
