/*
  Booster.chpl
  ------------
  Gradient boosting training loop.  Ties together Binning, Objectives,
  Histogram, Splits, and Tree into a complete end-to-end GBM.

  One call to boost():
    1. obj.initF(data)     — set data.F to objective-optimal constant
    2. For each tree:
       a. obj.gradients    — compute per-sample grad/hess
       b. Level-wise tree building (0..maxDepth):
            buildHistograms → findBestSplits → recordLevel → updateNodeAssign
          then finalizeLeaves at maxDepth
       c. applyTree        — data.F += eta * tree predictions
    3. Returns a GBMEnsemble bundling the fitted trees and baseScore

  The histogram is allocated once per tree (sized to tree.nNodes =
  2^(maxDepth+1)-1) and reused across depths.  nodeId uses absolute heap
  indexing throughout (see Tree.chpl).

  GBMEnsemble
  -----------
  Bundles the fitted [] FittedTree with the baseScore so callers never
  need to thread the scalar separately through boost() and predict().
*/

module Booster {

  use DataLayout;
  use Objectives;
  use Binning;
  use Histogram;
  use Splits;
  use Tree;
  use Logger;
  use Time;

  // ------------------------------------------------------------------
  // GBMEnsemble — the complete fitted model
  // ------------------------------------------------------------------
  record GBMEnsemble {
    var treeDom  : domain(1);
    var trees    : [treeDom] FittedTree;
    var baseScore: real;

    proc init(trees: [] FittedTree, baseScore: real) {
      this.treeDom   = trees.domain;
      this.trees     = trees;
      this.baseScore = baseScore;
    }
  }

  // ------------------------------------------------------------------
  // BoosterConfig
  // ------------------------------------------------------------------
  record BoosterConfig {
    var nTrees   : int  = 100;
    var maxDepth : int  = 6;
    var eta      : real = 0.1;    // learning rate (shrinkage)
    var lambda   : real = 1.0;    // L2 regularisation on leaf weights
  }

  // ------------------------------------------------------------------
  // subtractSiblings
  //
  // For each split node at depth-1, derives the right child's histogram
  // as  hist[parent] − hist[left].  Skips unsplit/phantom parents so
  // their right-child slots remain zero.
  //
  // Called after buildHistogramsLeft; together they implement the
  // histogram subtraction trick: only left children are built from
  // scratch, halving the sample-pass work at every depth after depth 0.
  // ------------------------------------------------------------------
  private proc subtractSiblings(
      ref hist  : HistogramData,
      splits    : [] SplitInfo,
      depth     : int
  ) {
    for n in 0..#(1 << (depth - 1)) {
      const parent = (1 << (depth - 1)) - 1 + n;   // heapIdx(depth-1, n)
      if !splits[parent].valid then continue;
      const left  = 2 * parent + 1;
      const right = 2 * parent + 2;
      hist.grad[right, .., ..] = hist.grad[parent, .., ..] - hist.grad[left, .., ..];
      hist.hess[right, .., ..] = hist.hess[parent, .., ..] - hist.hess[left, .., ..];
    }
  }

  // ------------------------------------------------------------------
  // logSplits — trace-level logging of split decisions at one depth.
  // ------------------------------------------------------------------
  private proc logSplits(t: int, d: int, splits: [] SplitInfo) {
    if logLevel < LogLevel.TRACE then return;
    for n in 0..#(1 << d) {
      const idx = heapIdx(d, n);
      const s   = splits[idx];
      if s.valid then
        logTrace("boost: tree="   + t:string
               + " depth="       + d:string
               + " node="        + idx:string
               + " feature="     + s.feature:string
               + " bin="         + s.bin:string
               + " gain="        + s.gain:string
               + " leftH="       + s.leftHess:string);
    }
  }

  // ------------------------------------------------------------------
  // boost
  //
  // Trains a GBM ensemble on data for the given objective.
  // data.F is updated in-place as trees are added.
  // Returns a GBMEnsemble (fitted trees + baseScore).
  //
  // The objective (MSE, LogLoss, or Pinball) must provide:
  //   obj.initF(data)         — set data.F to optimal constant, return it
  //   obj.gradients(F,y,g,h)  — fill grad/hess arrays in-place
  //   obj.loss(F,y)           — scalar loss for logging
  //   obj.defaultMinHess()    — minimum hessian for valid splits
  // ------------------------------------------------------------------
  proc boost(
      ref data : GBMData,
      obj      : ?T,
      cfg      : BoosterConfig
  ): GBMEnsemble
  {

    // Caller is responsible for calling computeBins(data) before boost()
    // so that data.Xb is populated.  This keeps binning explicit and
    // allows the same BinCuts to be reused for test-set prediction.

    // Initialise F to the optimal constant for this objective so that
    // the first gradient computation is informative.
    const baseScore = obj.initF(data);

    var trees: [0..#cfg.nTrees] FittedTree;

    var boostTimer: stopwatch;
    boostTimer.start();

    for t in 0..#cfg.nTrees {
      trees[t] = new FittedTree(cfg.maxDepth);

      // ----------------------------------------------------------
      // Step 1: gradients for this round
      // ----------------------------------------------------------
      obj.gradients(data.F, data.y, data.grad, data.hess);

      // ----------------------------------------------------------
      // Step 2: level-wise tree building
      //
      // Histogram is sized to tree.nNodes so a single allocation
      // covers all depths.  At depth 0 the histogram is built from
      // scratch.  At depth d > 0 the histogram subtraction trick is
      // used: buildHistogramsLeft accumulates only into left children
      // (odd heap index); subtractSiblings derives each right child as
      //   hist[right] = hist[parent] − hist[left]
      // halving the sample-pass work at every depth after the root.
      //
      // splits is declared outside the loop so it persists as the
      // "previous-depth splits" needed by subtractSiblings.
      // ----------------------------------------------------------
      var nodeId : [data.rowDom] int = 0;
      var hist    = new HistogramData(maxNodes = trees[t].nNodes,
                                      nFeatures = data.numFeatures);
      var splits  : [0..#trees[t].nNodes] SplitInfo;

      for d in 0..cfg.maxDepth {
        if d == 0 {
          buildHistograms(data, nodeId, hist);
        } else {
          buildHistogramsLeft(data, nodeId, hist, d);
          subtractSiblings(hist, splits, d);
        }

        if d < cfg.maxDepth {
          splits = findBestSplits(hist, cfg.lambda, obj.defaultMinHess());
          logSplits(t, d, splits);
          recordLevel(trees[t], splits, hist, d, cfg.lambda, cfg.eta);
          updateNodeAssign(data, splits, nodeId);
        } else {
          finalizeLeaves(trees[t], hist, d, cfg.lambda, cfg.eta);
          // Post-hoc leaf refit: for quantile objectives, replace Newton-step
          // leaf values with the tau-quantile of per-leaf residuals.
          // No-op for MSE and LogLoss.
          obj.leafRefit(trees[t], nodeId, data.F, data.y, cfg.eta);
          const nLeaves = + reduce trees[t].isLeaf: int;
          logInfo("boost: tree=" + t:string + " leaves=" + nLeaves:string
                + " trainLoss=" + obj.loss(data.F, data.y):string);
        }
      }

      // ----------------------------------------------------------
      // Step 3: update predictions
      // ----------------------------------------------------------
      applyTree(data, trees[t], data.F);
    }

    boostTimer.stop();
    logInfo("boost: elapsed=" + boostTimer.elapsed():string + "s"
          + " trees=" + cfg.nTrees:string
          + " samples=" + data.numSamples:string
          + " features=" + data.numFeatures:string);

    return new GBMEnsemble(trees=trees, baseScore=baseScore);
  }

  // ------------------------------------------------------------------
  // predict
  //
  // Apply a fitted ensemble to data and return predictions.
  // data.Xb must already be filled (boost() does this for training data;
  // call applyBins(testData, cuts) before predicting on new data).
  // data.F is NOT modified.
  // ------------------------------------------------------------------
  proc predict(ensemble: GBMEnsemble, data: GBMData): [] real {
    var preds: [data.rowDom] real = ensemble.baseScore;
    for t in ensemble.trees.domain do
      applyTree(data, ensemble.trees[t], preds);
    return preds;
  }

} // module Booster
