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
  use Logger;
  use Sort;

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
  // initF
  //
  // Initialises data.F to the optimal constant prediction for the given
  // objective before any trees are added.  This is critical for Pinball
  // loss: starting from F=0 with all y > 0 makes every gradient
  // identically -tau, giving near-zero split gain and extremely slow
  // convergence.  MSE and LogLoss are less sensitive but also benefit.
  //
  //   MSE     : F = mean(y)
  //   Pinball : F = tau-th quantile of y
  //   LogLoss : F = log(p_bar / (1 - p_bar)),  p_bar = mean(y)
  // ------------------------------------------------------------------
  private proc initF(ref data: GBMData, obj: Objective, tau: real): real {
    var baseScore: real;
    select obj {
      when Objective.MSE {
        baseScore = (+ reduce data.y) / data.numSamples: real;
      }
      when Objective.Pinball {
        // Gather y to a local array, sort, pick the tau-th quantile.
        var yLocal: [0..#data.numSamples] real;
        forall i in data.rowDom do yLocal[i] = data.y[i];
        sort(yLocal);
        const qIdx = min((tau * data.numSamples: real): int,
                         data.numSamples - 1);
        baseScore = yLocal[qIdx];
        logInfo("initF: Pinball tau=" + tau:string
              + " baseline=" + baseScore:string);
      }
      when Objective.LogLoss {
        use Math;
        const pMean = (+ reduce data.y) / data.numSamples: real;
        const p     = max(1e-7, min(1.0 - 1e-7, pMean));
        baseScore = log(p / (1.0 - p));
      }
      otherwise halt("initF: unknown objective");
    }
    data.F = baseScore;
    return baseScore;
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
        logTrace("tree="     + t:string
               + " depth="   + d:string
               + " node="    + idx:string
               + " feature=" + s.feature:string
               + " bin="     + s.bin:string
               + " gain="    + s.gain:string
               + " leftH="   + s.leftHess:string);
    }
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
  ) {

    // Caller is responsible for calling computeBins(data) before boost()
    // so that data.Xb is populated.  This keeps binning explicit and
    // allows the same BinCuts to be reused for test-set prediction.

    // Initialise F to the optimal constant for this objective so that
    // the first gradient computation is informative.
    const baseScore = initF(data, obj, cfg.tau);

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
          logSplits(t, d, splits);
          recordLevel(trees[t], splits, hist, d, cfg.lambda, cfg.eta);
          updateNodeAssign(data, splits, nodeId);
        } else {
          finalizeLeaves(trees[t], hist, d, cfg.lambda, cfg.eta);
          const nLeaves = + reduce trees[t].isLeaf: int;
          logInfo("tree=" + t:string + " leaves=" + nLeaves:string);
        }
      }

      // ----------------------------------------------------------
      // Step 3: update predictions
      // ----------------------------------------------------------
      applyTree(data, trees[t], data.F);
      logInfo("tree=" + t:string
            + " trainLoss=" + computeLoss(obj, data.F, data.y, cfg.tau):string);
    }

    return (trees, baseScore);
  }

  // ------------------------------------------------------------------
  // predict
  //
  // Apply the fitted ensemble to data and return predictions.
  // data.Xb must already be filled (boost() does this for training data;
  // call applyBins(testData, cuts) before predicting on new data).
  // data.F is NOT modified.
  // ------------------------------------------------------------------
  proc predict(trees: [] FittedTree, data: GBMData,
               baseScore: real = 0.0): [] real {
    var preds: [data.rowDom] real = baseScore;
    for t in trees.domain {
      applyTree(data, trees[t], preds);
    }
    return preds;
  }

} // module Booster
