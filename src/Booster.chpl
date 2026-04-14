/*
  Booster.chpl
  ------------
  Gradient boosting training loop.  Ties together Binning, Objectives,
  Histogram, Splits, and Tree into a complete end-to-end GBM.

  One call to boost():
    1. obj.initF(data)       — set data.F to objective-optimal constant
    2. For each tree:
       a. obj.gradients      — compute per-sample grad/hess
       b. Leaf-wise tree building:
            build histograms for all active leaves (one sample pass)
            → findBestSplits → pick highest-gain leaf → split
            repeat until numLeaves budget is exhausted
       c. applyTree          — data.F += eta * tree predictions
    3. Returns a GBMEnsemble bundling the fitted trees and baseScore

  Leaf-wise growth: each round expands the single highest-gain active
  leaf.  This produces asymmetric trees that concentrate depth where the
  data has the most signal, matching LightGBM's default growth strategy.

  The histogram is allocated once per tree (sized to 2*numLeaves-1) and
  rebuilt from scratch on each expansion.  All active leaves are
  accumulated in a single sample pass via nodeId routing.

  GBMEnsemble
  -----------
  Bundles the fitted [] FittedTree with the baseScore so callers never
  need to thread the scalar separately.
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
  use Random;

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
    var nTrees          : int  = 100;
    var numLeaves       : int  = 31;    // max leaves per tree
    var eta             : real = 0.1;   // learning rate (shrinkage)
    var lambda          : real = 1.0;   // L2 regularisation on leaf weights
    var colsampleByTree : real = 1.0;   // fraction of features sampled per tree
    var seed            : int  = 42;    // RNG seed for column subsampling
  }

  // ------------------------------------------------------------------
  // boost
  //
  // Trains a GBM ensemble on data for the given objective.
  // data.F is updated in-place as trees are added.
  // Returns a GBMEnsemble (fitted trees + baseScore).
  //
  // The objective must provide:
  //   obj.initF(data)         — set data.F to optimal constant, return it
  //   obj.gradients(F,y,g,h)  — fill grad/hess arrays in-place
  //   obj.loss(F,y)           — scalar loss for logging
  //   obj.defaultMinHess()    — minimum hessian for valid splits
  //   obj.leafRefit(tree,…)   — post-hoc leaf refit (no-op for MSE/LogLoss)
  // ------------------------------------------------------------------
  proc boost(
      ref data : GBMData,
      obj      : ?T,
      cfg      : BoosterConfig
  ): GBMEnsemble
  {
    const baseScore = obj.initF(data);

    var trees: [0..#cfg.nTrees] FittedTree;

    const nF        = data.numFeatures;
    const nFSub     = max(1, (nF * cfg.colsampleByTree): int);
    const maxNodes  = 2 * cfg.numLeaves - 1;

    // Single RNG advanced continuously across all trees.
    var rng = new randomStream(real, seed = cfg.seed);

    var boostTimer: stopwatch;
    boostTimer.start();

    for t in 0..#cfg.nTrees {
      trees[t] = new FittedTree(cfg.numLeaves);

      // ----------------------------------------------------------
      // Column subsampling: partial Fisher-Yates on persistent RNG.
      // ----------------------------------------------------------
      var allFeats: [0..#nF] int = [i in 0..#nF] i;
      if nFSub < nF {
        for i in 0..#nFSub {
          const j = i + (rng.next() * (nF - i)): int;
          allFeats[i] <=> allFeats[j];
        }
      }
      const featSubset = allFeats[0..#nFSub];
      const anchorFeat = featSubset[featSubset.domain.low];

      // ----------------------------------------------------------
      // Step 1: gradients for this round
      // ----------------------------------------------------------
      obj.gradients(data.F, data.y, data.grad, data.hess);

      // ----------------------------------------------------------
      // Step 2: leaf-wise tree building
      // ----------------------------------------------------------
      var nodeId: [data.rowDom] int = 0;
      var hist = new HistogramData(maxNodes = maxNodes, nFeatures = nF);

      // Initialise tree arrays: all nodes start as leaves with no children.
      trees[t].isLeaf     = true;
      trees[t].leftChild  = -1;
      trees[t].rightChild = -1;

      // Build first histogram (all samples at root, node 0).
      buildHistograms(data, nodeId, hist, featSubset);

      // Root leaf value (overwritten if root gets split).
      {
        const G = + reduce hist.grad[anchorFeat, .., 0];
        const H = + reduce hist.hess[anchorFeat, .., 0];
        trees[t].value[0] = cfg.eta * leafValue(G, H, cfg.lambda);
      }

      // Active leaf tracking and cached splits.
      var activeLeaves: [0..#cfg.numLeaves] int;
      activeLeaves[0] = 0;
      var nActive = 1;

      var cachedSplits: [0..#maxNodes] SplitInfo;
      {
        const splits = findBestSplits(hist, cfg.lambda, obj.defaultMinHess(), featSubset);
        cachedSplits[0] = splits[0];
      }

      // Expand one leaf at a time until numLeaves budget is exhausted.
      while nActive < cfg.numLeaves {

        // Find the highest-gain active leaf.
        var bestLeaf = -1;
        var bestGain = 0.0;
        for k in 0..#nActive {
          const leaf = activeLeaves[k];
          if cachedSplits[leaf].valid && cachedSplits[leaf].gain > bestGain {
            bestGain = cachedSplits[leaf].gain;
            bestLeaf = leaf;
          }
        }
        if bestLeaf == -1 then break;   // no profitable splits remain

        // Allocate two child nodes.
        const left  = trees[t].nNodes; trees[t].nNodes += 1;
        const right = trees[t].nNodes; trees[t].nNodes += 1;

        // Record split: bestLeaf becomes internal.
        trees[t].isLeaf[bestLeaf]     = false;
        trees[t].feature[bestLeaf]    = cachedSplits[bestLeaf].feature;
        trees[t].splitBin[bestLeaf]   = cachedSplits[bestLeaf].bin;
        trees[t].leftChild[bestLeaf]  = left;
        trees[t].rightChild[bestLeaf] = right;

        // Route samples from bestLeaf to its children.
        updateNodeAssign(data, bestLeaf, cachedSplits[bestLeaf], left, right, nodeId);

        // Update active leaf list: replace bestLeaf with left, append right.
        for k in 0..#nActive {
          if activeLeaves[k] == bestLeaf {
            activeLeaves[k] = left;
            break;
          }
        }
        activeLeaves[nActive] = right;
        nActive += 1;

        // Rebuild histograms for all active leaves and find splits.
        buildHistograms(data, nodeId, hist, featSubset);
        const splits = findBestSplits(hist, cfg.lambda, obj.defaultMinHess(), featSubset);

        // Refresh cached splits and leaf values for every active leaf.
        for k in 0..#nActive {
          const leaf = activeLeaves[k];
          cachedSplits[leaf] = splits[leaf];
          const G = + reduce hist.grad[anchorFeat, .., leaf];
          const H = + reduce hist.hess[anchorFeat, .., leaf];
          trees[t].value[leaf] = cfg.eta * leafValue(G, H, cfg.lambda);
        }
      }

      // Post-hoc leaf refit (Pinball only; no-op for MSE / LogLoss).
      obj.leafRefit(trees[t], nodeId, data.F, data.y, cfg.eta);

      logInfo("boost: tree=" + t:string + " leaves=" + nActive:string
            + " trainLoss=" + obj.loss(data.F, data.y):string);

      // ----------------------------------------------------------
      // Step 3: update predictions
      // ----------------------------------------------------------
      applyTree(data, trees[t], data.F);
    }

    boostTimer.stop();
    logInfo("boost: elapsed=" + boostTimer.elapsed():string + "s"
          + " trees="   + cfg.nTrees:string
          + " samples=" + data.numSamples:string
          + " features=" + data.numFeatures:string);

    return new GBMEnsemble(trees=trees, baseScore=baseScore);
  }

  // ------------------------------------------------------------------
  // predict
  //
  // Apply a fitted ensemble to data and return predictions.
  // data.Xb must already be filled (call applyBins before predicting
  // on held-out data).  data.F is NOT modified.
  // ------------------------------------------------------------------
  proc predict(ensemble: GBMEnsemble, data: GBMData): [] real {
    var preds: [data.rowDom] real = ensemble.baseScore;
    for t in ensemble.trees.domain do
      applyTree(data, ensemble.trees[t], preds);
    return preds;
  }

} // module Booster
