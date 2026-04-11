/*
  Histogram.chpl
  --------------
  Histogram accumulation for GBM tree building.

  A Histogram holds per-(node, feature, bin) gradient and hessian sums.
  Memory layout: [node, feature, bin]  (see notes.md for the trade-off
  vs [feature, bin, node] — benchmark deferred).

  Key operations:
    buildHistograms    — accumulate grad/hess over all samples in active nodes
    subtractHistograms — derive larger child hist as parent minus smaller child

  Implicit master-worker pattern:
    buildHistograms runs a distributed forall (all locales contribute) with
    + reduce intent so each task accumulates locally before merging back to
    locale 0.  The resulting HistogramData lives on locale 0.
    subtractHistograms is pure local arithmetic and should be called on
    locale 0.
*/

module Histogram {

  use DataLayout;
  use Binning;   // MAX_BINS

  // ------------------------------------------------------------------
  // HistogramData record
  //
  // Plain (non-distributed) arrays — small enough to live on locale 0.
  // Indexed [node, feature, bin].
  // ------------------------------------------------------------------
  record HistogramData {
    var maxNodes  : int;
    var nFeatures : int;

    var histDom : domain(3);
    var grad    : [histDom] real;
    var hess    : [histDom] real;

    proc init(maxNodes: int, nFeatures: int) {
      this.maxNodes  = maxNodes;
      this.nFeatures = nFeatures;
      this.histDom   = {0..#maxNodes, 0..#nFeatures, 0..#MAX_BINS};
    }
  }

  // ------------------------------------------------------------------
  // buildHistograms
  //
  // Accumulates gradient and hessian sums into hist for all active nodes.
  //
  // nodeId[i] — which tree node sample i belongs to (0-indexed)
  //
  // Uses + reduce intent so each task accumulates into a task-local copy
  // before the final merge, avoiding cross-task races in the inner loop.
  // ------------------------------------------------------------------
  proc buildHistograms(
      data    : GBMData,
      nodeId  : [] int,
      ref hist: HistogramData
  ) {
    const nF = data.numFeatures;

    hist.grad = 0.0;
    hist.hess = 0.0;

    // Parallel over features: each task owns a disjoint [*, f, *] slice of
    // hist, so no data races and no per-task reduce copies are needed.
    forall f in 0..#nF with (ref hist) {
      for i in data.rowDom {
        const node = nodeId[i];
        const b    = data.Xb[i, f]: int;
        hist.grad[node, f, b] += data.grad[i];
        hist.hess[node, f, b] += data.hess[i];
      }
    }
  }

  // ------------------------------------------------------------------
  // subtractHistograms
  //
  // Derives the larger child's histogram as parent - smallerChild.
  // Called once per split to avoid rebuilding the larger child from
  // scratch.  Run on locale 0 — pure local arithmetic.
  // ------------------------------------------------------------------
  proc subtractHistograms(
      parent     : HistogramData,
      smallChild : HistogramData,
      ref large  : HistogramData
  ) {
    large.grad = parent.grad - smallChild.grad;
    large.hess = parent.hess - smallChild.hess;
  }

} // module Histogram
