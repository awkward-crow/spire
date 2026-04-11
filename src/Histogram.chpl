/*
  Histogram.chpl
  --------------
  Histogram accumulation for GBM tree building.

  A Histogram holds per-(node, feature, bin) gradient and hessian sums.
  Memory layout: [node, feature, bin]  (see notes.md for the trade-off
  vs [feature, bin, node] — benchmark deferred).

  Key operations:
    buildHistograms     — full rebuild: accumulate over all samples (depth 0)
    buildHistogramsLeft — partial rebuild: accumulate only into left children;
                          right children are derived by subtractSiblings in
                          Booster.chpl (histogram subtraction trick)
    subtractHistograms  — derive larger child hist as parent minus smaller child
                          (low-level helper; also used directly in tests)

  buildHistograms parallelises over features: each task owns a disjoint
  [*, f, *] slice so no reduce copies are needed.
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
  // buildHistogramsLeft
  //
  // Histogram subtraction trick: only accumulate into left children
  // (odd heap index) at this depth.  Parallel over features, same as
  // buildHistograms.
  //
  // Only the left-child slots for this depth are zeroed; parent slots
  // from the previous depth are left intact so that Booster.chpl can
  // derive right children as  parent − left  via subtractSiblings.
  //
  // depth >= 1.  Left children at depth d have heap indices
  //   (2^d − 1), (2^d + 1), (2^d + 3), ...   (odd numbers in level d).
  // ------------------------------------------------------------------
  proc buildHistogramsLeft(
      data    : GBMData,
      nodeId  : [] int,
      ref hist: HistogramData,
      depth   : int
  ) {
    const nF        = data.numFeatures;
    const firstLeft = (1 << depth) - 1;    // heapIdx(depth, 0) — first left child
    const nLeft     = 1 << (depth - 1);    // 2^(depth-1) left children at this depth

    forall f in 0..#nF with (ref hist) {
      // Zero only the left-child slots for this feature.
      for ln in 0..#nLeft {
        const left = firstLeft + 2 * ln;
        hist.grad[left, f, ..] = 0.0;
        hist.hess[left, f, ..] = 0.0;
      }
      // Accumulate samples in left children at this depth.
      // Both conditions are required:
      //   node & 1 == 1    — left children have odd heap indices
      //   node >= firstLeft — exclude samples retained at shallower
      //                       leaf nodes (their nodeId < firstLeft)
      for i in data.rowDom {
        const node = nodeId[i];
        if node & 1 == 1 && node >= firstLeft {
          const b = data.Xb[i, f]: int;
          hist.grad[node, f, b] += data.grad[i];
          hist.hess[node, f, b] += data.hess[i];
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // subtractHistograms
  //
  // Derives the larger child's histogram as parent - smallerChild.
  // Low-level helper used in tests; see subtractSiblings in Booster.chpl
  // for the in-loop version that operates on node indices.
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
