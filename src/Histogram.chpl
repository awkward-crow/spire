/*
  Histogram.chpl
  --------------
  Histogram accumulation for GBM tree building.

  A Histogram holds per-(node, feature, bin) gradient and hessian sums.
  Memory layout: [feature, bin, node]  — node varies fastest so that the
  scatter writes in buildHistograms (varying node, fixed feature) are
  stride-1.  findBestSplits scans bins for a fixed (feature, node), which
  has stride maxNodes; that is a smaller access volume than the scatter.

  Key operations:
    buildHistograms     — full rebuild: accumulate over all samples (root only)
    buildHistogramsNode — filtered rebuild: accumulate only samples at one node
                          (smaller-child pass for the subtraction trick)
    subtractNode        — in-place: hist[larger] = hist[parent] - hist[smaller]
                          O(nFeatures × MAX_BINS), derives larger child for free
    buildHistogramsLeft — partial rebuild: accumulate only into left children
    subtractHistograms  — derive larger child hist as parent minus smaller child
                          (low-level helper; also used directly in tests)

  buildHistograms parallelises over features: each task owns a disjoint
  [*, f, *] slice so no reduce copies are needed.

  Gradient storage is real(32) throughout (data.grad, data.hess, and the
  histogram bins).  Benefits vs real(64):
    - Histogram arrays 2× smaller → better L2 cache fit for the bin scan
    - Local accumulator (lg, lh) 2× smaller → better L1 fit per feature
    - Multi-locale reduction payload 2× smaller (≈55 KB per locale per
      split for CoverType vs ≈110 KB)
  Callers in Splits.chpl and Booster.chpl mix real(32) histogram values
  with real(64) lambda/eta; Chapel widens as needed.
*/

module Histogram {

  use DataLayout;
  use Binning;   // MAX_BINS

  // ------------------------------------------------------------------
  // HistogramData record
  //
  // Plain (non-distributed) arrays — small enough to live on locale 0.
  // Indexed [feature, bin, node].  real(32) to halve memory vs real(64).
  // ------------------------------------------------------------------
  record HistogramData {
    var maxNodes  : int;
    var nFeatures : int;

    var histDom : domain(3);
    var grad    : [histDom] real(32);
    var hess    : [histDom] real(32);

    proc init(maxNodes: int, nFeatures: int) {
      this.maxNodes  = maxNodes;
      this.nFeatures = nFeatures;
      this.histDom   = {0..#nFeatures, 0..#MAX_BINS, 0..#maxNodes};
    }
  }

  // ------------------------------------------------------------------
  // buildHistograms
  //
  // Accumulates gradient and hessian sums into hist for all active nodes.
  //
  // NOTE: iterates over the full block-distributed data.rowDom from the
  // calling locale, causing remote GETs on multi-locale runs.  Kept for
  // backward compatibility with tests (which run single-locale).  The
  // Booster.chpl training loop uses buildHistogramsNode instead.
  // ------------------------------------------------------------------
  proc buildHistograms(
      data       : GBMData,
      nodeId     : [] int,
      ref hist   : HistogramData,
      featSubset : [] int
  ) {
    hist.grad = 0: real(32);
    hist.hess = 0: real(32);

    forall f in featSubset with (ref hist) {
      for i in data.rowDom {
        const node = nodeId[i];
        const b    = data.Xb[i, f]: int;
        hist.grad[f, b, node] += data.grad[i];
        hist.hess[f, b, node] += data.hess[i];
      }
    }
  }

  // ------------------------------------------------------------------
  // buildHistogramsNode
  //
  // Filtered accumulation: accumulates only samples where
  // nodeId[i] == targetNode into the corresponding slot of hist.
  //
  // Multi-locale: each locale accumulates a private [nFeatures, MAX_BINS]
  // partial histogram over its local rows (no remote GETs in the inner
  // loop), then bulk-copies its partial to a slot in a locale-indexed
  // staging array on locale 0.  Locale 0 then reduces the partials into
  // hist.  Reduction payload: numLocales × nFeatures × MAX_BINS × 4 B
  // (≈55 KB per locale for CoverType).
  //
  // Cost: O(N) nodeId reads (local per locale) + O(nFeatures × MAX_BINS)
  // reduction.
  // ------------------------------------------------------------------
  proc buildHistogramsNode(
      data       : GBMData,
      nodeId     : [] int,
      ref hist   : HistogramData,
      targetNode : int,
      featSubset : [] int
  ) {
    const nF   = hist.nFeatures;
    const pDom = {0..#nF, 0..#MAX_BINS};

    var partialGrad: [0..#numLocales, 0..#nF, 0..#MAX_BINS] real(32) = 0: real(32);
    var partialHess: [0..#numLocales, 0..#nF, 0..#MAX_BINS] real(32) = 0: real(32);

    coforall loc in Locales {
      on loc {
        const lid        = loc.id;
        const localDom   = data.rowDom.localSubdomain();
        const localFeats = featSubset;

        var lg: [pDom] real(32) = 0: real(32);
        var lh: [pDom] real(32) = 0: real(32);

        forall f in localFeats with (ref lg, ref lh) {
          for i in localDom {
            if nodeId[i] == targetNode {
              const b = data.Xb[i, f]: int;
              lg[f, b] += data.grad[i];
              lh[f, b] += data.hess[i];
            }
          }
        }

        partialGrad[lid, 0..#nF, 0..#MAX_BINS] = lg;
        partialHess[lid, 0..#nF, 0..#MAX_BINS] = lh;
      }
    }

    forall f in featSubset with (ref hist) {
      hist.grad[f, .., targetNode] = 0: real(32);
      hist.hess[f, .., targetNode] = 0: real(32);
      for loc in 0..#numLocales {
        for b in 0..#MAX_BINS {
          hist.grad[f, b, targetNode] += partialGrad[loc, f, b];
          hist.hess[f, b, targetNode] += partialHess[loc, f, b];
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // subtractNode
  //
  // In-place derivation of the larger child's histogram:
  //   hist[*, *, larger] = hist[*, *, parent] - hist[*, *, smaller]
  //
  // Cost: O(nFeatures × MAX_BINS) — negligible vs. the sample scan.
  // ------------------------------------------------------------------
  proc subtractNode(
      ref hist   : HistogramData,
      parent     : int,
      smaller    : int,
      larger     : int,
      featSubset : [] int
  ) {
    forall f in featSubset with (ref hist) {
      for b in 0..#MAX_BINS {
        hist.grad[f, b, larger] = hist.grad[f, b, parent] - hist.grad[f, b, smaller];
        hist.hess[f, b, larger] = hist.hess[f, b, parent] - hist.hess[f, b, smaller];
      }
    }
  }

  // ------------------------------------------------------------------
  // buildHistogramsLeft
  //
  // Histogram subtraction trick: only accumulate into left children
  // (odd heap index) at this depth.  Parallel over features, same as
  // buildHistograms.
  // ------------------------------------------------------------------
  proc buildHistogramsLeft(
      data       : GBMData,
      nodeId     : [] int,
      ref hist   : HistogramData,
      depth      : int,
      featSubset : [] int
  ) {
    const firstLeft = (1 << depth) - 1;
    const nLeft     = 1 << (depth - 1);

    forall f in featSubset with (ref hist) {
      for ln in 0..#nLeft {
        const left = firstLeft + 2 * ln;
        hist.grad[f, .., left] = 0: real(32);
        hist.hess[f, .., left] = 0: real(32);
      }
      for i in data.rowDom {
        const node = nodeId[i];
        if node & 1 == 1 && node >= firstLeft {
          const b = data.Xb[i, f]: int;
          hist.grad[f, b, node] += data.grad[i];
          hist.hess[f, b, node] += data.hess[i];
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // subtractHistograms
  //
  // Derives the larger child's histogram as parent - smallerChild.
  // Low-level helper used in tests.
  // ------------------------------------------------------------------
  proc subtractHistograms(
      parent     : HistogramData,
      smallChild : HistogramData,
      ref large  : HistogramData
  ) {
    large.grad = parent.grad - smallChild.grad;
    large.hess = parent.hess - smallChild.hess;
  }

  // ------------------------------------------------------------------
  // Backward-compatible overloads — use all features (no subsampling).
  // ------------------------------------------------------------------
  proc buildHistograms(data: GBMData, nodeId: [] int, ref hist: HistogramData) {
    const allFeats: [0..#data.numFeatures] int = [i in 0..#data.numFeatures] i;
    buildHistograms(data, nodeId, hist, allFeats);
  }

  proc buildHistogramsLeft(data: GBMData, nodeId: [] int, ref hist: HistogramData, depth: int) {
    const allFeats: [0..#data.numFeatures] int = [i in 0..#data.numFeatures] i;
    buildHistogramsLeft(data, nodeId, hist, depth, allFeats);
  }

} // module Histogram
