/*
  Histogram.chpl
  --------------
  Histogram accumulation for GBM tree building.

  A Histogram holds per-(node, feature, bin) gradient and hessian sums,
  stored as an array of GH records (AoS layout).

  AoS vs SoA for the scatter:
    SoA (old): separate grad[] and hess[] arrays.  Each scatter pair touches
               two different memory regions → two cache misses per sample.
    AoS (new): interleaved {grad, hess} per bin.  grad and hess for the same
               (f, b, slot) are 4 bytes apart in the same cache line →
               one cache miss per scatter pair.  Halves the dominant scatter
               cache miss rate in buildHistogramsNodes.

  4-sample unrolled C kernel (hist_kernel.c):
    The per-feature inner loop calls spire_histKernel, which processes 4
    samples per iteration.  Four independent scatter ops in-flight allows
    the CPU to overlap their address computations and cache-line fetches,
    improving throughput ~2–3× vs the 1-sample/iteration Chapel loop.
    nodeToSlot[nodeId[i]] is precomputed once per locale (not per feature)
    into an int32 nodeSlots[] array so the kernel receives a single slot
    lookup per sample.

  Memory layout: [feature, bin, node] — node varies fastest so that the
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
    - Local accumulator (lgh) 2× smaller → better L1 fit per feature
    - Multi-locale reduction payload 2× smaller (≈55 KB per locale per
      split for CoverType vs ≈110 KB)
  Callers in Splits.chpl and Booster.chpl mix real(32) histogram values
  with real(64) lambda/eta; Chapel widens as needed.
*/

module Histogram {

  use DataLayout;
  use Binning;   // MAX_BINS
  use CTypes;    // c_ptr, c_ptrTo, c_int

  // 4-sample unrolled histogram scatter kernel — separate C translation unit
  // so the compiler sees a clean scatter loop with no Chapel overhead.
  require "hist_kernel.h", "hist_kernel.c";

  extern proc spire_histKernel(
      grad    : c_ptr(real(32)),
      hess    : c_ptr(real(32)),
      bins    : c_ptr(uint(8)),
      slots   : c_ptr(int(8)),
      nSamples: c_int,
      nSlots  : c_int,
      lghOut  : c_ptr(real(32))
  );

  // ------------------------------------------------------------------
  // GH — interleaved gradient/hessian pair for one histogram bin.
  //
  // Packing grad and hess together means a single cache-line load covers
  // both fields.  In the scatter-heavy histogram inner loop, this halves
  // cache misses vs separate grad[]/hess[] arrays (SoA layout).
  // ------------------------------------------------------------------
  record GH {
    var grad: real(32);
    var hess: real(32);
  }

  // ------------------------------------------------------------------
  // HistogramData record
  //
  // Plain (non-distributed) arrays — small enough to live on locale 0.
  // Indexed [feature, bin, node].  AoS GH pairs; real(32) to halve
  // memory vs real(64).
  // ------------------------------------------------------------------
  record HistogramData {
    var maxNodes  : int;
    var nFeatures : int;

    var histDom : domain(3);
    var bins    : [histDom] GH;   // bins[f, b, n] = {grad, hess}

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
    hist.bins = new GH();

    forall f in featSubset with (ref hist) {
      for i in data.rowDom {
        const node = nodeId[i];
        const b    = data.Xb[f, i]: int;
        hist.bins[f, b, node].grad += data.grad[i];
        hist.bins[f, b, node].hess += data.hess[i];
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
  // hist.  Reduction payload: numLocales × nFeatures × MAX_BINS × 8 B
  // (≈110 KB per locale for CoverType with GH pairs).
  //
  // Cost: O(N) nodeId reads (local per locale) + O(nFeatures × MAX_BINS)
  // reduction.
  // ------------------------------------------------------------------
  proc buildHistogramsNode(
      ref data   : GBMData,
      nodeId     : [] int,
      ref hist   : HistogramData,
      targetNode : int,
      featSubset : [] int
  ) {
    const nF   = hist.nFeatures;
    const pDom = {0..#nF, 0..#MAX_BINS};

    var partial: [0..#numLocales, 0..#nF, 0..#MAX_BINS] GH;

    coforall loc in Locales with (ref data) {
      on loc {
        const lid        = loc.id;
        const localDom   = data.rowDom.localSubdomain();
        const localFeats = featSubset;
        const nLocal     = localDom.size;

        // Precompute per-sample slot once (not per feature):
        // slot = 0 if sample is in targetNode, -1 otherwise.
        var nodeSlots: [0..#nLocal] int(8);
        for ii in 0..#nLocal {
          nodeSlots[ii] = if nodeId[localDom.low + ii] == targetNode
                          then 0: int(8) else -1: int(8);
        }

        var lgh: [pDom] GH;   // local accumulator — 1 cache miss per scatter pair

        // Pass local grad/hess/Xb pointers and nodeSlots to the C kernel.
        // c_ptrTo(lgh[f, 0].grad) points to the first float of the
        // MAX_BINS-wide GH block for feature f (contiguous, stride-1).
        forall f in localFeats with (ref lgh, ref data) {
          spire_histKernel(
              c_ptrTo(data.grad[localDom.low]),
              c_ptrTo(data.hess[localDom.low]),
              c_ptrTo(data.Xb[f, localDom.low]),
              c_ptrTo(nodeSlots[0]),
              nLocal: c_int,
              1: c_int,
              c_ptrTo(lgh[f, 0].grad)
          );
        }

        partial[lid, 0..#nF, 0..#MAX_BINS] = lgh;
      }
    }

    forall f in featSubset with (ref hist) {
      hist.bins[f, .., targetNode] = new GH();
      for loc in 0..#numLocales {
        for b in 0..#MAX_BINS {
          hist.bins[f, b, targetNode].grad += partial[loc, f, b].grad;
          hist.bins[f, b, targetNode].hess += partial[loc, f, b].hess;
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
        hist.bins[f, b, larger].grad = hist.bins[f, b, parent].grad - hist.bins[f, b, smaller].grad;
        hist.bins[f, b, larger].hess = hist.bins[f, b, parent].hess - hist.bins[f, b, smaller].hess;
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
        hist.bins[f, .., left] = new GH();
      }
      for i in data.rowDom {
        const node = nodeId[i];
        if node & 1 == 1 && node >= firstLeft {
          const b = data.Xb[f, i]: int;
          hist.bins[f, b, node].grad += data.grad[i];
          hist.bins[f, b, node].hess += data.hess[i];
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
    forall (f, b, n) in large.histDom with (ref large) {
      large.bins[f, b, n].grad = parent.bins[f, b, n].grad - smallChild.bins[f, b, n].grad;
      large.bins[f, b, n].hess = parent.bins[f, b, n].hess - smallChild.bins[f, b, n].hess;
    }
  }

  // ------------------------------------------------------------------
  // buildHistogramsNodes
  //
  // Batched version of buildHistogramsNode: accumulates only samples
  // whose nodeId[i] is in targetNodes (the k smaller children) into
  // the corresponding histogram slots — all in a single sample pass.
  //
  // targetNodes  — array of k smaller-child node indices to populate.
  //                Length k must be >= 1.
  //
  // A nodeToSlot lookup (maxNodes-length int array, -1 = unset) maps
  // each node ID to its position in targetNodes so the inner loop is
  // O(1) instead of O(k).  Local accumulator lgh is [nF, MAX_BINS, k]
  // of GH records so each forall-f task owns a disjoint [f, *, *] slice
  // — race-free, same pattern as buildHistogramsNode.
  //
  // Reduction payload: numLocales × nF × MAX_BINS × k × 8 B.
  // For CoverType, k=4: ≈1.76 MB per reduce.
  // ------------------------------------------------------------------
  proc buildHistogramsNodes(
      ref data    : GBMData,
      nodeId      : [] int,
      ref hist    : HistogramData,
      targetNodes : [] int,
      featSubset  : [] int
  ) {
    const nF   = hist.nFeatures;
    const maxN = hist.maxNodes;
    const k    = targetNodes.size;

    // Map node ID → slot index (0..#k).  Entries not in targetNodes stay -1.
    var nodeToSlot: [0..#maxN] int = -1;
    for slot in 0..#k do nodeToSlot[targetNodes[slot]] = slot;

    var partial: [0..#numLocales, 0..#nF, 0..#MAX_BINS, 0..#k] GH;

    coforall loc in Locales with (ref data) {
      on loc {
        const lid        = loc.id;
        const localDom   = data.rowDom.localSubdomain();
        const localFeats = featSubset;
        const localN2S   = nodeToSlot;   // broadcast: maxNodes ints, tiny
        const nLocal     = localDom.size;

        // Precompute per-sample slot once (not per feature):
        // slot = nodeToSlot[nodeId[i]], or -1 if sample is inactive.
        var nodeSlots: [0..#nLocal] int(8);
        for ii in 0..#nLocal {
          nodeSlots[ii] = localN2S[nodeId[localDom.low + ii]]: int(8);
        }

        var lgh: [0..#nF, 0..#MAX_BINS, 0..#k] GH;

        // Call the C kernel per feature.  c_ptrTo(lgh[f, 0, 0].grad) points
        // to the first float of the [MAX_BINS × k] GH block for feature f
        // (contiguous, row-major: slot varies fastest, then bin, then feature).
        forall f in localFeats with (ref lgh, ref data) {
          spire_histKernel(
              c_ptrTo(data.grad[localDom.low]),
              c_ptrTo(data.hess[localDom.low]),
              c_ptrTo(data.Xb[f, localDom.low]),
              c_ptrTo(nodeSlots[0]),
              nLocal: c_int,
              k: c_int,
              c_ptrTo(lgh[f, 0, 0].grad)
          );
        }

        partial[lid, .., .., ..] = lgh;
      }
    }

    // Reduce partials into hist, one slot per target node.
    forall f in featSubset with (ref hist) {
      for slot in 0..#k {
        const n = targetNodes[slot];
        hist.bins[f, .., n] = new GH();
        for loc in 0..#numLocales {
          for b in 0..#MAX_BINS {
            hist.bins[f, b, n].grad += partial[loc, f, b, slot].grad;
            hist.bins[f, b, n].hess += partial[loc, f, b, slot].hess;
          }
        }
      }
    }
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
