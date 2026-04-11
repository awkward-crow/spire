/*
  Binning.chpl
  ------------
  Quantile binning: converts the float64 feature matrix X into a uint8
  bin matrix Xb.  Done once at startup before any boosting rounds.

  Algorithm: t-digest per locale, merge at locale 0.
    1. Each locale builds one TDigest per feature over ALL its local rows.
    2. Centroid arrays are PUT to locale 0.
    3. Locale 0 merges the per-locale digests (mergeAndCompress) and
       extracts MAX_BINS-1 evenly-spaced quantile cut-points.
    4. Cut-points are broadcast; each locale bins its own rows locally.

  This replaces the earlier sqrt(nLocalRows) random-sampling approach.
  Using all local rows for the digest gives accurate tail cut-points,
  which is important for Pinball/quantile objectives (tau near 0 or 1).
  Communication cost is O(numLocales × nFeatures × MAX_CENTS) centroid
  records rather than O(numLocales × sqrt(n) × nFeatures) raw values;
  the two are comparable at moderate locale counts and the digest is
  strictly more accurate.

  See TDigest.chpl for the k1-scale algorithm and tail-accuracy argument.
*/

module Binning {

  use DataLayout;
  use Sort;
  use Math;
  use Logger;
  use TDigest;
  use Time;

  param MAX_BINS = 255;   // bins per feature; uint(8) covers 0..254

  // ------------------------------------------------------------------
  // BinCuts
  //
  // Stores the quantile cut-points computed from training data so they
  // can be reapplied to a held-out (test) set via applyBins.
  // ------------------------------------------------------------------
  record BinCuts {
    var nFeatures : int;
    var nCuts     : int;
    var cutDom    : domain(2);
    var values    : [cutDom] real;

    proc init(nFeatures: int, nCuts: int = MAX_BINS - 1) {
      this.nFeatures = nFeatures;
      this.nCuts     = nCuts;
      this.cutDom    = {0..#nFeatures, 0..#nCuts};
    }
  }

  // ------------------------------------------------------------------
  // computeBins
  //
  // Populates data.Xb with bin indices in 0..MAX_BINS-1.
  // Call once before training begins.
  // ------------------------------------------------------------------
  proc computeBins(ref data: GBMData): BinCuts {
    var t: stopwatch;
    t.start();

    const nF         = data.numFeatures;
    const nCuts      = MAX_BINS - 1;       // 254 cut-points → 255 bins
    const compression = 500.0;             // k1 max centroids ≈ 250 > 254? —
                                           // tail singletons push above C/2,
                                           // so 500 is comfortably sufficient

    // ------------------------------------------------------------------
    // Step 1 & 2: Each locale builds one digest per feature from ALL
    // its local rows, then PUTs the centroid arrays to locale 0.
    // ------------------------------------------------------------------
    var allCents  : [0..#numLocales, 0..#nF, 0..#MAX_CENTS] Centroid;
    var allNCents : [0..#numLocales, 0..#nF] int;
    var allWeights: [0..#numLocales, 0..#nF] real;

    coforall loc in Locales with (ref allCents, ref allNCents, ref allWeights) {
      on loc {
        const localRows = data.XDom.localSubdomain().dim(0);
        const nLocal    = localRows.size;

        var localCents  : [0..#nF, 0..#MAX_CENTS] Centroid;
        var localNCents : [0..#nF] int;
        var localWeights: [0..#nF] real;

        // Extract each feature column into a 0-indexed buffer and digest it
        var col: [0..#nLocal] real;
        for f in 0..#nF {
          forall (s, i) in zip(0..#nLocal, localRows) do
            col[s] = data.X[i, f];
          buildDigest(col, compression,
                      localCents[f, ..], localNCents[f], localWeights[f]);
        }

        // PUT results to locale 0's storage
        allCents[loc.id, .., ..]  = localCents;
        allNCents[loc.id, ..]     = localNCents;
        allWeights[loc.id, ..]    = localWeights;
      }
    }

    // ------------------------------------------------------------------
    // Step 3: Locale 0 merges digests and extracts cut-points.
    // ------------------------------------------------------------------
    var cuts: [0..#nF, 0..#nCuts] real;

    on Locales[0] {
      // Scratch buffers — reused across features
      var totalBuf : [0..#(numLocales * MAX_CENTS)] Centroid;
      var merged   : [0..#MAX_CENTS] Centroid;
      var mergedN  : int;

      for f in 0..#nF {
        // Gather centroid contributions for this feature
        var nTotal = 0;
        var totalW = 0.0;
        for l in 0..#numLocales {
          const nc = allNCents[l, f];
          for c in 0..#nc {
            totalBuf[nTotal] = allCents[l, f, c];
            nTotal += 1;
          }
          totalW += allWeights[l, f];
        }

        mergeAndCompress(totalBuf, nTotal, totalW, compression,
                         merged, mergedN);

        logTrace("binning: feature=" + f:string
               + " centroids=" + mergedN:string
               + " totalW=" + totalW:string);

        for b in 0..#nCuts {
          const q  = (b + 1): real / MAX_BINS: real;
          cuts[f, b] = digestQuantile(merged, mergedN, totalW, q);
        }
      }
    }

    // ------------------------------------------------------------------
    // Step 4: Each locale bins its own rows using a local copy of cuts.
    // ------------------------------------------------------------------
    coforall loc in Locales with (ref data) {
      on loc {
        var localCuts: [0..#nF, 0..#nCuts] real = cuts;
        const localDom = data.XDom.localSubdomain();
        forall (i, f) in localDom with (ref data) {
          data.Xb[i, f] = findBin(data.X[i, f], localCuts, f, nCuts): uint(8);
        }
      }
    }

    t.stop();
    logInfo("binning: elapsed=" + t.elapsed():string + "s"
          + " compression=" + compression:string
          + " locales=" + numLocales:string
          + " cuts/feature=" + nCuts:string);

    var bc = new BinCuts(nFeatures=nF);
    bc.values = cuts;
    return bc;
  }

  // ------------------------------------------------------------------
  // applyBins
  //
  // Applies training bin cuts to a new (e.g. test) dataset.
  // data.Xb is populated in-place; data.X must already be filled.
  // ------------------------------------------------------------------
  proc applyBins(ref data: GBMData, bc: BinCuts) {
    coforall loc in Locales with (ref data) {
      on loc {
        var localCuts: [0..#bc.nFeatures, 0..#bc.nCuts] real = bc.values;
        const localDom = data.XDom.localSubdomain();
        forall (i, f) in localDom with (ref data) {
          data.Xb[i, f] = findBin(data.X[i, f], localCuts, f, bc.nCuts): uint(8);
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // findBin — binary search over cut-points for a single value.
  //
  // Returns bin index in 0..MAX_BINS-1.
  // Bin b spans [cuts[f, b-1], cuts[f, b]).
  // ------------------------------------------------------------------
  inline proc findBin(v: real, cuts: [?D] real, f: int, nCuts: int): int {
    var lo = 0;
    var hi = nCuts;
    while lo < hi {
      const mid = (lo + hi) / 2;
      if v < cuts[f, mid] then hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }

} // module Binning
