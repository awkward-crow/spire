/*
  Binning.chpl
  ------------
  Quantile binning: converts the float64 feature matrix X into a uint8
  bin matrix Xb.  Done once at startup before any boosting rounds.

  Algorithm: sampling.
    1. Each locale randomly samples sqrt(nLocalRows) rows.
    2. Sampled values are gathered to locale 0.
    3. Locale 0 sorts each feature column and picks MAX_BINS-1 evenly-
       spaced quantile cut-points.
    4. Cut-points are broadcast; each locale bins its own rows locally.

  Alternatives considered (see notes.md):
    - Greenwald-Khanna: epsilon-exact streaming quantiles, complex merge.
    - t-digest: better tail accuracy, moderate complexity.
  Sampling matches XGBoost/LightGBM defaults; bin boundary error is
  self-correcting across boosting rounds.
*/

module Binning {

  use DataLayout;
  use Random;
  use Sort;
  use Math;

  param MAX_BINS = 255;   // bins per feature; uint(8) covers 0..254

  // ------------------------------------------------------------------
  // computeBins
  //
  // Populates data.Xb with bin indices in 0..MAX_BINS-1.
  // Call once before training begins.
  // ------------------------------------------------------------------
  proc computeBins(ref data: GBMData, seed: int = 42) {
    const nF    = data.numFeatures;
    const nCuts = MAX_BINS - 1;   // 254 cut-points → 255 bins

    // ------------------------------------------------------------------
    // Step 1: Decide how many rows each locale will sample
    // ------------------------------------------------------------------
    var samplesPerLocale: [0..#numLocales] int;
    coforall loc in Locales with (ref samplesPerLocale) {
      on loc {
        const nLocal = data.XDom.localSubdomain().dim(0).size;
        samplesPerLocale[loc.id] = max(1, sqrt(nLocal: real): int);
      }
    }

    const totalSamples = + reduce samplesPerLocale;

    // Prefix-sum offsets so each locale writes to a unique row slice
    var offsets: [0..#numLocales] int;
    for l in 1..<numLocales do
      offsets[l] = offsets[l-1] + samplesPerLocale[l-1];

    // ------------------------------------------------------------------
    // Step 2: Each locale fills its slice of the gather buffer (PUT to locale 0)
    // ------------------------------------------------------------------
    var gathered: [0..#totalSamples, 0..#nF] real;

    coforall loc in Locales {
      on loc {
        const nSamp    = samplesPerLocale[loc.id];
        const off      = offsets[loc.id];
        const rowRange = data.XDom.localSubdomain().dim(0);
        var   rng      = new randomStream(real, seed = seed + loc.id);

        for s in 0..#nSamp {
          const row = rowRange.low +
                      (rng.next() * rowRange.size: real): int;
          for f in 0..#nF do
            gathered[off + s, f] = data.X[row, f];
        }
      }
    }

    // ------------------------------------------------------------------
    // Step 3: Sort each feature column and pick quantile cut-points
    // ------------------------------------------------------------------
    var cuts: [0..#nF, 0..#nCuts] real;

    on Locales[0] {
      var col: [0..#totalSamples] real;
      for f in 0..#nF {
        for s in 0..#totalSamples do col[s] = gathered[s, f];
        sort(col);
        for b in 0..#nCuts {
          const pos = ((b + 1): real / MAX_BINS: real
                        * totalSamples: real): int;
          cuts[f, b] = col[min(pos, totalSamples - 1)];
        }
      }
    }

    // ------------------------------------------------------------------
    // Step 4: Each locale bins its own rows using a local copy of cuts
    // ------------------------------------------------------------------
    coforall loc in Locales with (ref data) {
      on loc {
        // Bulk-fetch cuts once to avoid per-sample remote GETs in the hot loop
        var localCuts: [0..#nF, 0..#nCuts] real = cuts;
        const localDom = data.XDom.localSubdomain();
        forall (i, f) in localDom with (ref data) {
          data.Xb[i, f] = findBin(data.X[i, f], localCuts, f, nCuts): uint(8);
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
