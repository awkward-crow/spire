/*
  TDigest.chpl
  ------------
  Compact quantile sketching via the t-digest algorithm (Dunning 2019).
  Used by Binning to compute accurate quantile cut-points from distributed
  feature data without centralising all values on locale 0.

  Key property (k1 scale function):
    The maximum weight of a centroid at normalised rank q is proportional
    to sqrt(q·(1−q)).  Tail centroids (q near 0 or 1) are forced small —
    often single points — giving high resolution exactly where quantile
    regression (Pinball loss) needs it most.

  Public API
  ----------
    buildDigest(col, compression, ref cents, ref nCents, ref totalW)
      Compress a column of reals into at most MAX_CENTS centroids.
      Sorts a local copy; does not modify col.

    mergeAndCompress(allCents, nTotal, totalW, compression,
                     ref outCents, ref outN)
      Merge centroid arrays from multiple locales (concatenated into
      allCents[0..#nTotal]) and re-compress using the same k1 criterion.

    digestQuantile(cents, nCents, totalW, q): real
      Linear-interpolation quantile query.  cents must be sorted by
      mean (guaranteed by buildDigest / mergeAndCompress).

  Choosing compression
  --------------------
  The k1 total range is compression/2, so in the worst case there are
  at most compression/2 "full-size" centroids.  Tail singletons add
  more entries beyond that bound.  For 254 cut-points use compression
  ≥ 500; MAX_CENTS = 1024 is a safe ceiling up to compression = 1000.
*/

module TDigest {

  use Sort;
  use Math;

  param MAX_CENTS = 1024;

  // ------------------------------------------------------------------
  // Centroid — weighted mean summarising a cluster of data points.
  // ------------------------------------------------------------------
  record Centroid {
    var mean  : real;
    var weight: real;
  }

  record ByCentMean : keyComparator {
    proc key(c: Centroid) { return c.mean; }
  }

  // ------------------------------------------------------------------
  // k1 scale:  k1(q) = C/(2π) · arcsin(2q − 1)
  //
  // Its derivative C/(π·sqrt(q(1−q))) is large at the tails, which
  // tightens the centroid-size limit there — the tail-accuracy property.
  // ------------------------------------------------------------------
  private inline proc k1(q: real, C: real): real {
    const qc = max(1e-10, min(1.0 - 1e-10, q));
    return C / (2.0 * pi) * asin(2.0 * qc - 1.0);
  }

  // ------------------------------------------------------------------
  // buildDigest
  //
  // Sort col once, then make a single left-to-right greedy pass.
  // At each point: absorb into the last centroid if doing so keeps its
  // k1-span under 1; otherwise open a new centroid.
  //
  // Outputs (caller allocates cents with size ≥ MAX_CENTS):
  //   cents     — produced centroids, sorted by mean
  //   nCents    — number of valid entries in cents
  //   totalW    — total weight (= col.size for unweighted input)
  // ------------------------------------------------------------------
  proc buildDigest(
      col        : [] real,
      compression: real,
      ref cents  : [] Centroid,
      ref nCents : int,
      ref totalW : real
  ) {
    const n = col.size;
    if n == 0 { nCents = 0; totalW = 0.0; return; }

    totalW = n: real;

    var sorted: [0..#n] real = col;
    sort(sorted);

    nCents = 0;
    var cumW: real = 0.0;

    for i in 0..#n {
      const x = sorted[i];

      if nCents == 0 {
        cents[0] = new Centroid(mean=x, weight=1.0);
        nCents   = 1;
        cumW     = 1.0;
        continue;
      }

      // k1-span of the last centroid after absorbing x:
      //   left  edge (normalised) = (cumW − prevW) / totalW
      //   right edge after merge  = (cumW + 1)     / totalW
      const prevW  = cents[nCents - 1].weight;
      const qLeft  = (cumW - prevW) / totalW;
      const qRight = (cumW + 1.0)   / totalW;
      const spanOk = k1(qRight, compression) - k1(qLeft, compression) < 1.0;

      if spanOk && nCents < MAX_CENTS {
        const w2 = prevW + 1.0;
        cents[nCents - 1].mean   = (cents[nCents - 1].mean * prevW + x) / w2;
        cents[nCents - 1].weight = w2;
      } else {
        if nCents >= MAX_CENTS then break;
        cents[nCents] = new Centroid(mean=x, weight=1.0);
        nCents       += 1;
      }
      cumW += 1.0;
    }
  }

  // ------------------------------------------------------------------
  // mergeAndCompress
  //
  // Merges centroid arrays from multiple locales and re-compresses.
  // Called by locale 0 after gathering per-locale digests.
  //
  // allCents[0..#nTotal] holds all centroids concatenated; totalW is
  // their combined weight (denominator for normalised ranks).
  // ------------------------------------------------------------------
  proc mergeAndCompress(
      allCents    : [] Centroid,
      nTotal      : int,
      totalW      : real,
      compression : real,
      ref outCents: [] Centroid,
      ref outN    : int
  ) {
    if nTotal == 0 { outN = 0; return; }

    var tmp: [0..#nTotal] Centroid = allCents[0..#nTotal];
    sort(tmp, comparator = new ByCentMean());

    outN = 0;
    var cumW: real = 0.0;

    for i in 0..#nTotal {
      const c = tmp[i];

      if outN == 0 {
        outCents[0] = c;
        outN        = 1;
        cumW        = c.weight;
        continue;
      }

      const prevW  = outCents[outN - 1].weight;
      const qLeft  = (cumW - prevW)    / totalW;
      const qRight = (cumW + c.weight) / totalW;
      const spanOk = k1(qRight, compression) - k1(qLeft, compression) < 1.0;

      if spanOk && outN < MAX_CENTS {
        const w2 = prevW + c.weight;
        outCents[outN - 1].mean   = (outCents[outN - 1].mean * prevW
                                      + c.mean * c.weight) / w2;
        outCents[outN - 1].weight = w2;
      } else {
        if outN >= MAX_CENTS then break;
        outCents[outN] = c;
        outN          += 1;
      }
      cumW += c.weight;
    }
  }

  // ------------------------------------------------------------------
  // digestQuantile
  //
  // Estimate the value at normalised rank q ∈ [0, 1] by linear
  // interpolation between adjacent centroid centres.
  //
  // Centroid i is centred at cumulative-weight position
  //   c_i = sum(w_0 .. w_{i-1}) + w_i / 2
  // We find the pair (i-1, i) that straddles target = q · totalW
  // and interpolate linearly between their means.
  // ------------------------------------------------------------------
  proc digestQuantile(
      cents  : [] Centroid,
      nCents : int,
      totalW : real,
      q      : real
  ): real {
    if nCents == 0 then return 0.0;
    if nCents == 1 then return cents[0].mean;

    const target = q * totalW;
    var   cumW   : real = 0.0;

    for i in 0..#nCents {
      const ci = cumW + cents[i].weight / 2.0;

      if target <= ci {
        if i == 0 then return cents[0].mean;
        // ciPrev: centre of centroid i-1.  cumW already includes w_{i-1}.
        const ciPrev = cumW - cents[i - 1].weight / 2.0;
        if ci <= ciPrev then return cents[i].mean;   // degenerate
        const t = (target - ciPrev) / (ci - ciPrev);
        return cents[i - 1].mean + t * (cents[i].mean - cents[i - 1].mean);
      }

      cumW += cents[i].weight;
    }

    return cents[nCents - 1].mean;
  }

} // module TDigest
