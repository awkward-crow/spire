# spire -- gradient boosting machines

"continue the histogram optimization work"

claude --resume "chapel-gbm-histogram-optimization"

## latest

 - histogram AoS layout + 4-sample unrolled C kernel: CoverType 7.4s → 1.74s (4.3×), SUSY 27.1s → 11.9s (2.3×)
 - column-major Xb: transposed Xb to [nF, nSamples] → stride-1 histogram reads; CoverType 8.4s → 7.4s (14%), SUSY 28.8s → 27.1s (6%)
 - batched leaf-wise: batchSize=4 → 3× fewer sample passes per tree (5 vs 15 for numLeaves=16)
 - float32 gradient quantization: y, grad, hess, histogram bins all real(32); F stays real(64)
 - parallel CSV loading: 4× speedup on SUSY (60s → 15s), see section below

## next steps

The primary thread is closing the speed gap with LightGBM while building
toward correct and efficient multi-locale execution.  Current baselines
(20 trees, numLeaves=16):

| Dataset | Chapel | LightGBM | Gap |
|---------|--------|----------|-----|
| SUSY (5M × 18) | 11.9 s | 4.16 s | 2.9× |
| CoverType (396k × 54) | 1.74 s | 0.39 s | 4.5× |

Accuracy within 0.1% of LightGBM in both cases.  Gap is entirely in the
histogram kernel (random scatter writes); CoverType gap larger due to more features.

### Performance / multi-locale path (ordered)

1. ~~**Fix histogram remote GETs**~~ — done.  `buildHistogramsNode` now uses
   `coforall loc in Locales`, each locale accumulating into a local `real(32)`
   partial histogram over its `localSubdomain()` before reducing to locale 0.

2. ~~**Gradient quantization**~~ — done.  `y`, `grad`, `hess`, and all histogram
   bins are `real(32)`; `F` stays `real(64)` for prediction accuracy.  Halves
   per-sample read bandwidth in the scatter loop and halves the multi-locale
   reduction payload (~55 KB per locale per split vs ~110 KB).  Single-locale
   training time is within noise of float64 (bottleneck is random histogram
   writes, not sequential grad reads); multi-locale benefit will be larger.

   Also: interleave grad and hess a la lightGBM for a bit of cache localization.

   Questions:
    - branching in inline proc sigmoid?

3. ~~**Batched leaf-wise**~~ — done.  `batchSize: int = 4` in `BoosterConfig`.
   `buildHistogramsNodes` (Histogram.chpl) accumulates k smaller children in one
   sample pass via a `nodeToSlot` lookup and `lg[f, b, slot]` local accumulators.
   `updateNodeAssignBatch` (Tree.chpl) routes all k splits in one coforall pass.
   numLeaves=16, batchSize=4: 5 sample passes per tree instead of 15 (3× fewer
   coforall barriers); numLeaves=31: ~8 instead of 30 (≈3.5×).

4. ~~**Pre-sorted sample indices**~~ — implemented and reverted.  4.5× regression on
   both benchmarks (SUSY: 28.8 s → 130 s, CoverType: 8.4 s → 38 s).
   `lg[f, *, *]` is only 4 KB and lives in L1 — the histogram scatter was never
   cache-thrashing.  Sorting by bin converts sequential reads of `nodeId`, `grad`,
   `hess` (hardware-prefetchable at N=4M) into random L3 misses; that cost dwarfs
   any histogram benefit.

4b. ~~**Column-major Xb**~~ — done.  Transposed `Xb` to `[numFeatures, numSamples]`
   with a 1×numLocales locale grid so `Xb[f, localRows]` stays local.  Histogram
   inner loop (fixed f, sequential i) is now stride-1.  CoverType: 8.4 s → 7.4 s
   (14%); SUSY: 28.8 s → 27.1 s (6%).  Gain is real but modest — `forall f`
   parallelism was already partially amortizing row-major waste by having all
   feature tasks share the same cache lines per row.  Changes: new `XbDom` in
   DataLayout.chpl; index flip in Binning, Histogram, Tree (~15 sites).

5. **SIMD prefix scan in split finding** — the 255-bin prefix scan in `findBestSplitsNodes`
   is the natural AVX2 target: sequential, no scatter, fits in L1 cache.  Implement as
   an `extern` C function using `_mm256` horizontal prefix sums; expect 2–4×.  Single-locale
   only — runs on locale 0 after the reduction.  Check whether `CHPL_TARGET_CPU=native`
   already auto-vectorizes this before writing intrinsics.

### Remaining features

- **Row subsampling** — `rowsampleByTree` in `BoosterConfig`; reduces sample scan cost
  proportionally and aids generalisation on noisy datasets.

- **Early stopping** — halt training when held-out validation loss stops improving for
  `earlyStoppingRounds` consecutive trees.  Requires a validation split passed to `boost`.

- **Min-split-gain pruning** — `minGain: real = 0.0` in `BoosterConfig`; check
  `gain > cfg.minGain` in `findBestSplitsNodes`.  Regularisation knob, not a speed win.

- **Missing value handling** — required for most real-world datasets beyond the current examples.

- **Parallel CSV loading** — done.  `readCSV` now divides the file into
  `here.maxTaskPar` byte-range chunks, aligns each to the nearest newline boundary,
  counts rows in parallel (pass 1), allocates once, then parses floats in parallel
  (pass 2).  On SUSY (5M rows, 18 features): 60s serial → 14.8s parallel on 4 cores
  (4× speedup).  All 119 tests pass; accuracy unchanged.

## usage/tests

```sh
cd test
make               # build all tests
make run           # build and run all tests
```

To run a single test,

```sh
make TestObjectives  
./build/TestObjectives
```

Or, after compiling for multi-locale, e.g. start a shell in docker with the project root as its working directory,

```sh
cd test
make TestObjectives
./build/TestObjectives -nl 4
```

### logging

```sh
./build/TestBooster -logLevel=INFO 2>&1 | less -X
```

or log level `TRACE`.

### clean up

```sh
make clean
```

## `CHPL_TARGET_CPU=native`

By default tests are compiled with `CHPL_TARGET_CPU=native`, optimising for
the build machine's CPU.  On a cluster with a specific microarchitecture,
override this:

```sh
make CHPL_TARGET_CPU=broadwell
```

## performance

### tldr; `--fast` default + histogram parallelism rewrite

### --fast

`examples/Makefile` now compiles with `--fast` by default (removes Chapel's
nil/bounds/overflow checks).  Use `DEBUG=1` to restore checks, `PROFILE=1`
for a `--fast -g` profiling build.

### build histograms

`buildHistograms` was re-parallelised over features instead of samples.  The
old `forall i in samples with (+ reduce accumGrad, + reduce accumHess)` pattern
allocated per-task copies of the full `[nodes × features × bins]` histogram
(~2 MB each) on every call — ~22 GB of allocations over a 100-tree run.  The
new loop is `forall f in 0..#nF with (ref hist)`: each task owns a disjoint
`[*, f, *]` slice, so no copies and no reduce are needed.

| Example | Before | After | Speedup |
|---------|--------|-------|---------|
| CaliforniaHousing (16 k samples, depth 6) | 38.8 s | 0.66 s | 59× |
| Bicycle (14 k samples, depth 4, 2 quantiles) | 8.7 s | 0.57 s | 15× |
| BreastCancer (455 samples, depth 4) | 9.8 s | 0.11 s | 89× |

All outputs are numerically identical; 119/119 tests pass.

## column subsampling

 - column subsampling (`colsampleByTree`) — partial Fisher-Yates per tree,
   single persistent RNG advanced across all trees; exposed as config const
   in all example drivers.  Timing on CoverType (495 k × 54, 50 trees, depth 4):

   | colsample | wall time | test log-loss |
   |-----------|-----------|---------------|
   | 1.0       | 35 s      | 0.4337        |
   | 0.8       | 32 s      | 0.4413        |
   | 0.6       | 26 s      | 0.4636        |
   | 0.4       | 22 s      | 0.4957        |

   Training time scales roughly linearly with colsample.  This dataset has
   mostly informative features so subsampling hurts accuracy; on wider datasets
   with redundant features it will help.

## quantile regression ...

.. on bicycle data; also

  - Records (MSE, LogLoss, Pinball) replacing the enum Objective + dispatch chains
  - GBMEnsemble bundling trees + baseScore
  - BoosterConfig stripped of tau and minHess
  - boost() generic via duck typing, predict() taking GBMEnsemble
  - t-digest binning replacing random sampling
  - Pinball hessian fixed to tau*(1-tau)

And see `refactor.md`, objectives mature from an enum to separate records.

## see also

 - file `notes.md` in particular `open questions: leaf-wise growth, distributed angle`
 - file `chapel_arkouda_gbm_conversation.md`
 - file `docker.md`


### end
