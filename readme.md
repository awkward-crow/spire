# spire -- gradient boosting machines

claude --resume c2633297-e74c-42f2-b026-ef54b31e259a

## latest

 - float32 gradient quantization: y, grad, hess, histogram bins all real(32); F stays real(64)
 - parallel CSV loading: 4× speedup on SUSY (60s → 15s), see section below
 - column subsampling, see section below
 - much improved performance, see section below
 - quantile regression, see section below

## next steps

The primary thread is closing the speed gap with LightGBM while building
toward correct and efficient multi-locale execution.  Current baseline:
CoverType (396 k × 54, 100 trees, numLeaves=31): Chapel 69 s, LightGBM 0.95 s.
Accuracy is within 0.3% — the gap is entirely in the histogram kernel.

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

3. **Batched leaf-wise** — instead of one split per histogram pass, pick the top-k
   highest-gain active leaves and expand them all in a single sample pass.  Each locale
   scans its rows once, scattering into k smaller-child slots simultaneously; k
   subtractions then derive the k larger children.  Cost: O(N) sample reads for k splits
   instead of O(k × N).  Also reduces the number of `coforall` barriers per tree from
   O(numLeaves) to O(numLeaves / k) — the key win for multi-locale where barrier cost
   scales with locale count.

4. **Pre-sorted sample indices** — for each feature, pre-sort sample indices by bin
   value and store as a `uint32[]` index array (one per feature, distributed).  The
   histogram scatter then accesses bins in monotonically increasing order, improving
   cache reuse on the histogram array.  One-time preprocessing cost, amortized over all
   trees.

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
