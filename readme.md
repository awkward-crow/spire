# spire -- gradient boosting machines

    claude --resume 750c1825-187f-47bf-ba62-595de9b3086e

    just say "performance profiling"

## latest

 - much improved performance, see section below

---

 - quantile regression on bicycle data
  = Records (MSE, LogLoss, Pinball) replacing the enum Objective + dispatch chains
  = GBMEnsemble bundling trees + baseScore
  = BoosterConfig stripped of tau and minHess
  = boost() generic via duck typing, predict() taking GBMEnsemble
  = t-digest binning replacing random sampling
  = Pinball hessian fixed to tau*(1-tau)

And see `refactor.md`, objectives mature from an enum to separate records.

## next steps

0.1 try california housing data with multi-locale! and other examples!!

1. **Min-split-gain pruning** — add `minGain: real = 0.0` to `BoosterConfig`, check `gain > cfg.minGain` in `findBestSplits`. Single-field change, very low cost.
2. **Early stopping** — add `valData` and `earlyStoppingRounds` to `BoosterConfig`; track best validation loss in `boost` and halt early.
3. **Row/column subsampling** — both LightGBM and XGBoost default to subsampling; omitting it widens the accuracy gap on noisy datasets.
4. **Histogram memory layout benchmark** — `[node, feature, bin]` vs `[feature, bin, node]`.
5. **Leaf-wise growth** — larger change; batched leaf-wise is the Chapel-native angle.
6. **Missing value handling** — needed for most real-world datasets beyond California Housing.

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

## see also

 - file `notes.md`
 - file `chapel_arkouda_gbm_conversation.md`
 - file `docker.md`


### end
