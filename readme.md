# spire -- gradient boosting machines

claude --resume c2633297-e74c-42f2-b026-ef54b31e259a

in ./examples,

```sh
python -u save_susy.py
```

## latest

 - column subsampling, see section below
 - much improved performance, see section below
 - quantile regression on bicycle data
  = Records (MSE, LogLoss, Pinball) replacing the enum Objective + dispatch chains
  = GBMEnsemble bundling trees + baseScore
  = BoosterConfig stripped of tau and minHess
  = boost() generic via duck typing, predict() taking GBMEnsemble
  = t-digest binning replacing random sampling
  = Pinball hessian fixed to tau*(1-tau)

And see `refactor.md`, objectives mature from an enum to separate records.

## next steps

Ordered by performance impact. And try examples with multi-locale!

1. **Histogram subtraction trick** — implemented in `buildHistogramsLeft` +
   `subtractSiblings` but currently slower on the existing examples (CaliforniaHousing
   1.21 s vs 0.66 s, Bicycle 0.81 s vs 0.57 s).  After the feature-parallel rewrite
   each sample pass is so cheap that the extra conditional, per-feature zeroing, and
   sibling-subtraction arithmetic costs more than it saves.  The trick is designed for
   the regime where sample passes are expensive — large datasets or many features.
   The crossover point is above ~20 k samples × 12 features; leave it in and revisit
   when larger data is available.  Candidate datasets (all public):
   - **Cover Type** — 581 k × 54, binary classification; `sklearn.datasets.fetch_covtype()`
   - **Year Prediction MSD** — 515 k × 90, regression; UCI ML Repository
   - **HIGGS** — 11 M × 28, binary classification; standard GBM stress test
   - **SUSY** — 5 M × 18, binary classification; UCI ML Repository

2. **Histogram memory layout** — done: `[feature, bin, node]` adopted.  6% faster
   than `[node, feature, bin]` on CoverType (31.7 s vs 33.7 s); stride-1 node access
   in the scatter writes is the win.

3. **Parallelise `findBestSplits`** — tried `forall node` instead of `for node`;
   slower at both depth 4 (31 nodes) and depth 6 (127 nodes) on CoverType.  Per-node
   work (54 features × 255 bins = 13.8 k iterations) is too cheap relative to
   Qthreads task-creation overhead.  Not worth doing until there are many more nodes
   (deeper trees or leaf-wise growth with many leaves) or a heavier inner loop
   (more features/bins).

4. **Column subsampling** — done: `colsampleByTree` in `BoosterConfig`, partial
   Fisher-Yates with a single persistent RNG.  Training time scales linearly
   with colsample.  See latest section for numbers.

5. **Row subsampling** — sample a fraction of rows per tree.  Reduces the inner
   `for i in data.rowDom` loop proportionally; aids generalisation.

6. **Leaf-wise growth** — split only the highest-gain leaf each round instead of
   the whole depth level.  Fewer total histogram builds for the same number of
   leaves; changes scaling behaviour.  Larger implementation effort.

7. **Early stopping** — halt training when validation loss stops improving.
   Training-cost feature; no impact on per-tree speed.

8. **Min-split-gain pruning** — add `minGain: real = 0.0` to `BoosterConfig`.
   `findBestSplits` is already fast; this is a regularisation knob, not a
   performance win.

9. **Missing value handling** — needed for most real-world datasets beyond the
   examples here.

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

## see also

 - file `notes.md` in particular `open questions: leaf-wise growth, distributed angle`
 - file `chapel_arkouda_gbm_conversation.md`
 - file `docker.md`


### end
