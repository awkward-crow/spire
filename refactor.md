# OOP Objectives Refactor

## What changed and why

### Before: enum + dispatch chains

The original design used `enum Objective { MSE, LogLoss, Pinball }` with parallel
`select` chains spread across two files:

- `Objectives.chpl` — `computeGradients()` dispatched to `computeGradients_MSE`,
  `computeGradients_LogLoss`, `computeGradients_Pinball`, and `computeLoss()` did the same
- `Booster.chpl` — `initF()` had its own `select obj { when MSE ... when Pinball ... }`

Objective-specific parameters leaked into `BoosterConfig`:
- `tau: real` — only meaningful for Pinball
- `minHess: real` — meaningful for all, but the right default differs per objective
  (LogLoss hessian is `p*(1-p)` which can be tiny; using `minHess=1.0` for LogLoss
  over-prunes nodes where the model is confident)

`boost()` returned an awkward `([] FittedTree, real)` tuple because the base score
computed inside `initF` had to be threaded out to callers for use in `predict()`.

### After: records + duck typing

Each objective is now a Chapel **record** with four methods:

```chapel
record MSE {
  proc initF(ref data: GBMData): real    // set data.F = mean(y), return it
  proc gradients(F, y, ref g, ref h)     // F-y, hess=1
  proc loss(F, y): real                  // mseLoss(F, y)
  proc defaultMinHess(): real            // 1.0
}

record LogLoss {
  proc initF(ref data: GBMData): real    // log-odds of mean(y)
  proc gradients(F, y, ref g, ref h)     // sigmoid(F)-y, p*(1-p)
  proc loss(F, y): real                  // logLoss(F, y)
  proc defaultMinHess(): real            // 1e-6  ← changed from 1.0
}

record Pinball {
  var tau: real                          // carries its own quantile level
  proc initF(ref data: GBMData): real    // tau-quantile of y
  proc gradients(F, y, ref g, ref h)     // -tau / (1-tau), hess=1
  proc loss(F, y): real                  // pinballLoss(F, y, tau)
  proc defaultMinHess(): real            // 1.0
}
```

`boost()` is now generic over the objective type:

```chapel
proc boost(ref data: GBMData, obj: ?T, cfg: BoosterConfig): GBMEnsemble
```

`GBMEnsemble` bundles the fitted trees with the base score so callers never
thread the scalar manually:

```chapel
record GBMEnsemble {
  var trees    : [] FittedTree;
  var baseScore: real;
}
```

`BoosterConfig` loses `tau` and `minHess` — both now live inside the objective.

`predict()` takes the ensemble directly:

```chapel
proc predict(ensemble: GBMEnsemble, data: GBMData): [] real
```

### Call-site translation

| Before | After |
|--------|-------|
| `boost(data, Objective.MSE, cfg)` → `(trees, base)` | `boost(data, new MSE(), cfg)` → `ensemble` |
| `boost(data, Objective.Pinball, cfg)` (cfg.tau=0.9) | `boost(data, new Pinball(tau=0.9), cfg)` → `ensemble` |
| `predict(trees, data, base)` | `predict(ensemble, data)` |
| `computeGradients(Objective.MSE, F, y, g, h)` | `(new MSE()).gradients(F, y, g, h)` |
| `cfg.tau = 0.1` | gone — tau lives in the Pinball instance |
| `cfg.minHess = 1.0` | gone — `obj.defaultMinHess()` is called inside boost |

---

## Why duck typing instead of a Chapel interface

The natural OOP move is to declare a `GBMObjective` interface and constrain `boost()`
with `where T implements GBMObjective`.  We tried this:

```chapel
interface GBMObjective {
  proc Self.initF(ref data: GBMData): real;
  proc Self.gradients(F: [] real, y: [] real, ref g: [] real, ref h: [] real);
  proc Self.loss(F: [] real, y: [] real): real;
  proc Self.defaultMinHess(): real;
}
```

Chapel rejected it:

```
error: the interface function gradients contains a where clause,
       which is currently not supported
```

The problem: `[] real` in an interface method signature is a *generic* array type.
Chapel's constrained-generics implementation (Chapel 2.x interfaces) generates an
implicit `where` clause for any generic parameter, and `where` clauses inside
interface declarations are not yet supported.

The alternative — changing `gradients` and `loss` to take `GBMData` instead of raw
arrays — would work around the type system issue but would force the unit tests
(which exercise gradients on small hand-crafted arrays) to construct `GBMData`
objects, adding noise to tests that are deliberately low-level.

**Duck typing** is the right call here because:

1. **All three call sites are inside this codebase.** There is no external implementor
   that needs the interface to know what to write.  The contract is documented in
   comments and enforced at compile time: if a type passed to `boost()` is missing
   `initF` or `gradients`, Chapel will error at the call site with a clear
   "no matching proc" message.

2. **Chapel specialises boost() statically.** The compiler generates a separate
   instantiation for `MSE`, `LogLoss`, and `Pinball` — identical in performance
   to what a formal interface would produce.  There is no runtime dispatch.

3. **We can add the interface later** when Chapel lifts the where-in-interface
   restriction, without changing any call sites.

---

## Files changed

| File | Change |
|------|--------|
| `src/Objectives.chpl` | Removed enum + dispatch procs; added MSE, LogLoss, Pinball records; moved Sort import here |
| `src/Booster.chpl` | Added GBMEnsemble; simplified BoosterConfig (removed tau, minHess); rewrote boost() and predict() |
| `test/TestObjectives.chpl` | `computeGradients(Objective.X, ...)` → `(new X()).gradients(...)` |
| `test/TestBooster.chpl` | `Objective.X` → `new X()`; removed tau/minHess from cfg |
| `test/TestPredict.chpl` | `(trees, base) = boost(...)` → `ensemble = boost(...)`; updated predict calls |
| `test/TestTree.chpl` | `computeGradients(Objective.MSE, ...)` → `(new MSE()).gradients(...)` |
| `test/TestHistogram.chpl` | Same |
| `examples/CaliforniaHousing.chpl` | `new MSE()`, GBMEnsemble pattern |
| `examples/Bicycle.chpl` | `new Pinball(tau=x)`, GBMEnsemble pattern |

**Unchanged:** Splits, Tree, Binning, Histogram, DataLayout, CSVReader, Logger,
TestBinning, TestSplits, TestCSVReader.

---

## Implementation notes

### Chapel interface limitation (first compile attempt)

Tried declaring `interface GBMObjective` with method signatures using `[] real`
array parameters.  Compiler error:

```
error: the interface function gradients contains a where clause,
       which is currently not supported
```

### Second compile error: generic array field in record

`var trees: [] FittedTree` as a record field is rejected:

```
error: fields cannot specify generic array types
```

Resolution: added an explicit `treeDom: domain(1)` field and a custom `init`
that sets `treeDom = trees.domain` before copying the array, matching the
pattern used by `FittedTree` in `Tree.chpl`.

### California Housing smoke test (50 trees, depth 4, level-wise)

|          | RMSE train | RMSE test |
|----------|-----------|----------|
| LightGBM | 0.5167    | 0.5361   |
| Chapel   | 0.5210    | 0.5373   |

LightGBM forced to complete binary trees (`num_leaves=16`, `min_child_samples=1`,
`min_split_gain=0.0`) to match Chapel's level-wise growth.  Chapel is within ~1%
on both splits — the remaining gap is likely binning differences (Chapel uses
random quantile bins, LightGBM uses frequency-based bins).

### Bicycle smoke test (50 trees, depth 4, level-wise)

|               | tau=0.1 train | tau=0.1 test | tau=0.9 train | tau=0.9 test | coverage | width |
|---------------|---------------|--------------|---------------|--------------|----------|-------|
| LightGBM      | 10.46         | 10.67        | 11.36         | 11.30        | 82.3%    | 186   |
| Chapel        | 18.30         | 18.18        | 40.30         | 40.17        | 81.7%    | 443   |

Coverage is comparable (~82% vs target 80%), but Chapel's pinball loss is roughly
2× higher for tau=0.1 and 3.5× higher for tau=0.9, and its intervals are ~2.4×
wider.  This is unchanged from pre-refactor, so nothing regressed.

The gap is likely driven by binning: Chapel uses random-sample quantile bins
(fixed seed), while LightGBM uses frequency-based bins optimised for the
training distribution.  With a target variable that spans 0–977 counts/hour
and a heavily skewed distribution (most hours have low counts), poor bin
placement at the tails hurts quantile models disproportionately — the q90
model in particular is trying to predict rare high-count events.

### t-digest binning (implemented, single-locale result unchanged)

Replaced the sqrt(nLocalRows) random-sampling step in Binning.chpl with a
full t-digest pass over all local rows, merged at locale 0.  New files:
`src/TDigest.chpl` (buildDigest, mergeAndCompress, digestQuantile).  `seed`
parameter removed from computeBins() — t-digest is deterministic.

Single-locale Bicycle results after the change: **identical**.

This is expected: with ~13,900 training rows and 12 features, 118 random
samples were already sufficient to capture the X-feature distributions.
The performance gap with LightGBM is not a binning issue.

The t-digest change is still the right investment for multi-locale runs where
each locale sees only n/L rows and sqrt(n/L) would be tiny.  The communication
cost is now O(numLocales × nFeatures × MAX_CENTS) centroids regardless of
dataset size, versus O(numLocales × sqrt(n/L)) raw values before.

### Root cause of Pinball gap vs LightGBM

The Pinball gradient is a constant: −τ when under-predicting, (1−τ) when
over-predicting — no information about the magnitude of the error.  Each leaf's
optimal weight ≈ τ (for an under-predicting leaf), so with eta=0.1 each tree
contributes only ~0.01 update per sample.  After 50 trees the cumulative
correction is small relative to the target-variable scale (0–977 counts/hour).

LightGBM achieves 10.46 on the same data with the same number of trees,
suggesting it handles Pinball convergence more efficiently — possibly via
gradient scaling, a different leaf-weight normalisation, or internal
step-size adaptation.

### Pinball hessian fix (implemented)

Changed `Pinball.gradients()` hessian from `1.0` to `tau*(1-tau)` — the Fisher
information of the asymmetric Laplace distribution.  Also updated
`defaultMinHess()` to return `tau*(1-tau)` to keep the 1-sample-per-leaf
threshold consistent.

Results (50 trees, depth 4):

|          | tau=0.1 test | tau=0.9 test | coverage | width |
|----------|-------------|-------------|----------|-------|
| Before   | 18.18       | 40.17       | 81.8%    | 443   |
| After    | 17.67       | 39.00       | 88.1%    | 440   |
| LightGBM | 10.67       | 11.30       | 82.3%    | 186   |

Modest improvement in loss (~3%), but coverage overshot (88% vs 80% target).

The gap persists because even with the correct hessian the leaf value for a
pure under-predicting leaf is `tau/(tau*(1-tau)) = 1/(1-tau) ≈ 1.11` — still a
bounded constant, independent of actual residual magnitude.  The optimal leaf
update for Pinball is the tau-quantile of `y_i − F_i` within the leaf, which
can be 100s of counts; the histogram-based gradient approach cannot see this.

The stale comment in `applyTree` (claiming eta was applied there) was also
fixed — eta is baked into leaf values at `recordLevel`/`finalizeLeaves` time.

### Leaf refit (implemented, gap closed)

Replaced Newton-step leaf values for Pinball with the tau-quantile of per-leaf
residuals `y_i - F_i` — LightGBM's `RenewTreeOutput()` approach.  New method
`leafRefit` added to all three objectives; MSE and LogLoss are no-ops.  Called
in `boost()` after `finalizeLeaves`, before `applyTree`, while `nodeId` is
still in scope.

Results (50 trees, depth 4):

|          | tau=0.1 test | tau=0.9 test | coverage | width |
|----------|-------------|-------------|----------|-------|
| Newton step (before) | 18.18 | 40.17 | 88.1% | 440 |
| Leaf refit (after)   | 10.58 | 11.51 | 81.9% | 187 |
| LightGBM             | 10.67 | 11.30 | 82.3% | 186 |

Chapel now matches LightGBM within ~1% on both loss values.  Coverage is on
target (81.9% vs 80% target vs 82.3% LightGBM).  Interval width 187 vs 186.

New file `leaf-refit.md` documents the design and multi-locale gather pattern.

---

### Interface limitation detail

Root cause: `[] real` is a generic type; Chapel implicitly adds a `where` clause
for generic parameters in interface methods, and those are not yet supported.
Resolution: removed the interface declaration and `: GBMObjective` from the three
records; removed `where T implements GBMObjective` from `boost()`.  The records
and the generic `boost()` remain — Chapel specialises statically via duck typing.

---

### Leaf-wise tree growth (implemented)

Replaced level-wise (breadth-first) growth with leaf-wise (best-first) growth,
matching LightGBM's default `num_leaves` strategy.

Key changes:
- `src/Tree.chpl` — rewritten. `FittedTree` now uses explicit child pointers
  (`leftChild`, `rightChild`) instead of heap indexing.  Capacity is
  `2*numLeaves-1` nodes.  `updateNodeAssign` routes samples for a single split
  node; `applyTree` walks the child-pointer tree per sample.
- `src/Booster.chpl` — rewritten. `BoosterConfig` drops `maxDepth`, adds
  `numLeaves` (default 31).  `boost()` inner loop maintains an `activeLeaves`
  list and `cachedSplits`; each iteration expands the highest-gain active leaf.
  Histograms are rebuilt from scratch each round (one sample pass per split).
- `src/Histogram.chpl`, `src/Splits.chpl` — backward-compat overloads added
  (no-featSubset variants retained).
- All test files and example drivers updated: `maxDepth` → `numLeaves`
  (depth=4 → numLeaves=16, depth=6 → numLeaves=64).

Result: CoverType 20 trees / 8 leaves → 78% accuracy.

---

### Histogram subtraction trick for leaf-wise growth (implemented)

After each split, instead of rebuilding histograms for all active leaves
from scratch (O(N × nActive) per split), use the subtraction trick:

1. Identify the smaller child by hess sum (leftHess from cachedSplits vs
   total parent hess).
2. `buildHistogramsNode` — scan only samples at the smaller child,
   accumulate into its node slot.  O(N) reads, O(|smaller| × nFeatures) writes.
3. `subtractNode` — derive the larger child in-place:
   `hist[larger] = hist[parent] - hist[smaller]`.  O(nFeatures × MAX_BINS), cheap.
4. `findBestSplitsNodes` — find splits only for the two new children;
   all other active leaves keep their cached SplitInfo.

New procs in `src/Histogram.chpl`: `buildHistogramsNode`, `subtractNode`.
New proc in `src/Splits.chpl`: `findBestSplitsNodes`.
`src/Booster.chpl` inner loop updated accordingly.

Results (CoverType, 100 trees, 1 locale):

| numLeaves | Full rebuild | Subtraction trick | LightGBM | Chapel test acc | LGBM test acc |
|-----------|-------------|-------------------|----------|-----------------|---------------|
| 16        | 161s        | 43s               | 0.8s     | 82.52%          | 82.47%        |
| 31        | 310s        | 69s               | 0.95s    | 85.09%          | 85.37%        |

3.7–4.5× speedup. Accuracy is bit-identical to the full-rebuild version.
Remaining gap vs LightGBM (~50–70×) is SIMD histogram kernels + more
aggressive parallelism.
