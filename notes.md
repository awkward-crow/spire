# Chapel GBM Prototype

## Files

| File | Purpose |
|---|---|
| `Objectives.chpl` | Gradient/hessian computation for MSE, LogLoss, Pinball |
| `DataLayout.chpl` | Distributed array layout (`GBMData` record, `printDataSummary`) |
| `SyntheticData.chpl` | Synthetic dataset generators (classification, regression) |
| `TestObjectives.chpl` | Unit tests + distributed smoke test |
| `Binning.chpl` | *(Phase 2)* Sampling-based quantile binning → `uint8` bin matrix |
| `Histogram.chpl` | *(Phase 2)* Histogram accumulation and subtraction trick |
| `Splits.chpl` | *(Phase 2)* Split finding, `SplitInfo` record |
| `Tree.chpl` | *(Phase 2)* Node assignment, leaf values, `applyTree` |
| `Booster.chpl` | *(Phase 2)* End-to-end training loop, `BoosterConfig`, `boost` |

## Build & Run

```bash
cd test

make               # build all tests
make run           # build and run all tests
make TestObjectives  # build one test
./build/TestObjectives -nl 4  # multi-locale (requires GASNet)
make clean         # remove build/
```

To add a new test, append its module name to `TESTS` in `test/Makefile`.

## Design Notes

### Why row-partitioned?

Each locale owns a contiguous block of rows (all features for those
samples).  During histogram building, gradient/hessian reads are purely
local — no cross-locale traffic.  The cost is cross-locale movement
when repartitioning samples into left/right child nodes at each tree
level, offset by the histogram subtraction trick (only rebuild the
smaller child's histogram; the larger is parent minus smaller).

### Pinball loss for quantile regression

The pinball (quantile) loss is:

```
L(y, F; tau) = tau * max(y - F, 0)  +  (1 - tau) * max(F - y, 0)
```

Gradient w.r.t. F:
- `-tau`     when F < y  (under-predicted — push F up)
- `+(1-tau)` when F > y  (over-predicted  — push F down)
- `0`        when F == y

Hessian is 0 almost everywhere (piecewise linear loss), so we use the
constant approximation `hess = 1.0` — the same convention as LightGBM
and sklearn's HistGradientBoosting.

For time series applications, training separate models at e.g.
`tau = 0.1` and `tau = 0.9` gives an 80% prediction interval directly
from the GBM output without any distributional assumptions.

### Distributed quantile binning — sampling approach

**Decision: use sampling.**

Each locale draws a random subset of its local rows (e.g. `sqrt(nLocalRows)`
or a fixed cap), communicates those values to locale 0, which sorts each
feature column and picks 255 evenly-spaced quantile boundaries.  The
resulting `uint8` cut-points are broadcast back and each locale bins its
own rows independently.

Alternatives considered:

| Algorithm | Accuracy | Complexity | Why not chosen |
|---|---|---|---|
| **Sampling** *(chosen)* | Approximate; improves with sample size | Low | Simple, no streaming state, same approach as XGBoost/LightGBM default |
| Greenwald-Khanna | ε-exact | High | Non-trivial merge step; overkill for initial binning pass |
| t-digest | Approximate, good tails | Medium | Better tail accuracy than sampling but adds a dependency and complexity |

For typical GBM use (255 bins, millions of rows) sampling error is
negligible — a misplaced bin boundary shifts one split threshold slightly,
which the next boosting round corrects. Exact quantiles are not worth the
implementation cost here.

### SIMD / vectorization scope

`CHPL_TARGET_CPU=native` (set in `test/Makefile`) enables auto-vectorization
for the build machine's CPU at zero implementation cost.  Beyond that, the
three hot loops have different vectorization prospects:

| Loop | Vectorizable? | Approach |
|---|---|---|
| Gradient/hessian computation | Yes — element-wise over contiguous arrays | Auto-vectorizes with `native` |
| Binning application | Yes — independent binary search per sample | Auto-vectorizes with `native` |
| Split finding prefix scan | Partially — independent across features, sequential within | `extern C` AVX2 horizontal prefix sum (future) |
| Histogram accumulation | No — data-dependent scatter (`accumGrad[node,f,b] +=`) | Not worth pursuing; focus on cache layout instead |

The split finding scan (255 iterations per feature) is the most worthwhile
manual SIMD target — LightGBM uses AVX2 horizontal sums there for ~4–8×.
Deferred until `Splits.chpl` is written and profiled.

The histogram accumulation scatter is the fundamental bottleneck.  AVX-512
has scatter instructions but they are slow on current hardware.  Better
returns come from the memory layout benchmark (see below).

Override `CHPL_TARGET_CPU` for heterogeneous clusters where node CPUs differ,
e.g. `make CHPL_TARGET_CPU=broadwell`.

### Histogram memory layout

**Decision: deferred — implement `[node, feature, bin]` first, benchmark,
then try `[feature, bin, node]` if cache misses are a bottleneck.**

During histogram accumulation the inner loop touches one `(node, feature,
bin)` cell per sample. With `[node, feature, bin]` layout, samples in the
same node and feature land in adjacent memory; with `[feature, bin, node]`
the node dimension is innermost, which may suit split-finding better.
Benchmark before optimising.

## Next Steps

### Phase 3 — Comparison against LightGBM / XGBoost

Critical path to a benchmark on a standard dataset (e.g. California Housing,
Titanic):

1. **CSV reader** (`src/CSVReader.chpl`) — load a dense float/label matrix from
   a text file into `GBMData`.  Even a minimal implementation (no missing
   values, all columns numeric) unblocks real-dataset testing.

2. **`predict` on held-out data** — a `predict(trees: [] FittedTree, data: GBMData): [] real`
   procedure that applies the fitted ensemble to a new dataset without
   modifying `data.F`.  Can be tested against the synthetic data we already
   have.

3. **Comparison driver** (`src/` or standalone) — load dataset, train/test
   split, train the Chapel GBM, print RMSE (regression) or log-loss
   (classification) on the test set alongside equivalent LightGBM/XGBoost
   numbers for direct comparison.

Order: `predict` first (no I/O dependency, testable immediately), then CSV
reader, then the driver.

### Open questions

- Histogram memory layout benchmark (`[node, feature, bin]` vs `[feature, bin, node]`)
- Row/column subsampling (both LGBM and XGBoost default to subsampling;
  omitting it will widen the accuracy gap on noisy datasets)
- Missing value handling (required for most real-world datasets)
