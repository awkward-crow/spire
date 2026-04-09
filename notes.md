# Chapel GBM Prototype — Phase 1: Data Layout & Objectives

## Files

| File | Purpose |
|---|---|
| `Objectives.chpl` | Gradient/hessian computation for MSE, LogLoss, Pinball |
| `DataLayout.chpl` | Distributed array layout (`GBMData` record, `printDataSummary`) |
| `SyntheticData.chpl` | Synthetic dataset generators (classification, regression) |
| `TestObjectives.chpl` | Unit tests + distributed smoke test |

## Build & Run

```bash
cd test

make               # build all tests
make run           # build and run all tests
make TestObjectives  # build one test
./TestObjectives -nl 4  # multi-locale (requires GASNet)
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

## Next Steps

Phase 2 — Histogram builder:
- Quantile binning: bin features into uint8 buckets at startup
- `buildHistograms`: distributed forall over rows, accumulate
  (grad_sum, hess_sum) per (node, feature, bin)
- `findBestSplits`: pure local arithmetic on completed histograms
- `updateNodeAssign`: reroute each sample left/right based on split

Open questions to resolve in Phase 2:
1. Histogram memory layout: `[node, feature, bin]` vs `[feature, bin, node]`
2. Distributed quantile binning algorithm: sampling vs Greenwald-Khanna vs t-digest
