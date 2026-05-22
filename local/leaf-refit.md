# Post-hoc Leaf Refit for Quantile Objectives

## The problem

Standard GBM with pinball loss uses the Newton step to compute leaf values:

```
leaf value = -G / (H + λ)
```

where `G = Σ g_i` and `H = Σ h_i` over samples in the leaf.

For pinball loss the gradient is a constant: `g_i = -tau` (under-predicting) or
`g_i = 1 - tau` (over-predicting). The hessian is approximated as `tau*(1-tau)`.
So the leaf value is at most `tau / (tau*(1-tau)) = 1 / (1-tau)` — a constant
independent of residual magnitude. With `eta=0.1` and `tau=0.1` the per-tree
step is at most ~0.11, while actual residuals can be hundreds.

This is the root cause of the Chapel vs LightGBM performance gap on Bicycle
(pinball loss ~18 vs ~10 after 50 trees).

## What LightGBM does instead

LightGBM's `RenewTreeOutput()` (in `regression_objective.hpp`) runs after the
tree structure is fixed:

1. For each leaf, collect the residuals `r_i = y_i - F_i` for all samples
   assigned to that leaf.
2. Sort them.
3. Set `leaf value = eta × quantile(r, 1 - tau)`.

This is the exact minimiser of pinball loss on the leaf's partition. Tree
*growth* still uses the approximate histogram gradients (to find good splits),
but the final leaf *values* are set to the true quantiles of the residuals. No
bounded-step approximation.

XGBoost 2.x does the same thing via `UpdateTreeLeaf()`, using a weighted
quantile of the labels in each leaf (equivalent under the GBM residual
decomposition).

## Chapel implementation

### New method on each objective

Add a `leafRefit` method to the duck-typing contract:

```chapel
// MSE, LogLoss — no-op, Newton step is already optimal
proc leafRefit(ref tree: FittedTree, nodeId: [] int,
               F: [] real, y: [] real, eta: real) { }

// Pinball — replace Newton step with tau-quantile of residuals
proc leafRefit(ref tree: FittedTree, nodeId: [] int,
               F: [] real, y: [] real, eta: real) {
  // 1. Gather nodeId and residuals to locale 0
  // 2. Group by leaf, sort residuals per leaf
  // 3. Overwrite tree.value[leaf] = eta * quantile(residuals, tau)
}
```

### Hook in boost()

After `finalizeLeaves` and before `applyTree` in `Booster.chpl`:

```chapel
finalizeLeaves(trees[t], hist, d, cfg.lambda, cfg.eta);
obj.leafRefit(trees[t], nodeId, data.F, data.y, cfg.eta);   // ← new
// ...
applyTree(data, trees[t], data.F);
```

`nodeId` is a distributed array (`[data.rowDom] int`) already in scope at this
point, carrying the leaf heap index for every sample.

### Multi-locale gather

`nodeId`, `data.y`, and `data.F` are distributed. The gather pattern follows
`Pinball.initF`:

```chapel
var localNodeId  : [0..#data.numSamples] int;
var localResidual: [0..#data.numSamples] real;
forall i in data.rowDom {
  localNodeId[i]   = nodeId[i];
  localResidual[i] = y[i] - F[i];
}
```

Chapel aggregates remote reads across locales; the forall is cheaper than an
explicit coforall because samples are rarely on the "wrong" locale for the
forall body.

Locale 0 then groups by leaf index, sorts each group, and overwrites
`tree.value`. With 14k rows and 16 leaves (depth 4) this is ~112k reals —
negligible. Scales to tens of millions of rows before locale-0 memory pressure
matters.

### Complexity

| Step | Cost |
|------|------|
| Gather nodeId + residuals | O(N) communication |
| Group by leaf | O(N) |
| Sort each leaf's residuals | O(N log(N/L)) where L = number of leaves |
| Overwrite leaf values | O(L) |

Dominated by histogram building at O(N × D × B) where D = maxDepth, B = nBins.
The refit adds a negligible constant.

### Files changed

| File | Change |
|------|--------|
| `src/Objectives.chpl` | Add `leafRefit` to MSE (no-op), LogLoss (no-op), Pinball (quantile refit) |
| `src/Booster.chpl` | One extra call after `finalizeLeaves` |

No changes to Tree, Histogram, Splits, DataLayout, Binning, or any tests
(existing `testPinballDecreases` still passes — loss still decreases, just
faster).

## Results

50 trees, depth 4, Bicycle dataset:

|          | tau=0.1 test | tau=0.9 test | coverage | width |
|----------|-------------|-------------|----------|-------|
| Newton step | 18.18    | 40.17       | 88.1%    | 440   |
| Leaf refit  | 10.58    | 11.51       | 81.9%    | 187   |
| LightGBM    | 10.67    | 11.30       | 82.3%    | 186   |

Chapel matches LightGBM within ~1% on both quantiles after the refit.
