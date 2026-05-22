<h1 align="center">spire</h1>
<p align="center">Distributed gradient boosting machines in <a href="https://chapel-lang.org/">Chapel</a></p>

Spire implements leaf-wise gradient boosted trees with histogram approximation, targeting multi-locale distributed execution via Chapel's PGAS model. Accuracy is within 0.1% of LightGBM on standard benchmarks; the remaining speed gap is in the histogram scatter kernel.

## performance

Single-locale, 20 trees, 16 leaves:

| Dataset | Spire | LightGBM | Gap |
|---------|-------|----------|-----|
| SUSY (5M × 18) | 11.9 s | 4.16 s | 2.9× |
| CoverType (396k × 54) | 1.74 s | 0.39 s | 4.5× |
| Higgs (11M × 28) | — | — | 67.98% acc vs 68.30% (LightGBM, 10 trees) |

## optimisations

- **C histogram kernel** — AoS layout with 4-sample unrolling; CoverType 7.4 s → 1.74 s (4.3×), SUSY 27.1 s → 11.9 s (2.3×)
- **Column-major feature matrix** — stride-1 histogram reads; CoverType 8.4 s → 7.4 s (14%), SUSY 28.8 s → 27.1 s (6%)
- **Batched leaf-wise growth** — accumulates `batchSize=4` children per sample pass; 3× fewer passes per tree at `numLeaves=16`
- **float32 gradients** — halves per-sample read bandwidth and multi-locale reduction payload (~55 KB vs ~110 KB per locale per split)
- **Parallel CSV loading** — byte-range chunks with newline alignment; SUSY 60 s → 15 s (4×)

## distributed execution

Each locale accumulates a partial histogram over its `localSubdomain()` before reducing to locale 0, minimising inter-locale traffic. HDF5 loading uses independent per-locale hyperslab reads (`HDF5Reader.chpl`), requiring no MPI and scaling naturally to a shared filesystem (e.g. EFS).

Multi-locale build:

```sh
CHPL_COMM=gasnet make
./build/TestBooster -nl 4
```

## usage

```sh
cd test
make        # build all tests
make run    # build and run all tests
```

Single test:

```sh
make TestObjectives
./build/TestObjectives
```

Log level `INFO` or `TRACE`:

```sh
./build/TestBooster -logLevel=INFO 2>&1 | less -X
```

By default tests compile with `--fast` (removes Chapel's nil/bounds/overflow checks) and `CHPL_TARGET_CPU=native`. Use `DEBUG=1` to restore checks, `PROFILE=1` for a `--fast -g` profiling build. Override the target CPU for a specific microarchitecture:

```sh
make CHPL_TARGET_CPU=broadwell
make DEBUG=1
```

## column subsampling

`colsampleByTree` draws a random feature subset per tree via partial Fisher-Yates, using a single persistent RNG advanced across all trees. Timing on CoverType (396k × 54, 50 trees, 16 leaves):

| colsample | wall time | test log-loss |
|-----------|-----------|---------------|
| 1.0       | 35 s      | 0.4337        |
| 0.8       | 32 s      | 0.4413        |
| 0.6       | 26 s      | 0.4636        |
| 0.4       | 22 s      | 0.4957        |

Training time scales roughly linearly with colsample. CoverType has mostly informative features so subsampling hurts accuracy; on wider datasets with redundant features it will help.

## references

- **Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine.** Annals of Statistics 29(5). https://projecteuclid.org/euclid.aos/1013203451
- **Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.** KDD '16. https://arxiv.org/abs/1603.02754
- **Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
