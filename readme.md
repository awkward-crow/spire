<h1 align="center">spire</h1>

Spire implements leaf-wise gradient boosted trees with histogram approximation using [Chapel](https://chapel-lang.org)'s distributed programming model. It supports binary classification (log-loss), regression (MSE) and quantile regression (pinball loss). Single-node performance is competitive with LightGBM; the architecture is designed for scale-out across distributed memory where Chapel's model comes into its own.

## performance

Single-locale, 50 trees, 31 leaves. Timings from a modest laptop (Intel Core i7-1165G7, 16 GB RAM).

| Dataset | Rows | Features | Colsample | Spire (s) | LightGBM (s) | Ratio | Spire acc | LightGBM acc |
|---------|------|----------|-----------| --------- |--------------|-------|-----------|--------------|
| SUSY | 5 M | 18 | 1.0 | 38.3 | 12.45 | 3.1× | 80.01% | 80.04% |
| CoverType | 396 k | 54 | 1.0 | 8.23 | 0.62 | 13.3× | 82.94% | 83.31% |
| CoverType | — | — | 0.75 | 6.37 | 0.62 | 10.3× | 82.77% | 83.01% |
| Higgs | 11 M | 28 | 1.0 | 125.1 | 36.8 | 3.4× | 72.37% | 72.40% |
| Higgs | — | — | 0.8 | 113.8 | 38.5 | 3.0× | 72.32% | 72.30% |
| Higgs | — | — | 0.6 | 87.5 | 33.3 | 2.6× | 72.10% | 72.24% |

## optimisations

- **C histogram kernel** — AoS layout with 4-sample unrolling; CoverType 7.4 s → 1.74 s (4.3×), SUSY 27.1 s → 11.9 s (2.3×)
- **Column-major feature matrix** — stride-1 histogram reads; CoverType 8.4 s → 7.4 s (14%), SUSY 28.8 s → 27.1 s (6%)
- **Batched leaf-wise growth** — accumulates `batchSize=4` children per sample pass; 3× fewer passes per tree at `numLeaves=16`
- **float32 gradients** — halves per-sample read bandwidth and multi-locale reduction payload (~55 KB vs ~110 KB per locale per split)
- **Parallel CSV loading** — byte-range chunks with newline alignment; SUSY 60 s → 15 s (4×)

## distributed execution

Each locale accumulates a partial histogram over its `localSubdomain()` before reducing to locale 0, minimising inter-locale traffic. HDF5 loading uses independent per-locale hyperslab reads (`HDF5Reader.chpl`), requiring no MPI and scaling naturally to a shared filesystem (e.g. EFS). Multi-locale benchmarks on AWS are in progress.

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

## references

- **Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine.** Annals of Statistics 29(5). https://projecteuclid.org/euclid.aos/1013203451
- **Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.** KDD '16. https://arxiv.org/abs/1603.02754
- **Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
