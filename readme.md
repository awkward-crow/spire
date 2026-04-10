# spire -- gradient boosting machines

    claude --resume 750c1825-187f-47bf-ba62-595de9b3086e       

## latest

 - quantile regression on bicycle data
  = Records (MSE, LogLoss, Pinball) replacing the enum Objective + dispatch chains
  = GBMEnsemble bundling trees + baseScore
  = BoosterConfig stripped of tau and minHess
  = boost() generic via duck typing, predict() taking GBMEnsemble
  = t =digest binning replacing random sampling                                                            = Pinball hessian fixed to tau*(1 =tau)

And see `refactor.md`, objectives mature from an enum to separate records.

## next steps

0.1 try california housing data with multi-locale!!
0.2 logloss example

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

## see also

 - file `notes.md`
 - file `chapel_arkouda_gbm_conversation.md`
 - file `docker.md`


### end
