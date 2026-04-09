# spire -- gradient boosting machines


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
