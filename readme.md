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

## see also

 - file `notes.md`
 - file `chapel_arkouda_gbm_conversation.md`
 - file `docker.md`

### end
