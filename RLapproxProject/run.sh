#!/bin/bash

# Run nRuns (in nProcs synchronous processes) of the top-level Python
# file given (without the .py suffix) as the only command-line
# argument.

nRuns=10
nProcs=5

cleanup() {
    pkill -P $$
    wait
    exit 1
}

trap "cleanup" SIGINT SIGTERM

for (( run=0 ; run-nRuns ; run=run+1 )); do
    # ./"$1.py" $run > "$1"-0$run.log
    "./$1.py" $run > "$1"-0$run.log & echo "./$1.py" $run &
    (( (run + 1) % nProcs == 0 )) && wait
done

wait
