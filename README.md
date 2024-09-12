# DPconv: Super-Polynomially Faster Join Ordering

## Build

```
cd src/
mkdir -p build
cd build
cmake ..
make
```

## Clique Benchmark

Generating Fig. 6 and Fig. 8: To obtain the running times for clique queries, we will use `./clique_bench` (this will run 5 rounds of query graphs of 3 relations up to $n$).

In the experiments, we used the largest join cardinality $W = 10^8$, as follows:

```
./clique_bench 24 100000000
```

All results (even those of individual rounds) are stored in the `benchs` directory, labeled `normal-clique`. The average result is stored in a file with the suffix `-final`.

As regards the actual figures:
* Fig. 6 uses `minmax_dpsub_time` and `minmax_dpconv_instant_boosted_time`.
* Fig. 8 uses `hybrid_capped_dpsub_time`, `simple_capped_dpsub_time`, and `minplus_dpsub_time`.

## JOB & CEB

To obtain Fig. 5, we can use the `./bench` binary. We provide the query graphs with the true cardinalities of JOB and CEB directly in `queries`. The list of pairs represents (bitset, cardinality).

Since the previous clique benchmark does not use `DPccp`, due to its overhead to generate the ccp's, let's activate the mode to have it.

To this end, change the macro in `BenchmarkRunner.cpp` to the following configuration:

```
#define CLIQUE_BENCHMARK 0
#define CAPPED_COUT_BENCHMARK 1
```

And run `make` again:

```
make
```

Then,

```
./bench ../../queries/job/
```

generates the necessary runtimes in `benchs`, prefixed by `cap-cout`. Similarly, for CEB run the following (might take a couple of minutes due to the large number of queries):

```
./bench ../../queries/ceb-imdb-full/
```

You can also obtain the statistics about the $C_{out}, C_{max}$, and $C_{cap}$ values in `cout_cmax_ratios.csv`.

## Algorithms

All algorithms are in `src/algorithms`.

1. `DPconv`: Its simple and fast algorithm (Sec. 6) is in `DPconv.cpp`.
2. `DPsub`: All variants of `DPsub` are in `DPsub.cpp`.
3. `DPccp`: All variants of `DPccp` are in `DPccp.cpp`.
4. `SubsetConvolution`: The heart of our improved layered dynamic programs is `include/SubsetConvolution.hpp`.

In particular, `BooleanFSC` implements all our optimizations (Sec. 5.2, Sec. 5.3) and reduces the copy between the DP-table and the actual subset convolution.

In `BoostedBooleanFSC`, we reduce the number of steps in the last step.
