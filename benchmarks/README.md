# Benchmarks

The benchmark programs in this folder come from the [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu) polyhedral compiler. They were copied unchanged from `benchmarks/autoscheduler_benchmarks/polybench/` in the Tiramisu repository, at commit [`ef8659c`](https://github.com/Tiramisu-Compiler/tiramisu/tree/ef8659c6dbd7b31e7d93e06df2a83d2b4bf31478/benchmarks/autoscheduler_benchmarks/polybench).

They are Tiramisu implementations of kernels from the [PolyBench/C](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) benchmark suite.

## License

These files are distributed under Tiramisu's MIT License, included here as [`LICENSE`](LICENSE). The rest of ComPilot is under the Apache License 2.0 ([`../LICENSE`](../LICENSE)).

## Contents

| Folder | What it holds |
|---|---|
| `polybench/<SIZE>/` | All 30 PolyBench kernels, at each dataset size: `MINI`, `SMALL`, `MEDIUM`, `LARGE`, `XLARGE`. |
| `polybench_medium5/` | The quick-test set: copies of 5 kernels from `polybench/MEDIUM/` (`2mm`, `heat3d`, `symm`, `syr2k`, `trisolv`). |

Each benchmark folder `function_<kernel>_<SIZE>/` contains:

- `<name>_generator.cpp`: the kernel written in Tiramisu's C++ API, with no optimizations applied. This is the only file ComPilot reads.
- `<name>_wrapper.cpp` / `<name>_wrapper.h`: a test program that fills the input buffers and times the kernel. ComPilot does not use these; TiraLib builds its own.
- `<name>_autoscheduler.cpp`: the same kernel set up for Tiramisu's built-in auto-scheduler. ComPilot does not use it.

## Using a different set

Set `benchmarks_dir` in `../config.yaml` to any folder of benchmark folders, for example `./benchmarks/polybench/MEDIUM/` for all 30 kernels at MEDIUM size. To make a smaller set, copy the benchmark folders you want into a new folder, as `polybench_medium5/` does.
