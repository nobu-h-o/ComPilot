# Test environment and available transformations

## Test environment

- **CPU:** one AMD EPYC 7702P, 64 cores, 1 thread per core, 4 NUMA nodes. Rated up to 2.18 GHz with boost enabled; the `ondemand` governor means clock speed varies with load.
- **Cache (total):** 2 MiB L1d, 32 MiB L2, 256 MiB L3
- **Memory:** 503 GiB
- **System:** Ubuntu 24.04.5, kernel 6.8, GCC 13.3, LLVM 14.0.6. This is a shared login machine, so other users' load can affect timings.
- **Compiler stack:** Tiramisu, lowered through Halide and LLVM, driven by TiraLib
- **Instruction set:** AVX and SSE4.1 only. AVX2 and FMA are disabled in Tiramisu's target.
- **Threading:** 64-thread pool by default, used only by parallel loops. `HL_NUM_THREADS` controls it; `OMP_NUM_THREADS` has no effect.
- **Benchmarks:** five PolyBench MEDIUM kernels, double precision: 2mm, heat3d, symm, syr2k, trisolv
- **Timing:** minimum over 15 runs for the baseline, and over 3-10 runs per candidate, with a budget of 10x the baseline time
- **Search:** `claude-haiku-5-5`, thinking on, conversational mode, up to 50 responses per kernel, nudged to continue until 30 schedules are tried

The search settings are those of the `medium5_vanilla_epyc` run (2026-11-23).

## Available transformations

Every schedule is also checked by Tiramisu for data-dependency legality.

| Transformation | Syntax | What it does | Main restrictions |
|---|---|---|---|
| **Fusion** | `compA.Fuse(compB, L1)` | Merges two computations' loops down to a level | Two different computations; level must exist in both; best placed first |
| **Interchange** | `comp.Interchange(L0, L1)` | Swaps two loops | Perfectly nested; can't follow parallelization of either loop |
| **Parallelization** | `comp.Parallelize(L0)` | Runs a loop's iterations across threads | `L0` or `L1` only |
| **2D tiling** | `comp.Tile2D(L0, L1, 32, 32)` | Splits two loops into blocks | Consecutive, perfectly nested*; can't start above an already parallelized loop |
| **3D tiling** | `comp.Tile3D(L0, L1, L2, 32, 32, 32)` | Splits three loops into blocks | Same as 2D tiling |
| **Unrolling** | `comp.Unroll(L-1, 8)` | Repeats the loop body to cut loop overhead | Last on a computation; once per computation; factor above 1 and at most the trip count |
| **Skewing** | `comp.Skew(L0, L1)` | Shifts iterations to enable parallelism or improve locality; a solver picks the factors | Consecutive, perfectly nested |
| **Reversal** | `comp.Reverse(L0)` | Runs a loop backwards | No extra ComPilot rule |

\* The prompt tells the LLM this, but ComPilot's own tiling check only enforces consecutive levels and the parallelization rule.

- **Scope:** a command affects every computation sharing the target loop. For example, tiling `comp01` in 2mm also tiles `comp00`.
- **Levels:** `L0` is outermost and `L-1` is innermost. Tiling renumbers the deeper loops.
- **Not included:** TiraLib also has distribution, expansion and matrix modules, but the prompt doesn't offer them to the LLM.
