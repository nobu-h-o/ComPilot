# ComPilot: LLM-Guided Loop Optimization via Agentic Auto-Scheduling

ComPilot uses a general-purpose large language model as the search policy for loop optimization. The model proposes sequences of loop transformations for a given loop nest; the Tiramisu polyhedral compiler checks each proposal for legality, executes the legal ones, and reports the measured speedup. The model uses that feedback to decide what to try next, and the process repeats until it stops or a turn budget is reached. The model is used as-is through a standard chat API, with no fine-tuning or gradient updates.

This repository is the reference implementation for the paper *Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization* (PACT 2025), by Massinissa Merouani, Islem Kara Bernou, and Riyadh Baghdadi (NYU Abu Dhabi).

---

## How it works

Each benchmark is optimized through a dialogue between the model and the compiler:

1. The loop nest is shown to the model as anonymized C code, annotated with an ID per computation and accompanied by its measured baseline runtime. The model is first asked to analyze the program.
2. On each turn the model proposes a schedule (a sequence of transformations). ComPilot parses it, and either rejects a malformed or structurally invalid proposal immediately with a specific reason, or applies it through TiraLib.
3. Tiramisu checks legality by polyhedral dependence analysis. A legal schedule is compiled and timed, and its speedup over the baseline is returned to the model; an illegal or failing one is returned with the reason instead.

The transformations available to the model are fusion, distribution, interchange, parallelization, 2D and 3D tiling, unrolling, skewing, and reversal.

Rejected proposals are returned with a concrete cause (unknown computation ID, out-of-range loop level, wrong argument count, dependence violation, and so on) rather than a generic error, and for an illegal combination of transformations ComPilot can optionally point out which one is at fault. Correctness is enforced first by a lightweight validity check before any compiler call, then by Tiramisu's polyhedral legality analysis. The system is built for long unattended runs: LLM requests retry on transient failures, one benchmark's crash never aborts the rest, and the dialogue is checkpointed each turn so an interrupted run resumes where it stopped. Independent runs launched with distinct `--run-id` values give a best-of-K search, and two interaction formats are supported, a conversational transcript and a token-efficient compact mode (see [Interaction modes](#interaction-modes)).

---

## Setup & Dependencies

ComPilot drives the [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu) polyhedral compiler through **TiraLib**. Both must be built and available before running ComPilot.

1. Build Tiramisu and TiraLib, and make sure TiraLib's `config.yaml` points to your Tiramisu install. ComPilot runs TiraLib in server mode, which also needs [TiraLibCPP](https://github.com/Tiramisu-Compiler/TiraLibCPP). By default `config.yaml` expects the three checkouts inside the repository root (`./tiramisu`, `./TiraLib`, `./TiraLibCPP`, all gitignored):
   - **Tiramisu**: requires CMake ≥ 3.22, LLVM 14 (Halide at the pinned submodule commit is v14 and supports LLVM 12–15), ISL (vendored in `3rdParty/isl`), and Halide (`3rdParty/Halide` submodule). Check out the submodule at its pinned commit with `git submodule update --init 3rdParty/Halide`; `utils/scripts/install_submodules.sh` uses `--remote`, which pulls Halide `master` instead. Configure with `-DUSE_AUTO_SCHEDULER=TRUE` (TiraLib links `libtiramisu_auto_scheduler`) and install to a prefix, e.g. `cmake --install build --prefix tiramisu/install`.
   - **TiraLibCPP**: `cmake -S . -B build -DTIRAMISU_INSTALL=<tiramisu prefix> -DCMAKE_INSTALL_PREFIX=<prefix> && cmake --build build && cmake --install build`. The build needs Tiramisu's, Halide's and ISL's include directories on `CPATH` (`<tiramisu>/install/include`, `<tiramisu>/3rdParty/Halide/install/include`, `<tiramisu>/3rdParty/isl/build/include`) and their library directories on `LIBRARY_PATH` (`<tiramisu>/install/lib`, `<tiramisu>/3rdParty/Halide/install/lib`, `<tiramisu>/3rdParty/isl/build/lib`).
   - **Distribution** (`Distribute`) needs the `distribution` branches of the [TiraLib](https://github.com/nobu-h-o/TiraLib) and [TiraLibCPP](https://github.com/nobu-h-o/TiraLibCpp) forks; upstream TiraLibCPP does not parse the `D(...)` schedule token, so every schedule using it would fail.
   - **TiraLib**: copy `config.yaml.example` to `config.yaml` and list the include/lib directories of Tiramisu, Halide, ISL and TiraLibCPP under `dependencies`.
2. Install the Python dependencies (TiraLib requires `numpy<2`), ideally in a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The commands below assume this environment is active (or call `.venv/bin/python` directly).
3. Set your API key: `cp .env.example .env` and fill in `ANTHROPIC_API_KEY` (or `GEMINI_API_KEY` if you switch `config.yaml` to Gemini).
4. Copy/edit `config.yaml` (see below) to match your setup.

---

## Configuration

All settings live in `config.yaml`. String values may reference environment variables as `${VAR}` or `${VAR:-default}`; these are read from the shell or from a `.env` file in the repository root (see `.env.example`). By default ComPilot uses Claude (`claude-haiku-5-5`) through [Anthropic's OpenAI-compatible endpoint](https://platform.claude.com/docs/en/api/openai-sdk), with the key taken from `ANTHROPIC_API_KEY`. A commented-out Gemini block (`gemini-3.8-flash`, key from `GEMINI_API_KEY`) sits right below it in `config.yaml`; set `LLM_ID` / `LLM_BASE_URL` to point elsewhere. Provider quirks are handled in `compilot_utils._adapt_request_kwargs`: for Claude, `temperature`, `top_p` and `seed` are not sent (Claude 5 models reject non-default sampling values) and thinking stays on by default, so `enable_thinking` has no effect; for Gemini, `seed` and `chat_template_kwargs` are not sent.

| Key | Description |
|---|---|
| `llm_id` | Model id served by your OpenAI-compatible endpoint (e.g. `claude-haiku-5-5`, `gemini-3.8-flash`, `gpt-4o`, a local vLLM/Ollama model). |
| `base_url` | Base URL of the OpenAI-compatible API. |
| `openai_api_key` | API key for the endpoint (defaults to `${ANTHROPIC_API_KEY}`; any non-empty string for most local servers). |
| `interaction_mode` | `conversational` (default) or `compact` — see [Interaction modes](#interaction-modes). |
| `benchmarks_dir` | Directory of benchmark folders; each `<name>/` must contain `<name>_generator.cpp`. The PolyBench kernels (from Tiramisu, see [`benchmarks/README.md`](benchmarks/README.md)) are included in `benchmarks/polybench/<SIZE>/` for sizes MINI, SMALL, MEDIUM, LARGE, XLARGE. The default is the 5-kernel quick-test set `benchmarks/polybench_medium5/`. |
| `tiralib_dir` | Path to a compiled TiraLib checkout. |
| `lookup_dir_suffix` | Cache of already-tried schedules is stored under `lookup_dir_<suffix>/`. |
| `base_results_dir` | Where results and checkpoints are written. |
| `max_llm_responses` | Hard cap on LLM turns per benchmark. |
| `min_nb_suggestions` | Keep nudging the LLM to explore until it has tried at least this many schedules. |
| `nb_exec_init` / `min_runs_scheds` / `max_runs_scheds` / `time_budget_factor_scheds` | Execution-measurement budgets. |
| `temperature` / `top_p` / `nb_responses` / `enable_thinking` | LLM sampling settings. Not sent to Claude (see above); they still appear in the experiment folder name. |
| `use_response_lookup` | Reuse cached compiler feedback for previously-seen schedules. Set to `false` to force fresh feedback for every schedule (e.g. when comparing feedback wording or running independent stochastic runs). The numeric caches (initial execution time, skewing factors) are always kept. |
| `localize_illegality` | When an illegal schedule combines several transformations, legality-check each one on its own to point out the culprit. Off by default (one extra legality check per component). |
| `experiment_name` | Base experiment name, used when `--experiment-name` is not given. |

> **Security note:** do not commit real endpoint URLs or credentials. Put them in `.env` (gitignored) rather than in `config.yaml`. The API key is redacted from the configuration saved in results files.

---

## Interaction modes

Set `interaction_mode` in `config.yaml`:

- **`conversational`** (default) — a back-and-forth transcript. Every turn resends the full history (system prompt + all prior suggestions and feedback), so the prompt grows with each iteration.
- **`compact`** — every turn sends only two messages: the system prompt and a single message holding the loop nest, a *frozen* one-time program analysis, and a running **exploration log** (a de-duplicated, numbered summary of every schedule tried and its outcome, plus the current top schedules). The LLM's next suggestion is evaluated and appended to the log.

Both modes reach comparable speedups. Compact mode keeps the per-turn prompt small and roughly constant, so its **prompt-token cost grows far more slowly** with the number of turns. It trades this for slightly more reasoning per turn (the model re-reads the state each time), so the net token savings grow with the dialogue length — modest for very short runs, substantial for long ones. Compact runs are tagged with a `_compact` suffix on the experiment name.

## Random-search baseline (no LLM)

To measure what the LLM contributes, uncomment `search_mode: random` in `config.yaml` (comment it out again to go back to the LLM). ComPilot then draws transformations at random from the same space (Fuse, Distribute, Interchange, Parallelize, Tile2D, Tile3D, Unroll, Skew, Reverse), using the same compiler backend, cache and result files, but makes no LLM calls and needs no API key. It runs as a random walk: starting from the original program, it appends a random transformation to the current schedule and sends it to Tiramisu. A schedule Tiramisu reports as illegal (or that crashes) is dropped and never drawn or built on again. A legal one is executed and becomes the base for the next step. The walk restarts from the original program at random (`restart_prob`) or once a schedule reaches `max_schedule_length`. It stops after `nb_schedules` executed schedules. The `random_search` block in `config.yaml` sets the seed, budget, allowed transformations and tiling/unrolling factors. Runs are named `<experiment_name>_random_N<nb_schedules>_S<seed>`, and `--run-id` changes the random stream, so best-of-K works as it does for the LLM.

---

## Usage

Run all commands from the repository root.

### Quick test (5 benchmarks)

`config.yaml` points `benchmarks_dir` at `benchmarks/polybench_medium5/` by default: five PolyBench MEDIUM kernels (`2mm`, `heat3d`, `symm`, `syr2k`, `trisolv`), which is a fast way to check the whole pipeline end to end:

```bash
cp .env.example .env          # first time only; fill in ANTHROPIC_API_KEY
python src/compilot.py --experiment-name medium5_test
```

For the full 30-kernel suite, switch `benchmarks_dir` in `config.yaml` to the commented-out `./benchmarks/polybench/MINI/` line (or another size directory) and use a new experiment name.

> Benchmarks already recorded in an experiment's results JSON, including ones that failed during setup, are skipped when the same experiment is re-run. To retry them, delete `results/<experiment folder>/` or use a new `--experiment-name`.

### Single exploration

```bash
python src/compilot.py --experiment-name my_first_run
```

Command-line options: `--experiment-name/-n` (overrides `experiment_name` in the config), `--run-id/-i` (see below), `--config` (path to an alternative config file, default `config.yaml`), and `--completion-file/-c` (a file created when the run finishes, used by `run_duplicated_compilot.sh`).

### Best-of-K (multiple independent runs)

Because different runs converge to different optima, running several independent dialogues and keeping the best schedule is recommended (K=5 is a good default). Launch runs with distinct `--run-id` values (they can run in parallel; they safely share the schedule cache):

```bash
for i in $(seq 1 5); do
  python src/compilot.py --experiment-name my_experiment --run-id $i &
done
wait
```

### Resuming after a crash or interruption

Runs are resumable out of the box. If a run is interrupted (crash, timeout, LLM outage), simply re-launch the **same command**: completed benchmarks are skipped, and any benchmark that was mid-dialogue resumes from its checkpoint in `<results_dir>/checkpoints/`.

---

## Output

The experiment folder is named after the experiment name plus the model and main settings: `<experiment_name>_<model>_T<temperature>_TP<top_p>_L<min_nb_suggestions>_<wT|woT>` (with a `_compact` suffix in compact mode), e.g. `results/medium5_test_claude-haiku-5-5_T1_TP1_L30_wT/`. With `--run-id N`, all runs share that folder and each run's files get a `_runN` suffix. The folder contains:
- `results_<name>.json` — full optimization dialogue, explored schedules, best schedule, timings, and token usage per benchmark (the API key is redacted from the saved configuration).
- `live_results_<name>.csv` — one summary row per benchmark (initial and best execution time, best speedup, schedule counts, optimization time).
- `params.json` (or `params_runN.json`) — the run's main parameters, model id and start time.
- `checkpoints/` — per-benchmark resume state (each checkpoint is deleted once its benchmark finishes).

---

## Evaluation

The paper reports two aggregate speedup metrics, both measured against each program's original runtime and as a function of the exploration budget `T` (the number of schedules tried):

- **COMPILOT@T** — the single-run result: for each benchmark take the best speedup found within the first `T` schedules, take the median of that across runs, then the geometric mean across benchmarks.
- **COMPILOT_K@T** — the best-of-K result: the typical speedup obtained by running `K` independent explorations and keeping the best, again aggregated across benchmarks.

Because the metrics aggregate over independent runs, first produce several runs of the same experiment with distinct `--run-id` values (see [Best-of-K](#best-of-k-multiple-independent-runs)), then aggregate:

```bash
python evaluation/aggregate_json_results.py -r results/<experiment folder> --k 5
```

`-r` takes the full experiment folder (see [Output](#output)), which must contain `results_<name>_run*.json` files from `--run-id` runs.

This prints a table of COMPILOT@T and COMPILOT_K@T over `T = 5, 10, ..., 30` and writes a per-benchmark CSV (`aggregate_results_*.csv`) alongside it. The benchmarks used in the paper are the [PolyBench](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) kernels.

For a quick per-run sanity check without aggregating, `python evaluation/analyze_dialogue.py results/<...>/results_*.json` prints, per benchmark, how many proposed schedules were runnable vs. rejected vs. crashed, the runnable rate, and the best speedup (works for both interaction modes).

To regenerate the paper's figures, see `evaluation/plotting.py` (requires `matplotlib`/`plotly`).

---

## File Structure

- `config.yaml` — centralized configuration.
- `.env.example` — template for the `.env` file holding API keys (copy to `.env`, which is gitignored).
- `benchmarks/` — PolyBench kernels from Tiramisu (MIT-licensed; see [`benchmarks/README.md`](benchmarks/README.md)).
- `run_duplicated_compilot.sh` — launches several runs as SLURM jobs in `screen` sessions, restarting each until it completes (cluster-specific; edit its Conda and SLURM settings first).
- `src/` — the core system:
  - `compilot.py` — main driver: runs the optimization dialogue for every benchmark, with checkpointing/resume.
  - `compilot_utils.py` — parsing, validation, compiler feedback generation, and the TiraLib/Tiramisu backend interface.
  - `prompts.py` — the system instructions and message templates used with the LLM.
  - `random_search.py` — the random-search baseline used when `search_mode: random`.
- `evaluation/` — analysis and metrics (not needed to run ComPilot):
  - `analyze_dialogue.py` — quick per-run outcome analysis.
  - `aggregate_json_results.py` / `eval_utils.py` — the paper's aggregate metrics (COMPILOT@T and best-of-K) across runs.
  - `plotting.py` — supplementary figure-generation code for the paper's plots (requires `matplotlib`/`plotly`).

## Citation

If you use this code in your research, please cite:
> Merouani, M., Bernou, I. K., & Baghdadi, R. (2025). Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization. In *Proceedings of the 2025 International Conference on Parallel Architectures and Compilation Techniques (PACT)*.

## License

ComPilot is released under the [Apache License 2.0](LICENSE).

The benchmark programs in `benchmarks/` come from the [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu) compiler and remain under Tiramisu's [MIT License](benchmarks/LICENSE). See [`benchmarks/README.md`](benchmarks/README.md).
