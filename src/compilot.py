"""ComPilot driver.

Runs the optimization dialogue for every benchmark in the configured directory:
for each program it initializes the LLM context, then iterates
suggestion -> compile/measure -> feedback until the LLM stops or the turn cap is
reached, keeping the fastest legal schedule. Supports two interaction formats
(conversational and compact, see `interaction_mode` in config.yaml), per-iteration
checkpointing for crash/interruption resume, and best-of-K via `--run-id`.

Usage (run from the repository root):
    python src/compilot.py --experiment-name my_run [--run-id 1] [--config config.yaml]
"""
import os
import sys
import argparse
import time
import json
import pickle
import traceback
import re
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', '-i', help="run_ID to be appended to experiment name", type= str)
parser.add_argument('--experiment-name', '-n', help="name of the experiment", type= str)
parser.add_argument('--completion-file', '-c', help="a name for the completion signaling file", type= str, default= "0")
parser.add_argument('--config', help="path to the yaml configuration file", type= str, default= "config.yaml")
args = parser.parse_args()

# Load environment variables from a .env file (repo root or cwd), without
# overriding variables already set in the shell. See .env.example.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

def _expand_env(value):
    """Recursively expand ${VAR} / ${VAR:-default} references in config strings."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    return value

# Load config
with open(args.config, 'r') as f:
    config = _expand_env(yaml.safe_load(f))

# Search strategy: "llm" (default, ComPilot) or "random" (random-search baseline,
# see random_search.py). Set with `search_mode: random` in the config.
search_mode = str(config.get("search_mode") or "llm").lower()
if search_mode not in ("llm", "random"):
    sys.exit(f'Unknown search_mode "{search_mode}" in the config: use "llm" or "random".')
random_mode = search_mode == "random"
if random_mode:
    config["openai_api_key"] = config.get("openai_api_key") or "unused-random-search"

if not config.get("openai_api_key"):
    sys.exit("No API key configured: set ANTHROPIC_API_KEY (or GEMINI_API_KEY for Gemini) in your environment or in .env "
             "(cp .env.example .env), or set openai_api_key in the config.")

os.environ['OPENAI_API_KEY'] = config.get("openai_api_key", "<SET_OPENAI_API_KEY>")
os.environ['LLM_ID'] = config.get("llm_id", "<LLM_ID>")
os.environ['BASE_URL'] = config.get("base_url", "<BASE_URL>")
os.environ['BENCHMARKS_DIR'] = config.get("benchmarks_dir", "<BENCH_FOLDER_PATH>")
os.environ['TIRALIB_DIR'] = config.get("tiralib_dir", "<TIRALIB_PATH>")
os.environ['LOOKUP_DIR_SUFFIX'] = config.get("lookup_dir_suffix", "base")
# Gate the textual compiler-response cache (see compilot_utils.RESPONSE_LOOKUP_ENABLED).
# Set use_response_lookup: false in the config to force freshly generated feedback
# for every schedule (recommended when comparing feedback wording or running
# multiple stochastic runs).
os.environ['USE_RESPONSE_LOOKUP'] = "1" if config.get("use_response_lookup", True) else "0"
# Optionally localize the offending transformation of an illegal compound schedule
# (see compilot_utils.LOCALIZE_ILLEGALITY). Off by default: it adds one legality
# check per component, which is noticeable on large programs.
os.environ['LOCALIZE_ILLEGALITY'] = "1" if config.get("localize_illegality", False) else "0"

import compilot_utils
from compilot_utils import (
    TiramisuProgram, Schedule, check_in_lookup_file, update_lookup_file,
    annots_to_c_prog, initialize_conversation, print_compiler_message,
    print_llm_message, get_LLM_response, extract_transformation_commands,
    get_compiler_response, build_compact_message, get_compact_LLM_response, logger
)
from prompts import (
    system_instructions, start_exploration_message, continue_optimization_messages,
    compact_start_instruction, compact_next_instruction,
)
from random_search import run_random_search, resolve_random_config

base_results_dir = config.get("base_results_dir", "./results")

experiment_name = config.get("experiment_name", "name_of_experiment")
max_llm_responses = config.get("max_llm_responses", 50)
min_nb_suggestions = config.get("min_nb_suggestions", 10)
nb_exec_init = config.get("nb_exec_init", 15)
min_runs_scheds = config.get("min_runs_scheds", 5)
max_runs_scheds = config.get("max_runs_scheds", 10)
time_budget_factor_scheds = config.get("time_budget_factor_scheds", 10)
top_p = config.get("top_p", 1)
temperature = config.get("temperature", 1)
nb_responses = config.get("nb_responses", 1)
enable_thinking = config.get("enable_thinking", True)

# Interaction format: "conversational" (default, a growing back-and-forth transcript
# resent each turn) or "compact" (a single message carrying the loop nest, the frozen
# analysis, and a summary of everything explored so far, to minimize token cost).
interaction_mode = config.get("interaction_mode", "conversational")
compact_mode = interaction_mode == "compact"

if args.experiment_name is not None:
    base_experiment_name = args.experiment_name
else:
    base_experiment_name = experiment_name

# update experiment name with relevant params
model_name = os.environ['LLM_ID'].split('/')[1] if '/' in os.environ['LLM_ID'] else os.environ['LLM_ID']
if random_mode:
    random_cfg = resolve_random_config(config.get("random_search"), max_llm_responses)
    experiment_name = f"{base_experiment_name}_random_N{random_cfg['nb_schedules']}_S{random_cfg['seed']}"
else:
    experiment_name = f"{base_experiment_name}_{model_name}_T{temperature}_TP{top_p}_L{min_nb_suggestions}_{'wT' if enable_thinking else 'woT'}"
    if compact_mode:
        experiment_name += "_compact"

# update the results dir before adding the run's id
results_dir = base_results_dir + '/' + experiment_name
if args.run_id is not None:
    experiment_name += '_run' + args.run_id

benchmarks_folder = os.environ["BENCHMARKS_DIR"]
benchmark_paths = sorted(list(Path(benchmarks_folder).iterdir()))

# Create the directory if it doesn't exist
Path(results_dir).mkdir(parents=True, exist_ok=True)

# --- Dialogue checkpointing (resume after a crash or interruption) ---
# The full dialogue state of an in-progress benchmark is saved after every
# iteration. A finished benchmark is recorded in the results JSON and its
# checkpoint deleted; an interrupted one keeps its checkpoint and is resumed
# mid-dialogue on the next run. Completed benchmarks (in the results JSON) are
# skipped entirely.
checkpoints_dir = f"{results_dir}/checkpoints"
Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
MAX_RESUME_ATTEMPTS = 5  # give up on a benchmark that keeps crashing on resume

def _ckpt_path(name):
    suffix = f"_run{args.run_id}" if args.run_id is not None else ""
    return f"{checkpoints_dir}/{name}{suffix}.pkl"

def save_checkpoint(name, state):
    path = _ckpt_path(name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint

def load_checkpoint(name):
    path = _ckpt_path(name)
    if not Path(path).is_file():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # corrupt checkpoint: start the benchmark fresh

def delete_checkpoint(name):
    Path(_ckpt_path(name)).unlink(missing_ok=True)

# save the parameters into a json file
params_file = f"{results_dir}/params_run{args.run_id}.json" if args.run_id is not None else f"{results_dir}/params.json"
with open(params_file, 'w') as f:
    params_dict = {
        'experiment_name': experiment_name,
        'max_llm_responses': max_llm_responses,
        'min_nb_suggestions': min_nb_suggestions,
        'top_p': top_p,
        'temperature': temperature,
        'nb_responses': nb_responses,
        'benchmarks': os.environ['BENCHMARKS_DIR'],
        'LLM_ID': os.environ['LLM_ID'] if not random_mode else None,
        'search_mode': search_mode,
        'random_search': random_cfg if random_mode else None,
        'script_file': sys.argv[0],
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.mktime(time.gmtime()) + 4 * 3600)), # in gmt+4
        'node_name': os.environ.get('SLURMD_NODENAME', 'localhost')
    }
    f.write(json.dumps(params_dict))

slurmd_nodename = os.environ.get('SLURMD_NODENAME', '')

per_bench_results_json = f'{results_dir}/results_{experiment_name}.json'
if Path(per_bench_results_json).is_file():
    with open(per_bench_results_json, 'r') as f:
        results_json = json.load(f)
else:
    results_json = {
        "configuration": {**config, "openai_api_key": "<redacted>",
                          **({"random_search": random_cfg} if random_mode else {})},
        "programs": {}
    }

for benchmark_path in benchmark_paths:
    benchmark_name = str(benchmark_path.parts[-1])
    if benchmark_name in results_json["programs"]:
        continue
    try:
        start = time.time()
        print(benchmark_name)
        cpp_code = Path(str(benchmark_path)+'/'+benchmark_name+'_generator.cpp').read_text()

        tiramisu_program = TiramisuProgram.init_server(
            cpp_code=cpp_code,
            load_annotations=True,
            load_tree=True,
            reuse_server=True,
        )

        schedule = Schedule(tiramisu_program)
        if (lookup_res := check_in_lookup_file('', tiramisu_program.name, 'initial_exec')) != 'Not found':
            initial_exec_time = lookup_res
            logger.debug(f'Initial execution time for {tiramisu_program.name} retrieved from lookup is {initial_exec_time}.')
        else:
            initial_exec_time=min(schedule.execute(nb_exec_init))
            update_lookup_file('', tiramisu_program.name, 'initial_exec', initial_exec_time)
        print(initial_exec_time)
        annots = tiramisu_program.annotations

        loop_nest_str, new_loop_names_map, compute_blocks_IDs = annots_to_c_prog(annots, True)
        first_message = f'''
    Here is the loop structure 

    {loop_nest_str}


    The initial execution time is {initial_exec_time}ms.

    Before you start suggesting transformations. Tell me what are the different loops of the program and the nesting depth of each (L0, L1, ...) and give me a brief analysis of the loop nest. Do not suggest transformations yet.
        '''
    
        if random_mode:
            # Deterministic per (seed, run id, benchmark), so best-of-K runs differ.
            random_seed = f"{random_cfg['seed']}-{args.run_id or 0}-{benchmark_name}"
            # Random-search baseline: no LLM, no dialogue. Already-explored schedules
            # are served from the lookup cache, so a rerun with the same seed after an
            # interruption replays quickly up to where it stopped.
            explored_scheds = []
            search_log = run_random_search(
                tiramisu_program, annots, compute_blocks_IDs, initial_exec_time,
                explored_scheds, random_cfg, seed=random_seed,
                min_runs=min_runs_scheds, max_runs=max_runs_scheds,
                time_budget=time_budget_factor_scheds*initial_exec_time,
            )
            messages = [{"role": "random_search", "content": line} for line in search_log]
            total_tokens_usage = {}
            elapsed_before = 0.0
        else:
            checkpoint = load_checkpoint(benchmark_name)
            if checkpoint is not None and checkpoint.get('resume_attempts', 0) >= MAX_RESUME_ATTEMPTS:
                print(f"[WARN] {benchmark_name} exceeded {MAX_RESUME_ATTEMPTS} resume attempts; giving up.")
                results_json["programs"][benchmark_name] = {"error": "exceeded max resume attempts"}
                with open(per_bench_results_json, "w") as f:
                    json.dump(results_json, f, indent=4)
                delete_checkpoint(benchmark_name)
                continue

            if checkpoint is not None:
                # Resume an interrupted dialogue from where it stopped.
                messages = checkpoint['messages']
                full_message = checkpoint['full_message']
                analysis = checkpoint.get('analysis')  # frozen program analysis (compact mode)
                explored_scheds = checkpoint['explored_scheds']
                llm_sched_strs_dict = checkpoint['llm_sched_strs_dict']
                stop_message = checkpoint['stop_message']
                giveups_counter = checkpoint['giveups_counter']
                response_id = checkpoint['response_id']
                total_tokens_usage = checkpoint['total_tokens_usage']
                elapsed_before = checkpoint.get('elapsed', 0.0)
                resume_attempts = checkpoint.get('resume_attempts', 0) + 1
                print(f"Resuming {benchmark_name} from response {response_id} (attempt {resume_attempts}).")
            else:
                # Fresh start: run the context-initialization exchange (identical in both
                # modes) to obtain the program analysis.
                messages, full_message, tokens_usage = initialize_conversation(system_instructions, first_message, enable_thinking=enable_thinking)
                total_tokens_usage = tokens_usage
                print_compiler_message(0, messages[0]['content']) # system instructions
                print_compiler_message(1, messages[1]['content']) # first message about doing a program analysis
                print_llm_message(1, full_message) # program analysis
                analysis = full_message  # frozen once; reused by every compact-mode turn

                explored_scheds = [] # list of tuples (llm_sched_str, outcome_type={'ran', 'illegal', 'crash', 'invalid'}, is_lookedup:Bool, outcome_data:{exec_time if ran, crash exception str if crashed, None if illegal}, log_reason:str)

                if compact_mode:
                    # Ask for the first suggestion via a single self-contained message
                    # (loop nest + analysis + empty log + start instruction).
                    first_compact = build_compact_message(loop_nest_str, initial_exec_time, analysis, explored_scheds, compact_start_instruction)
                    print_compiler_message(2, compact_start_instruction)
                    messages.append({"role": "user", "content": first_compact})
                    last_line, full_message, tokens_usage = get_compact_LLM_response(system_instructions, first_compact, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
                    messages.append({"role": "assistant", "content": full_message})
                else:
                    print_compiler_message(2, start_exploration_message) # ask to start exploration
                    last_line, full_message, tokens_usage = get_LLM_response(start_exploration_message, messages, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
                for k in total_tokens_usage: total_tokens_usage[k] += tokens_usage[k]
                print_llm_message(2, full_message) # first suggestion

                llm_sched_strs_dict = dict() # maps llm_sched_str -> compiler response, to avoid re-running the same schedule
                stop_message = False
                giveups_counter = 0
                response_id = 2
                elapsed_before = 0.0
                resume_attempts = 0

            def _checkpoint_state():
                return {
                    'messages': messages, 'full_message': full_message, 'analysis': analysis,
                    'explored_scheds': explored_scheds, 'llm_sched_strs_dict': llm_sched_strs_dict,
                    'stop_message': stop_message, 'giveups_counter': giveups_counter,
                    'response_id': response_id, 'total_tokens_usage': total_tokens_usage,
                    'elapsed': elapsed_before + (time.time() - start), 'resume_attempts': resume_attempts,
                }
            # Persist the (possibly incremented) attempt count before doing more work.
            save_checkpoint(benchmark_name, _checkpoint_state())

            while (not stop_message) and response_id < max_llm_responses:
                llm_sched_str = extract_transformation_commands(full_message)
                # Each branch decides what to print to the console (`outcome_text`) and what
                # to ask the LLM next (`instruction`). In conversational mode the instruction
                # is sent as the next turn; in compact mode it is the trailing instruction
                # appended to the rebuilt summary message (the per-schedule outcome is already
                # captured in the exploration log, so it is not resent as guidance).
                outcome_text = None
                instruction = None
                if llm_sched_str is None:
                    if full_message and 'no_further_transformations' in full_message:
                        instruction = (
                            "No schedule found. To stop, reply <schedule>no_further_transformations</schedule>; "
                            "otherwise put transformations inside the tags, e.g. <schedule>comp00.Parallelize(L0)</schedule>."
                        )
                    else:
                        instruction = (
                            "No schedule found in your reply. Put the full schedule on one line between "
                            "<schedule></schedule> tags, joining transformations with '+', e.g. "
                            "<schedule>comp00.Parallelize(L0)+comp00.Tile2D(L1,L2,32,32)</schedule>. "
                            "To stop, reply <schedule>no_further_transformations</schedule>."
                        )
                    outcome_text = instruction
                elif 'no_further_transformations' in llm_sched_str:
                    if len(explored_scheds) < min_nb_suggestions:
                        instruction = continue_optimization_messages[giveups_counter%len(continue_optimization_messages)]
                        giveups_counter+=1
                        outcome_text = instruction
                    else:
                        stop_message = True
                else: # a schedule is provided
                    is_duplicate = llm_sched_str in llm_sched_strs_dict
                    if not is_duplicate:
                        compiler_response = get_compiler_response(
                            llm_sched_str, tiramisu_program, annots, compute_blocks_IDs,
                            initial_exec_time, explored_scheds,
                            min_runs=min_runs_scheds, max_runs=max_runs_scheds,
                            time_budget=time_budget_factor_scheds*initial_exec_time
                        )
                        llm_sched_strs_dict[llm_sched_str] = compiler_response
                    else:
                        compiler_response = "Already tried " + llm_sched_str + ": " + llm_sched_strs_dict[llm_sched_str]
                    outcome_text = compiler_response
                    if compact_mode:
                        instruction = compact_next_instruction
                        if is_duplicate:
                            instruction = ("You proposed a schedule already in the exploration log above; "
                                           "pick a different, untried one. " + compact_next_instruction)
                    else:
                        instruction = compiler_response

                if not stop_message:
                    print_compiler_message(response_id, outcome_text)
                    if compact_mode:
                        compact_msg = build_compact_message(loop_nest_str, initial_exec_time, analysis, explored_scheds, instruction)
                        messages.append({"role": "user", "content": compact_msg})
                        last_line, full_message, tokens_usage = get_compact_LLM_response(system_instructions, compact_msg, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
                        messages.append({"role": "assistant", "content": full_message})
                    else:
                        last_line, full_message, tokens_usage = get_LLM_response(instruction, messages, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
                    for k in total_tokens_usage: total_tokens_usage[k] += tokens_usage[k]
                    response_id += 1
                    print_llm_message(response_id, full_message)

                # Save progress after every iteration so an interruption loses at most one step.
                save_checkpoint(benchmark_name, _checkpoint_state())

        end = time.time()
        optim_time = elapsed_before*1000 + (end-start)*1000 # in ms (includes time across resumes)
        explored_schedules_json = [{
            "schedule": "",
            "outcome": "run",
            "execution_time": initial_exec_time
        }]
        for sched in explored_scheds:
            sched_str, outcome, is_lookup, data = sched[:4]
            explored_schedules_json.append({
                "schedule": sched_str,
                "outcome": outcome,  # 'ran' | 'invalid' | 'illegal' | 'crash'
                "execution_time": data if outcome == "ran" else None
            })

        with open(f'{results_dir}/live_results_{experiment_name}.csv','a+') as f:
            # Move the pointer to the beginning of the file
            f.seek(0)
            if f.read(1) == '':
                #if emtpy write header
                f.write('benchmark, init_exec_time, nb_explored, nb_executed, optim_time, best_exec_time, best_speedup, best_sched,' + '\n')
            ran_scheds = [s for s in explored_scheds if s[1]=='ran']  # all executed schedules
            # The best schedule is the fastest one that actually beats the original; if
            # none does, the best choice is to keep the program unchanged (speedup 1x).
            improving = [s for s in ran_scheds if s[3] < initial_exec_time]
            if improving:
                best_sched = min(improving, key=lambda x:x[3])
                best_sched_str = best_sched[0]
                best_sched_exec = best_sched[3]
                best_json_sched = [best_sched_str, best_sched_exec]
                best_sched_speedup = initial_exec_time/best_sched_exec
            else:
                best_sched_str = ''
                best_sched_exec = initial_exec_time
                best_json_sched = ["", initial_exec_time]
                best_sched_speedup = 1
            f.write(f'{benchmark_name}, {initial_exec_time}, {len(explored_scheds)}, {len(ran_scheds)}, {optim_time}, {best_sched_exec}, {best_sched_speedup}, "{best_sched_str}"\n')

        results_json["programs"][benchmark_name] = {
            "Initial_execution_time": initial_exec_time,
            "Optimization_dialogue": messages,
            "Explored_schedules": explored_schedules_json,
            "Optimization_time": optim_time,
            "token_usage": total_tokens_usage,
            "Best_schedule": best_json_sched,
            "program_annotation": annots
        }
        with open(per_bench_results_json, 'w') as f:
            json.dump(results_json, f, indent=4)
        delete_checkpoint(benchmark_name)  # benchmark finished cleanly

    except Exception as bench_exc:
        # A single benchmark failing (e.g. a Tiramisu backend crash or an LLM
        # outage) must not abort the whole run. If a checkpoint exists the
        # dialogue can be resumed on a later run, so keep it and move on without
        # marking the benchmark done. If it failed before any checkpoint (e.g.
        # during setup), record the error so we don't retry it forever.
        traceback.print_exc()
        if load_checkpoint(benchmark_name) is not None:
            print(f"[WARN] {benchmark_name} interrupted ({bench_exc}); checkpoint kept for resume.")
        else:
            print(f"[WARN] {benchmark_name} failed during setup and was skipped: {bench_exc}")
            results_json["programs"][benchmark_name] = {"error": str(bench_exc), "traceback": traceback.format_exc()}
            with open(per_bench_results_json, "w") as f:
                json.dump(results_json, f, indent=4)
        continue

if args.completion_file != "0": # if a filename is provided for signaling completion, create it 
    with open(args.completion_file, "w") as f:
        pass
