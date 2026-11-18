"""ComPilot core utilities.

Everything the driver (compilot.py) needs, grouped by concern:
  - rendering a program's annotations as an anonymized C loop nest for the LLM;
  - parsing an LLM-proposed schedule string into a validated TiraLib Schedule,
    turning malformed proposals into precise validity errors before any compiler
    call (InvalidScheduleError);
  - running the compiler backend and turning each outcome (invalid / illegal /
    crash / ran) into concise, actionable feedback;
  - a small on-disk cache of results for previously seen schedules;
  - the LLM client and the two interaction formats (conversational and compact).
"""
from sympy import simplify
import re
from string import ascii_lowercase
import logging
import os
from pathlib import Path
import time
import pickle
import sys
from filelock import FileLock
import traceback
import json

import numpy as np
import openai
from openai import OpenAI

def parse_isl_map(isl_map: str):
    # Extract the computation and buffer parts
    match = re.match(r'.*{\s*(\w+)\[([^\]]+)\]\s*->\s*(\w+)\[([^\]]+)\]\s*}', isl_map)
    if not match:
        raise ValueError("Invalid ISL map format")
    
    computation_name = match.group(1)
    comp_iterators = match.group(2).split(',')
    buffer_name = match.group(3)
    buffer_accesses = match.group(4).split(',')
    
    # Strip any spaces around iterators or accesses
    comp_iterators = [it.strip() for it in comp_iterators]
    buffer_accesses = [access.strip() for access in buffer_accesses]
    
    return comp_iterators, buffer_accesses

def extract_affine_coefficients(access, comp_iterators):
    """ 
    Extract coefficients of iterators in the affine expression.
    Example: access = 'i1 + 2*j2', comp_iterators = ['i1', 'j2', 'k']
    Output: [1, 2, 0, 0] (coefficients for i1, j2, k, and scalar term)
    """
    # remove all spaces
    access = access.replace(' ', '')
    
    # Initialize coefficients (including one for the scalar term)
    coefficients = [0] * (len(comp_iterators) + 1)
    
    # Match terms like '2*i1' or 'i1' or '-i1', and extract the coefficients
    for i, it in enumerate(comp_iterators):
        # Modify the pattern to match iterator names with numbers or underscores
        term_pattern = re.compile(r'([+-]?\d*)\s*\*?\s*(' + re.escape(it) + r')(\b|$)')
        match = term_pattern.search(access)
        if match:
            coeff = match.group(1)
            coefficients[i] = int(coeff) if coeff and coeff != '+' and coeff != '-' else 1 if coeff == '' or coeff == '+' else -1
    
    # Handle the scalar part, which is just a constant not attached to any iterator
    scalar_pattern = re.compile(r'([+-]?\b\d+\b)(?![\*\w])')  # Match isolated constants (scalars) not tied to iterators
    scalar_match = scalar_pattern.search(access)
    
    if scalar_match:
        coefficients[-1] = int(scalar_match.group(1))
    
    return coefficients

def isl_map_to_write_access_matrix(isl_map: str):
    # Parse the ISL map
    comp_iterators, buffer_accesses = parse_isl_map(isl_map)
    
    # Initialize the access matrix (rows = buffer dimensions, columns = iterators + scalar)
    access_matrix = []
    
    # Process each buffer access and extract affine coefficients
    for access in buffer_accesses:
        row = extract_affine_coefficients(access, comp_iterators)
        access_matrix.append(row)
    
    return np.array(access_matrix).tolist()

def get_program_preorder_traversal(program_annotations, comp_or_loop_name='program_root', depth=-1):
    """ Return a list the preorder traversal of the program. Each element in the list is a dict containing basic info about a loop or computation
    """
    if comp_or_loop_name=='program_root': # if this is the first call, the children are the roots
        iterators_dict = program_annotations['iterators']
        children_names = [iter for iter in iterators_dict if iterators_dict[iter]['parent_iterator'] == None]
        children_names = [iter for iter in children_names if not (iterators_dict[iter]['lower_bound']=='0' and iterators_dict[iter]['upper_bound']=='1')] # skip the dummy iterators (extent of 1)
        type_ = 'loop'
    elif comp_or_loop_name in program_annotations['iterators']:
        iter_dict = program_annotations['iterators'][comp_or_loop_name]
        children_names = iter_dict['child_iterators']+iter_dict['computations_list']
        type_ = 'loop'
    elif comp_or_loop_name in program_annotations['computations']:
        children_names = []
        type_ = 'comp'
    else:
        raise ValueError(f"{comp_or_loop_name} is neither a computation nor a loop name")

    pre_order_traversal = [{'name':comp_or_loop_name, 'type':type_, 'depth':depth, 'nb_children':len(children_names)}]

    children_names = sorted(children_names, key=lambda x:get_absolute_order(x,program_annotations))

    for child_name in children_names:
        pre_order_traversal+=get_program_preorder_traversal(program_annotations, child_name, depth+1)
    return pre_order_traversal

def get_absolute_order(comp_or_loop_name,program_annotations):
    """ Returns the absolute order of computation or loop w.r.t the preorder traversal of the program
    """
    if comp_or_loop_name in program_annotations['iterators']:
        iter_dict = program_annotations['iterators'][comp_or_loop_name]
        if 'absolute_order' in iter_dict: # if previously computed
            return iter_dict['absolute_order']
        # since we want to use the abs order for getting preorder traversal, we make sure the loop's order is smaller than all its children's. 
        # This is done by the -0.001. This works assuming that comp's abs order are always ints and a loop can't have >999 nested descendants 
        abs_order = min([get_absolute_order(child,program_annotations) for child in iter_dict['child_iterators']+iter_dict['computations_list']])-0.001
        iter_dict['absolute_order'] = abs_order # as an optimizatin to avoid recomputing during the recursion
        return iter_dict['absolute_order']
    elif comp_or_loop_name in program_annotations['computations']:
        return program_annotations['computations'][comp_or_loop_name]['absolute_order']
    else:
        raise ValueError(f"{comp_or_loop_name} is neither a computation nor a loop name")

# retruns a string that represents a memory access. eg buf_19(l0,l1+1). If iterator names are given, it uses iterator names instead of loop levels
def get_access_str_from_access_matrix(buf_id, acc_mat, iterator_names = None, c_style = False):
    """
    Extract access relation str (eg "(i1 , i1 + 2*j2, k + 10)") from access matrix.
    If iterator names are given, it uses iterator names instead of loop levels
    :param access_matrix: as a list of lists
    :param comp_iterator_names: a list of iterator names where the access occurs
    :return: buf_+id+ access relation str with parentheses
    """

    dim_accesses = []
    if iterator_names is None:
        iterator_names = ['L'+str(i) for i in range(len(acc_mat[0]))]
    for row in acc_mat:
        dim_acc = ''
        for col,e in enumerate(row):
            if e!=0:
                if col!=len(row)-1:
                    dim_acc += ('+' if dim_acc!='' else '') + (str(e)+'*' if e!=1 else '') + iterator_names[col]
                else:
                    dim_acc += ('+' if e>0 else '') + str(int(e))

        dim_accesses.append(dim_acc if dim_acc!='' else '0')
    if c_style:
        reconstructed_str = 'buf_' + str(buf_id) + '[' + ']['.join(dim_accesses) + ']'
    else:
        reconstructed_str = 'buf_' + str(buf_id) + '(' + ','.join(dim_accesses) + ')'
    return reconstructed_str

sys.path.insert(1, os.getenv("TIRALIB_DIR"))
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

class ScheduleExecutionCrashed(Exception):
    pass

class KnownScheduleCreationCrash(Exception):
    """Base class for known, explainable schedule-creation errors."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidScheduleError(KnownScheduleCreationCrash):
    """Raised when a proposed schedule is syntactically or structurally invalid
    (malformed command, unknown comp_ID, out-of-range loop level, non-integer
    factor, etc.). These problems are caught before any compiler call and are
    reported back to the LLM as precise, actionable validity errors instead of
    opaque "crashed for unclear reasons" messages."""
    pass


# --- Schedule-string parsing helpers (turn malformed proposals into clear errors) ---
def _valid_comp_ids_str(compute_blocks_IDs):
    """Human-readable, comma-separated list of the valid comp_IDs of a program."""
    return ', '.join(sorted(compute_blocks_IDs.keys()))


def _resolve_comp(comp_id, compute_blocks_IDs, transf):
    """Map a comp_ID (e.g. comp00) to its underlying computation name, raising a
    clear InvalidScheduleError if the comp_ID does not exist in the program."""
    if comp_id not in compute_blocks_IDs:
        raise InvalidScheduleError(
            f'Unknown computation ID "{comp_id}" in "{transf}". '
            f'Valid IDs: {_valid_comp_ids_str(compute_blocks_IDs)}.'
        )
    return compute_blocks_IDs[comp_id][0]


def _parse_level(loop_token, real_comp_name, schedule, transf, comp_id):
    """Parse a loop-level token (e.g. "L1", "L-1") into an integer level and
    validate it against the loops actually enclosing the computation. Raises a
    descriptive InvalidScheduleError on any problem."""
    if not loop_token or loop_token[0] != 'L':
        raise InvalidScheduleError(
            f'Invalid loop level "{loop_token}" in "{transf}": use L<depth>, '
            f'e.g. L0 (outermost) or L-1 (innermost).'
        )
    try:
        level = int(loop_token[1:])
    except ValueError:
        raise InvalidScheduleError(
            f'Invalid loop level "{loop_token}" in "{transf}": L must be followed by an '
            f'integer (L0, L1, ..., or L-1 for the innermost loop).'
        )
    innermost = get_innermost_level(real_comp_name, schedule)
    if level == -1:
        return innermost
    if level < 0 or level > innermost:
        raise InvalidScheduleError(
            f'Loop level {loop_token} does not exist for {comp_id} in "{transf}": '
            f'valid levels are L0 to L{innermost} (or L-1 for the innermost).'
        )
    return level


def _parse_call_args(transf_inst, transf_id, expected_n, transf, usage_example):
    """Validate the `Name(arg1, arg2, ...)` structure of a transformation command
    and return the list of raw argument strings. Raises InvalidScheduleError with
    a concrete usage example when the parentheses or argument count are wrong."""
    if not (len(transf_inst) > len(transf_id)
            and transf_inst[len(transf_id)] == '(' and transf_inst[-1] == ')'):
        raise InvalidScheduleError(
            f'Malformed command "{transf}": arguments must be in parentheses. Expected: {usage_example}.'
        )
    params = transf_inst[len(transf_id) + 1:-1]
    split_params = params.split(',')
    non_empty = [p for p in split_params if p != '']
    if len(split_params) != expected_n or len(non_empty) != expected_n:
        raise InvalidScheduleError(
            f'Wrong number of arguments in "{transf}": expected {expected_n}, got '
            f'{len(non_empty)}. Expected: {usage_example}.'
        )
    return split_params


def _parse_int_arg(token, transf, what='factor'):
    """Parse an integer transformation argument (tiling / unrolling factor)."""
    try:
        return int(token)
    except (ValueError, TypeError):
        raise InvalidScheduleError(
            f'Invalid {what} "{token}" in "{transf}": must be an integer.'
        )
    
# --- Module setup: LLM client, TiraLib backend, caches, logging ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"), max_retries=10)

BaseConfig.init(config_yaml=os.getenv("TIRALIB_DIR") + "/config.yaml")

lookup_files_dir = './lookup_dir_'+os.getenv("LOOKUP_DIR_SUFFIX")
Path(lookup_files_dir).mkdir(exist_ok=True)

llm_id = os.getenv("LLM_ID")

def _endpoint_kind():
    """Which OpenAI-compatible provider BASE_URL points at: 'gemini', 'anthropic' or 'other'."""
    base_url = os.getenv("BASE_URL") or ""
    if "generativelanguage.googleapis.com" in base_url:
        return "gemini"
    if "api.anthropic.com" in base_url:
        return "anthropic"
    return "other"

def _adapt_request_kwargs(kwargs):
    """Drop or adjust request fields a provider's OpenAI-compatible endpoint rejects.
    - Gemini: rejects unknown fields with a 400 (`seed`, vLLM's `chat_template_kwargs`);
      Gemini Flash models think by default.
    - Anthropic (https://platform.claude.com/docs/en/api/openai-sdk): on Claude 4.7+
      models any `temperature` below 1 or `top_p` other than the default returns a 400,
      as does sending both; `seed` is ignored. Thinking is always on for Claude 5 models,
      so `enable_thinking` has no effect. `max_tokens` is set explicitly so thinking plus
      the answer is not cut short.
    - Other endpoints (vLLM, SGLang, ...): sent unchanged."""
    kind = _endpoint_kind()
    if kind == "gemini":
        kwargs.pop("seed", None)
        kwargs.pop("extra_body", None)
    elif kind == "anthropic":
        for k in ("temperature", "top_p", "seed", "extra_body"):
            kwargs.pop(k, None)
        kwargs.setdefault("max_tokens", 16000)
    return kwargs

def _thinking_extra_body(enable_thinking):
    """`chat_template_kwargs` toggles thinking on vLLM/SGLang-served models (e.g. Qwen).
    Providers that reject it drop it in `_adapt_request_kwargs`."""
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

# Whether to reuse cached compiler feedback messages across schedules/runs.
# The numeric caches (initial execution time and skewing solver factors) are
# always kept because they are format-agnostic and expensive to recompute; only
# the textual compiler-response cache is gated here. Disabling it guarantees that
# every schedule gets a freshly generated message (important when comparing runs
# after changing the feedback wording, and for honest stochastic multi-run
# experiments). Controlled by the USE_RESPONSE_LOOKUP env var ("0" to disable).
RESPONSE_LOOKUP_ENABLED = os.getenv("USE_RESPONSE_LOOKUP", "1") != "0"

# When an illegal *compound* schedule is found, optionally pinpoint the offending
# transformation by legality-checking each component on its own. This is helpful
# feedback but costs one extra legality check per component, which is noticeable
# on large programs, so it is off by default. Controlled by LOCALIZE_ILLEGALITY.
LOCALIZE_ILLEGALITY = os.getenv("LOCALIZE_ILLEGALITY", "0") == "1"

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("|%(asctime)s|%(levelname)s| %(message)s")
)
logger.addHandler(console_handler)

# --- Rendering a program's annotations as an anonymized C loop nest ---
def expression_json_to_str(expr, iterator_names=None):
    operators = {
        'add': ('+', 2),  # Binary
        'sub': ('-', 2),  # Binary
        'mul': ('*', 2),  # Binary
        'div': ('/', 2),  # Binary
        'eq': ('==', 2),  # Binary
        'le': ('<=', 2),  # Binary
        'max': ('max', 2),  # Binary (with special formatting)
        'min': ('min', 2),  # Binary (with special formatting)
        'cast': ('cast', 1),  # Unary (with special formatting)
        'sqrt_f32': ('sqrt', 1),  # Unary (with special formatting)
        'sqrt_f64': ('sqrt', 1),  # Unary (with special formatting)
        'access': (None,0),
        'val': (None,0),
    }
    assert expr['expr_type'] in operators

     # Check if the token is an operator
    operator, arity = operators[expr['expr_type']]

    if arity == 1:  # Unary operator
        operand = expression_json_to_str(expr['children'][0],iterator_names)
        if expr['expr_type'].startswith('sqrt'):
            return f"sqrt({operand})"
        elif expr['expr_type'] == 'cast':
            return f"({expr['dtype']})({operand})"

    elif arity == 2:  # Binary operator
        operand1 = expression_json_to_str(expr['children'][0],iterator_names)
        operand2 = expression_json_to_str(expr['children'][1],iterator_names)

        if expr['expr_type'] in ['max', 'min']:
            # Special case for max and min
            return f"{operator}({operand1}, {operand2})"
        else:
            # Wrap in parentheses for proper order of operations
            return f"({operand1} {operator} {operand2})"
    elif arity==0:
        if expr['expr_type'] == 'access':
            return get_access_str_from_access_matrix(expr['buffer_id'], expr['access_matrix'], iterator_names=iterator_names)
        elif expr['expr_type'] == 'val':
            return str(int(expr['str'])) if 'int' in expr['dtype'] else str(float(expr['str']))
            
def expression_str_to_c_expression(expression_str):
    try:
        simplified_str = str(simplify(expression_str))
    except TypeError: # sympy can crash when there are casts or le 
        simplified_str = expression_str
    return re.sub(r'(\w+)\(([\s\w\d_,\-+*]+)\)', r'\g<1>[\g<2>]', simplified_str).replace(', ','][') 


def expression_json_to_c_expression(expr, iterator_names=None):
    return expression_str_to_c_expression(expression_json_to_str(expr, iterator_names))

def for_loop_str(iter_name, program_annotations):
    iter_dict = program_annotations['iterators'][iter_name]
    return f'for(int {iter_name}={str(simplify(iter_dict["lower_bound"]))}; {iter_name}<{str(simplify(iter_dict["upper_bound"]))}; {iter_name}++)'

def annots_to_c_prog(program_annotations, anonymize_loop_names = False):
    preorder_traversal = get_program_preorder_traversal(program_annotations)
    lines = []
    prev_depth = -1
    compute_blocks_IDs = dict() # a dict of comp_block_id:[list_comp_names_in_block]
    curr_comp_block_id = None
    
    for elem in preorder_traversal[1:]:
        while prev_depth>elem['depth']:
            prev_depth-=1
            lines.append('    '*prev_depth+'}')
            
        if elem['type'] == 'loop':
            curr_comp_block_id = None
                
            elem_str = '    '*elem['depth'] + for_loop_str(elem['name'],program_annotations) + '{'
            
            if program_annotations['iterators'][elem['name']]['computations_list']: # if loop has comps, make a comp block ID
                curr_comp_block_id = 'comp'+str(len(compute_blocks_IDs)).zfill(2)
                compute_blocks_IDs[curr_comp_block_id] = program_annotations['iterators'][elem['name']]['computations_list']
                elem_str += '\n'+'    '*(elem['depth']+1) + "// comp_ID: " + curr_comp_block_id

           
            
        else: # if comp
            elem_str = '    '*elem['depth'] + annots_to_c_comp(elem['name'], program_annotations) + ";"
        prev_depth = elem['depth']  
        lines.append(elem_str)
        
    while prev_depth>0:
        prev_depth-=1
        lines.append('    '*prev_depth+'}')
        
        
    loop_nest_str = '\n'.join(lines)
    
    if anonymize_loop_names:
        new_loop_names_map = {ascii_lowercase[d]:orig_name for d,orig_name in enumerate([elem['name'] for elem in preorder_traversal[1:] if elem['type'] == 'loop'])}
        for new_name, orig_name in new_loop_names_map.items():
            loop_nest_str = re.sub(fr'\b{orig_name}\b', new_name, loop_nest_str)
        return loop_nest_str, new_loop_names_map, compute_blocks_IDs
    else:
        return loop_nest_str, {}, compute_blocks_IDs

def annots_to_c_comp(comp_name, program_annotations):
    comp_dict = program_annotations['computations'][comp_name]
    write_acc_mat = isl_map_to_write_access_matrix(comp_dict['write_access_relation'])
    lhs_str = get_access_str_from_access_matrix(comp_dict['write_buffer_id'], write_acc_mat, comp_dict['iterators'])
    lhs_str = expression_str_to_c_expression(lhs_str)
    rhs_str = expression_json_to_c_expression(comp_dict['simplified_expression_representation'], comp_dict['iterators'])
    return lhs_str + ' = ' + rhs_str

# --- Building a validated TiraLib Schedule from an LLM-proposed schedule string ---
def llm_sched_str_to_sched(llm_sched_str, program, program_annots, compute_blocks_IDs=None):
    """Parse an LLM schedule string into a TiraLib Schedule, validating each
    transformation and raising InvalidScheduleError with an actionable message on
    any malformed or structurally-invalid command."""
    schedule = Schedule(program)
    llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
    llm_transfs = llm_sched_str.split('+')
    for transf_idx,transf in enumerate(llm_transfs):
        if transf.count('.') != 1:
            raise InvalidScheduleError(
                f'Malformed command "{transf}": expected comp_ID.Transformation(args), '
                f'e.g. comp00.Parallelize(L0). Combine transformations with "+".'
            )
        comp_id, transf_inst = tuple(transf.split('.'))
        real_comp_name = _resolve_comp(comp_id, compute_blocks_IDs, transf)
        if transf_inst.lower().startswith('fuse'):
            transf_id = 'fuse'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Fuse(other_comp_ID, L<level>)')
            second_comp_id = split_transf_params[0]
            second_comp_real_name = _resolve_comp(second_comp_id, compute_blocks_IDs, transf)
            if second_comp_real_name == real_comp_name:
                raise InvalidScheduleError(
                    f'Cannot fuse {comp_id} with itself in "{transf}": the two computation IDs must be distinct.')
            loop_level = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            second_innermost = get_innermost_level(second_comp_real_name, schedule)
            if loop_level > second_innermost:
                raise InvalidScheduleError(
                    f'Cannot fuse at L{loop_level} in "{transf}": {second_comp_id} only has levels '
                    f'L0 to L{second_innermost}. Fuse at a level shared by both computations.')
            loop_id1 = (real_comp_name,loop_level)
            loop_id2 = (second_comp_real_name,loop_level)
            new_optim = tiramisu_actions.Fusion([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('parallelize'):
            transf_id = 'parallelize'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 1, transf, 'comp_ID.Parallelize(L<level>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            if loop_level>=2:
                raise InvalidScheduleError(
                    f'Parallelization in "{transf}" targets L{loop_level}, but only outer loops '
                    f'(L0 or L1) should be parallelized (deeper loops add too much threading overhead).')
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Parallelization([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('reverse'):
            transf_id = 'reverse'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 1, transf, 'comp_ID.Reverse(L<level>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Reversal([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])


        elif transf_inst.lower().startswith('interchange'):
            transf_id = 'interchange'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Interchange(L<level1>, L<level2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            new_optim = tiramisu_actions.Interchange([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('skew'):
            transf_id = 'skew'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Skew(L<level1>, L<level2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)

            # Validate the skewing preconditions before invoking the (expensive)
            # solver that computes the skewing factors. The action is initialized
            # against the current tree with placeholder factors (0, 0) purely for
            # this validity check.
            temp_skew_action = tiramisu_actions.Skewing([loop_id1,loop_id2,0,0])
            temp_skew_action.initialize_action_for_tree(schedule.tree)
            assert is_new_optim_valid(schedule, temp_skew_action, llm_transfs[:transf_idx+1])


            lookup_res=check_in_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver')
            if lookup_res!='Not found':
                skewing_factors = lookup_res
                logger.debug(f"Skewing solu for {'+'.join(llm_transfs[:transf_idx+1])} retrieved from lookup are {skewing_factors}.")
            else:
                loop_level1_node = schedule.tree.get_iterator_of_computation(real_comp_name, 1)
                involved_computations = schedule.tree.get_iterator_subtree_computations(loop_level1_node.id)
                skewing_factors = tiramisu_actions.Skewing.get_factors(schedule, [loop_level1, loop_level2], involved_computations)
                update_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver', skewing_factors)

            if skewing_factors is None:
                raise InvalidScheduleError(
                    f'No valid skewing factors found for "{transf}"; try a different loop pair '
                    f'or drop the skewing.')
            factor1, factor2 = skewing_factors
            schedule.add_optimizations([tiramisu_actions.Skewing([loop_id1,loop_id2,factor1,factor2])])

        elif transf_inst.lower().startswith('tile2d'):
            transf_id = 'tile2d'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 4, transf,
                'comp_ID.Tile2D(L<level1>, L<level2>, <factor1>, <factor2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            factor1 = _parse_int_arg(split_transf_params[2], transf, 'tiling factor')
            factor2 = _parse_int_arg(split_transf_params[3], transf, 'tiling factor')
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            new_optim = tiramisu_actions.Tiling2D([loop_id1,loop_id2,factor1,factor2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('tile3d'):
            transf_id = 'tile3d'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 6, transf,
                'comp_ID.Tile3D(L<level1>, L<level2>, L<level3>, <factor1>, <factor2>, <factor3>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_level3 = _parse_level(split_transf_params[2], real_comp_name, schedule, transf, comp_id)
            factor1 = _parse_int_arg(split_transf_params[3], transf, 'tiling factor')
            factor2 = _parse_int_arg(split_transf_params[4], transf, 'tiling factor')
            factor3 = _parse_int_arg(split_transf_params[5], transf, 'tiling factor')
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            loop_id3 = (real_comp_name,loop_level3)
            new_optim = tiramisu_actions.Tiling3D([loop_id1,loop_id2,loop_id3,factor1,factor2,factor3])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('unroll'):
            transf_id = 'unroll'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Unroll(L<level>, <factor>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            factor = _parse_int_arg(split_transf_params[1], transf, 'unrolling factor')
            unroll_limit = _over_unroll_limit(real_comp_name, loop_level, schedule)
            if unroll_limit is not None and factor > unroll_limit:
                raise InvalidScheduleError(
                    f'Unrolling factor {factor} in "{transf}" exceeds the loop trip count '
                    f'({unroll_limit}); use a factor between 2 and {unroll_limit}.')
            loop_id = (real_comp_name,loop_level)
            new_optim = tiramisu_actions.Unrolling([loop_id,factor])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
        elif transf_inst.lower().startswith('distribute'):
            transf_id = 'distribute'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 1, transf, 'comp_ID.Distribute(L<level>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            groups = _distribution_groups(real_comp_name, loop_level, schedule, compute_blocks_IDs)
            if len(groups) < 2:
                raise InvalidScheduleError(
                    f'Nothing to distribute in "{transf}": loop L{loop_level} of {comp_id} has a '
                    f'single body (one comp_ID block and no other nested loop). Distribute a loop '
                    f'that contains several comp_ID blocks or nested loops.')
            new_optim = tiramisu_actions.Distribution([(real_comp_name, loop_level)], groups)
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
        else:
            transf_name = transf_inst.split('(')[0]
            raise InvalidScheduleError(
                f'Unknown transformation "{transf_name}" in "{transf}". Supported: Fuse, '
                f'Distribute, Interchange, Parallelize, Tile2D, Tile3D, Unroll, Skew, Reverse.')


    return schedule

def _distribution_groups(real_comp_name, loop_level, schedule, compute_blocks_IDs):
    """Groups of computations that each get their own copy of the loop when the
    loop at `loop_level` enclosing `real_comp_name` is distributed: one group per
    nested loop, and one per comp_ID block placed directly in the loop body (the
    computations of a block always stay together)."""
    tree = schedule.tree
    iterator = tree.iterators[tree.get_iterator_of_computation(real_comp_name, loop_level).id]
    groups = [list(tree.get_iterator_subtree_computations(child)) for child in iterator.child_iterators]
    block_of = {comp: block for block, comps in compute_blocks_IDs.items() for comp in comps}
    direct = {}
    for comp in iterator.computations_list:
        direct.setdefault(block_of.get(comp, comp), []).append(comp)
    groups.extend(direct.values())
    return groups

# --- Loop-extent helpers and the over-unrolling guard ---
def get_innermost_level(real_comp_name, schedule):
    return schedule.tree.get_iterator_of_computation(real_comp_name).level


def _static_loop_extent(real_comp_name, loop_level, schedule):
    """Return the static trip count of the loop at `loop_level` enclosing the
    computation, if statically known, else None.

    NOTE on correctness: this reads the *current* schedule tree. The tree is
    regenerated from the real ISL AST after tiling/fusion/distribution, and
    reversal/parallelization/skewing do not reorder levels (skewing turns bounds
    non-rectangular, in which case this returns None). The only transformation
    that permutes level->extent without refreshing the tree is interchange; call
    sites must account for that (see `_max_enclosing_static_extent`)."""
    try:
        node = schedule.tree.get_iterator_of_computation(real_comp_name, loop_level)
    except Exception:
        return None
    lb, ub = node.lower_bound, node.upper_bound
    if isinstance(lb, int) and isinstance(ub, int):
        return ub - lb
    return None


def _max_enclosing_static_extent(real_comp_name, schedule):
    """Return the maximum statically-known trip count among all loops enclosing
    the computation, or None if none are statically known.

    This is invariant under interchange (which only permutes the enclosing
    loops), so `factor > max_enclosing_extent` is a *sound* over-unrolling test
    even when the tree's per-level mapping is stale after an interchange."""
    extents = []
    try:
        node = schedule.tree.get_iterator_of_computation(real_comp_name)
    except Exception:
        return None
    while node is not None:
        lb, ub = node.lower_bound, node.upper_bound
        if isinstance(lb, int) and isinstance(ub, int):
            extents.append(ub - lb)
        parent_id = node.parent_iterator
        node = schedule.tree.iterators.get(parent_id) if parent_id else None
    return max(extents) if extents else None


def _over_unroll_limit(real_comp_name, loop_level, schedule):
    """Return an upper bound U such that unrolling by a factor > U is guaranteed
    to exceed the target loop's trip count (and therefore crash the Tiramisu code
    generator). Returns None when no safe bound can be established, in which case
    the guard abstains (the schedule is passed through and any resulting backend
    crash is reported with recovery guidance instead of being wrongly rejected).

    When no interchange has been applied, the per-level extent is accurate and
    used directly (this catches e.g. tiling-then-unrolling a small tile loop).
    When an interchange is present the level->extent mapping in the tree may be
    stale, so the interchange-invariant max enclosing extent is used instead."""
    has_interchange = any(
        getattr(op, 'is_interchange', lambda: False)() for op in schedule.optims_list
    )
    if not has_interchange:
        level_extent = _static_loop_extent(real_comp_name, loop_level, schedule)
        if level_extent is not None:
            return level_extent
    return _max_enclosing_static_extent(real_comp_name, schedule)

# --- On-disk cache of results for previously seen schedules ---
def check_in_lookup_file(llm_sched_str, prog_name, query_type):
    assert query_type in ['initial_exec', 'compiler_responses', 'skewing_solver']
    if query_type == 'compiler_responses' and not RESPONSE_LOOKUP_ENABLED:
        return 'Not found'
    prog_lookup_file = f'{lookup_files_dir}/{prog_name}_lookup.pkl'
    prog_lookup_lock = FileLock(f'{lookup_files_dir}/{prog_name}_lookup.lock')
    with prog_lookup_lock:
        if Path(prog_lookup_file).is_file():
            with open(prog_lookup_file, 'rb') as f:
                lookup_dict = pickle.load(f)
        else: 
            lookup_dict = {'initial_exec':'Not found' , 'compiler_responses':{}, 'skewing_solver':{}}
    normalized_llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
    if query_type == 'initial_exec':
        return lookup_dict[query_type]
    elif normalized_llm_sched_str in lookup_dict[query_type]:
        return lookup_dict[query_type][normalized_llm_sched_str]
    else:
        return 'Not found'

def update_lookup_file(llm_sched_str, prog_name, query_type, value):
    # value should be of the format 
    #   (compliler_response_str, outcome_type={'ran', 'illegal', 'crashed'},  outcome_data:{exec_time if ran, crash exception str if crashed, None if illegal}) if query_type==compiler_responses
    #   exec_time as float if query_type==compiler_responses
    #   skewing_sols_tuple if query_type==skewing_solver and solutions exist, else 'Not found'
    assert query_type in ['initial_exec', 'compiler_responses', 'skewing_solver']
    if query_type == 'compiler_responses' and not RESPONSE_LOOKUP_ENABLED:
        return
    prog_lookup_file = f'{lookup_files_dir}/{prog_name}_lookup.pkl'
    prog_lookup_lock = FileLock(f'{lookup_files_dir}/{prog_name}_lookup.lock')
    with prog_lookup_lock:
        # get the latest version of the file first
        if Path(prog_lookup_file).is_file():
            with open(prog_lookup_file, 'rb') as f:
                lookup_dict = pickle.load(f)
        else: 
            lookup_dict = {'initial_exec':'Not found' , 'compiler_responses':{}, 'skewing_solver':{}}
                
        normalized_llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
        if query_type == 'initial_exec':
            lookup_dict[query_type] = value
        elif normalized_llm_sched_str in lookup_dict[query_type]:
            # Entry already present: two parallel runs collided on the same new
            # schedule. Keep the existing value (idempotent).
            pass
        else:
            lookup_dict[query_type][normalized_llm_sched_str] = value
        with open(prog_lookup_file, 'wb') as f:
            pickle.dump(lookup_dict, f)

# --- Compiler backend and feedback generation ---
def _localize_illegality(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs):
    """For a compound schedule found illegal, help the LLM localize the culprit by
    legality-checking each of its transformations on its own. Returns a short
    guidance string (starting with a space, or empty). Best-effort and cheap
    (legality checks do not execute code); never raises."""
    transfs = [t for t in llm_sched_str.split('+') if t]
    if len(transfs) < 2:
        return ''
    individually_illegal = []
    individually_legal = []
    try:
        for t in transfs:
            try:
                sched = llm_sched_str_to_sched(t, tiramisu_program, annots, compute_blocks_IDs)
                if sched.is_legal():
                    individually_legal.append(t)
                else:
                    individually_illegal.append(t)
            except tiramisu_actions.CannotApplyException as e:
                if 'illegal schedule prefix' in str(e):
                    individually_illegal.append(t)
            except Exception:
                # This single transformation can't be cleanly isolated; skip it.
                continue
    except Exception:
        return ''
    if individually_illegal:
        return ' Illegal on their own: ' + ', '.join(f'"{t}"' for t in individually_illegal) + '.'
    if individually_legal and len(individually_legal) == len(transfs):
        return ' Each transformation is legal alone; the combination is the problem.'
    return ''


def _derive_log_reason(outcome_type, outcome_data, initial_exec_time):
    """Short, display-ready factual outcome for the compact exploration log. Used
    for the 'ran' branch and to render legacy cache entries that predate the stored
    reason. Guidance is intentionally omitted here (added once in the instruction)."""
    if outcome_type == 'ran':
        speedup = initial_exec_time / outcome_data
        if speedup >= 1:
            return f'runnable, {round(speedup, 2)}x speedup'
        return f'runnable, {round(speedup, 2)}x (slowdown {round(1 / speedup, 2)}x)'
    if outcome_type == 'invalid':
        return f'invalid: {outcome_data}'
    if outcome_type == 'illegal':
        return "illegal: violates the program's data dependencies"
    return 'crashed: compiler crash'  # outcome_type == 'crash'


def get_compiler_response(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs, initial_exec_time, explored_scheds, min_runs=10, max_runs=None, time_budget=None):
    # Each explored schedule is recorded as a 5-tuple:
    #   (sched_str, outcome_type, is_lookedup, outcome_data, log_reason)
    # `outcome_data` keeps the raw datum (exec time / error message / traceback);
    # `log_reason` is a short, display-ready factual outcome used by the compact-mode
    # exploration log (guidance is added once, not repeated per entry).
    prog_name = tiramisu_program.name
    if (lookup_res := check_in_lookup_file(llm_sched_str, prog_name, 'compiler_responses')) != 'Not found':
        logger.debug(f'Response for {llm_sched_str} retrieved from lookup.')
        message, outcome_type, outcome_data = lookup_res[:3]
        reason = lookup_res[3] if len(lookup_res) > 3 else _derive_log_reason(outcome_type, outcome_data, initial_exec_time)
        explored_scheds.append((llm_sched_str, outcome_type, True, outcome_data, reason))
        return message
    illegal_prefix = False
    try:
        schedule = llm_sched_str_to_sched(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs)
    except tiramisu_actions.CannotApplyException as e:
        # Fusion, distribution and tiling rebuild the loop tree from the
        # compiler's AST, which is refused once the schedule so far is illegal.
        if 'illegal schedule prefix' not in str(e):
            raise
        illegal_prefix = True
    except InvalidScheduleError as e:
        # A schedule that is syntactically/structurally invalid. Caught before any
        # compiler call and reported to the LLM with a precise, actionable reason.
        reason = f'invalid: {e.message}'
        explored_scheds.append((llm_sched_str, 'invalid', False, e.message, reason))
        message = f'Invalid schedule "{llm_sched_str}" (not applied). {e.message}'
        logger.info(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'invalid', e.message, reason))
        return message
    except KnownScheduleCreationCrash as e:
        reason = 'crashed: could not be applied by the compiler'
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc(), reason))
        message = f'"{llm_sched_str}" could not be applied. {e.message}'
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc(), reason))
        return message
    except Exception as e:
        reason = 'crashed: compiler crash while building (too-aggressive combination)'
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc(), reason))
        message = (f'"{llm_sched_str}" crashed the compiler while building ({type(e).__name__}). '
                   f'Likely an aggressive combination — reduce tiling/unrolling factors, apply '
                   f'fusion first and unrolling last, or drop a transformation.')
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc(), reason))
        return message
    try:
        is_legal = False if illegal_prefix else schedule.is_legal()
    except Exception as e:
        reason = 'crashed: compiler crash during the legality check'
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc(), reason))
        message = (f'"{llm_sched_str}" crashed the compiler during the legality check. '
                   f'Simplify or reorder the schedule (fusion first, unrolling last).')
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc(), reason))
        return message
    if is_legal:
        try:
            exec_time = min(schedule.execute(min_runs=min_runs,max_runs=max_runs,time_budget=time_budget))
            speedup = initial_exec_time/exec_time
            reason = _derive_log_reason('ran', exec_time, initial_exec_time)
            explored_scheds.append((llm_sched_str, 'ran', False, exec_time, reason))

            if speedup>=1:
                message = f'{llm_sched_str} ran successfully: speedup {round(speedup,2)}x.'
            else:
                message = f'{llm_sched_str} ran successfully: speedup {round(speedup,2)}x (slowdown {round(1/speedup,2)}x).'

            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'ran', exec_time, reason))
            return message

        except ScheduleExecutionCrashed as e:
            reason = 'crashed: legal but segfaulted at runtime'
            explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc(), reason))
            message = (f'"{llm_sched_str}" was legal but segfaulted at runtime. Often caused by a '
                       f'too-large unroll/tiling factor or unrolling a non-innermost loop — reduce '
                       f'factors or unroll only the innermost loop (L-1).')
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc(), reason))
            return message

        except Exception as e:
            reason = f'crashed: legal but crashed at runtime ({type(e).__name__})'
            explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc(), reason))
            message = (f'"{llm_sched_str}" was legal but crashed at runtime ({type(e).__name__}). '
                       f'Often caused by a too-large unroll/tiling factor or unrolling a non-innermost '
                       f'loop — reduce factors or unroll only the innermost loop (L-1).')
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc(), reason))

            return message

    else:
        reason = 'illegal: violates the program\'s data dependencies'
        explored_scheds.append((llm_sched_str, 'illegal', False, None, reason))
        localization_hint = (_localize_illegality(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs)
                             if LOCALIZE_ILLEGALITY else '')
        message = (f'"{llm_sched_str}" is illegal: it violates the program\'s data dependencies, so '
                   f'it was not executed.{localization_hint} Revoke or replace the responsible '
                   f'transformation(s), or reorder them.')
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'illegal', None, reason))
        return message



# --- LLM client: transient-error retry with backoff, and the interaction turns ---
# Error types that are always transient (network blips, timeouts, rate limits, 5xx).
_TRANSIENT_LLM_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
# HTTP statuses worth retrying when returned by the endpoint or a gateway in front
# of it (request timeout, too-early, rate limit, and 5xx gateway errors).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_transient(exc):
    """Whether an LLM request failure is worth retrying. Covers network/timeout/
    rate-limit/5xx errors and empty responses. A tunnel or gateway (e.g. ngrok, a
    load balancer) fronting the model can briefly return 404/5xx while the backend
    restarts, so those are retried too; a genuine 404 (wrong URL/model) is not."""
    if isinstance(exc, _TRANSIENT_LLM_ERRORS + (ValueError,)):
        return True
    status = getattr(exc, 'status_code', None)
    if status in _RETRYABLE_STATUS:
        return True
    if status == 404 and any(m in str(exc).lower()
                             for m in ('offline', 'ngrok', 'temporarily', 'unavailable', 'bad gateway')):
        return True
    return False


def _chat_completion(messages, max_attempts=60, **kwargs):
    """Call the chat completion endpoint, retrying transient failures with capped
    exponential backoff so external hiccups don't abort a run. Returns
    (assistant_message, tokens_usage). Raises after `max_attempts`, or immediately
    on a non-transient error (e.g. authentication or a malformed request)."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(model=llm_id, messages=messages, **_adapt_request_kwargs(dict(kwargs)))
            content = response.choices[0].message.content if response.choices else None
            if content is None:
                raise ValueError("empty response content")
            tokens_usage = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            return content, tokens_usage
        except Exception as e:
            if attempt == max_attempts or not _is_transient(e):
                if _is_transient(e):
                    logger.error(f"LLM request still failing after {max_attempts} attempts: {e}")
                raise
            wait = min(2 ** min(attempt, 6), 60)
            logger.warning(f"LLM request failed ({type(e).__name__}: {e}); retry {attempt}/{max_attempts} in {wait}s")
            time.sleep(wait)


def get_LLM_response(compiler_response, messages, top_p = 1, temperature=1, enable_thinking = True):
    """Conversational turn: append the compiler feedback, query the LLM, append its
    reply. `messages` grows in place (the full transcript is resent each turn)."""
    messages.append({"role": "user", "content": compiler_response})
    new_assistant_message, tokens_usage = _chat_completion(
        messages,
        top_p=top_p,
        temperature=temperature,
        extra_body=_thinking_extra_body(enable_thinking),
    )
    messages.append({"role": "assistant", "content": new_assistant_message})
    last_line = new_assistant_message.splitlines()[-1] if new_assistant_message else ''
    return last_line, new_assistant_message, tokens_usage


def initialize_conversation(system_instructions, first_message, temperature=0, seed=42, enable_thinking=True):
    """Start a dialogue: send the system prompt and the first (program-analysis)
    message, and return (messages, analysis_reply, tokens_usage)."""
    messages=[
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": first_message}
    ]
    new_assistant_message, tokens_usage = _chat_completion(
        messages,
        temperature=temperature,
        seed=seed,
        extra_body=_thinking_extra_body(enable_thinking),
    )
    messages.append({"role": "assistant", "content": new_assistant_message})
    return messages, new_assistant_message, tokens_usage


# ----------------------------------------------------------------------------
# Compact interaction mode
# ----------------------------------------------------------------------------
# Instead of a growing back-and-forth transcript (resent in full every turn), the
# compact mode sends only two messages per turn: the system prompt and a single
# user message holding the loop nest, the frozen program analysis, and a summary
# of everything explored so far. This keeps the per-turn prompt small and roughly
# constant in structure, minimizing token cost over a long exploration. The
# trailing instruction is supplied by the caller (see prompts.py).


def build_exploration_log(explored_scheds, initial_exec_time, top_k=5):
    """Render the exploration summary for compact mode: an optional 'top schedules'
    section (only schedules with speedup > 1) followed by the full chronological log
    of every distinct schedule tried, numbered, with its factual outcome. Duplicate
    schedule strings are collapsed to their first occurrence."""
    seen = set()
    unique = []
    for entry in explored_scheds:
        sched_str = entry[0]
        if sched_str in seen:
            continue
        seen.add(sched_str)
        reason = entry[4] if len(entry) > 4 else _derive_log_reason(entry[1], entry[3], initial_exec_time)
        unique.append((sched_str, entry[1], entry[3], reason))

    if not unique:
        return "Exploration log: no schedules tried yet."

    lines = []
    ran = [(s, data) for (s, otype, data, _) in unique if otype == 'ran' and initial_exec_time / data > 1]
    ran.sort(key=lambda x: x[1])  # fastest first
    if ran:
        lines.append("Top schedules so far (speedup > 1):")
        for i, (s, data) in enumerate(ran[:top_k], 1):
            lines.append(f"  {i}. {s} - {round(initial_exec_time / data, 2)}x")
        lines.append("")

    lines.append(f"Exploration log ({len(unique)} distinct schedules tried, chronological):")
    for i, (s, _otype, _data, reason) in enumerate(unique, 1):
        lines.append(f"  #{i} {s} -> {reason}")
    return "\n".join(lines)


def build_compact_message(loop_nest_str, initial_exec_time, analysis, explored_scheds, instruction, top_k=5):
    """Assemble the single user message sent each turn in compact mode: the loop
    nest, initial time, frozen analysis, exploration log, and a trailing instruction."""
    exploration_log = build_exploration_log(explored_scheds, initial_exec_time, top_k=top_k)
    return (
        f"Here is the loop structure\n\n{loop_nest_str}\n\n"
        f"The initial execution time is {initial_exec_time}ms.\n\n"
        f"Program analysis:\n{analysis}\n\n"
        f"{exploration_log}\n\n"
        f"{instruction}"
    )


def get_compact_LLM_response(system_instructions, compact_user_message, top_p=1, temperature=1, enable_thinking=True):
    """Stateless single-turn call for compact mode: sends exactly [system, user]
    (no accumulated history) and returns (last_line, full_message, tokens_usage)."""
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": compact_user_message},
    ]
    new_assistant_message, tokens_usage = _chat_completion(
        messages,
        top_p=top_p,
        temperature=temperature,
        extra_body=_thinking_extra_body(enable_thinking),
    )
    last_line = new_assistant_message.splitlines()[-1] if new_assistant_message else ''
    return last_line, new_assistant_message, tokens_usage


# --- Console output ---
def print_llm_message(id, message, as_markdown=False):
    if as_markdown:
        display_markdown(f'{id}⬅️ Assistant: '+ message + '\n', raw=True)    
    else:
        print(f'\033[1m{id}⬅️ Assistant:\033[0m '+ message + '\n')  
        
def print_compiler_message(id, message, as_markdown=False):
    if as_markdown:
        display_markdown(f'{id}➡️ Compiler: '+ message + '\n', raw=True)    
    else:
        print(f'\033[1m{id}➡️ Compiler:\033[0m '+ message + '\n')  

# --- Schedule-validity predicates (checked while a Schedule is constructed) ---
def is_new_optim_valid(prev_schedule, new_optim, llm_transfs):
    if new_optim.is_unrolling():
       
        if not unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
            return False
        elif has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
            return False
        
    elif new_optim.is_skewing():
        if not are_iterators_consecutive(new_optim.iterators):
            raise InvalidScheduleError(f'Skewing needs two adjacent loop levels (e.g. L1, L2), not "{llm_transfs[-1]}".')
        if not are_iterators_perfectly_nested(prev_schedule.tree, new_optim.iterators):
            raise InvalidScheduleError(f'Skewing requires perfectly nested loops (command "{llm_transfs[-1]}").')
        # check if solution found?

    elif new_optim.is_interchange():
        iterators = sorted(new_optim.params, key=lambda x:x[1])
        if not are_iterators_perfectly_nested(prev_schedule.tree, iterators):
            raise InvalidScheduleError(f'Interchange requires perfectly nested loops (command "{llm_transfs[-1]}").')

        if has_interchange_on_parallelized_loops(prev_schedule, new_optim, llm_transfs):
            return False

    elif new_optim.is_any_tiling():
        if not are_iterators_consecutive(new_optim.iterators):
            raise InvalidScheduleError(f'Tiling needs adjacent loop levels (e.g. L1, L2), not "{llm_transfs[-1]}".')

        if has_tiling_above_parallelized_loops(prev_schedule, new_optim, llm_transfs):
            return False

    # checks that need do be done no matter what's the new optim
    if not unrollings_applied_last_in_branch(prev_schedule, new_optim, llm_transfs):
        return False
    return True


def unrollings_applied_last_in_branch(prev_schedule, new_optim, llm_transfs):
    new_optim_comps = new_optim.comps
    
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_unrolling():
            unrolled_comps = optim.comps
            if any(comp in unrolled_comps for comp in new_optim_comps):
                raise InvalidScheduleError(f'{llm_transfs[-1]} cannot follow {llm_transfs[i]}: unrolling must be the last transformation on a computation — put all Unroll(...) at the end.')
    return True

def unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()

    new_optim_comps = new_optim.comps
    
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_unrolling():
            unrolled_comps = optim.comps
            if any(comp in unrolled_comps for comp in new_optim_comps):
                raise InvalidScheduleError(f'Cannot unroll again: {llm_transfs[-1]} targets computations already unrolled by {llm_transfs[i]}.')
    return True

def has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()
    if new_optim.unrolling_factor==1:
        raise InvalidScheduleError(f'{llm_transfs[-1]}: cannot unroll a loop with a factor of 1.')
    return False



def are_iterators_perfectly_nested(schedule_tree, iter_id_list_inp):
    # make sure that the ids are canonical (i.e. use the names that figure in tree.iterators)
    iter_id_list = []
    for iter_id in iter_id_list_inp:
        if iter_id in schedule_tree.iterators:
            iter_id_list.append(iter_id)
        else:
            iter_id_list.append(schedule_tree.get_iterator_of_computation(*iter_id).id)
        
        # assert iter_id in schedule_tree.iterators, f"{iter_id} not found in {schedule_tree.iterators}"

    for i in range(len(iter_id_list)-1):
        node = schedule_tree.iterators[iter_id_list[i]]
        if not (len(node.child_iterators)==1 and len(node.computations_list)==0 and iter_id_list[i+1] in node.child_iterators):
            return False
    return True

def are_iterators_consecutive(iter_id_list):
    if any((iter_id_list[i+1][1]-iter_id_list[i][1])!=1 for i in range(len(iter_id_list)-1)):
        return False
    return True
            


def has_interchange_on_parallelized_loops(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_interchange()
    interchanged_levels = [new_optim.params[0][1], new_optim.params[1][1]]
    interchange_comps = new_optim.comps
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_parallelization():
            parallel_level = optim.iterator_id[1]
            parallel_comps = optim.comps
            # if it affects parallelized comps and one of the interchanged levels is parallelized, this can cause a wrong legality evaluation
            if any(comp in interchange_comps for comp in parallel_comps) and (interchanged_levels[0]==parallel_level or interchanged_levels[1]==parallel_level):
                raise InvalidScheduleError(f'Interchange {llm_transfs[-1]} cannot follow parallelization of the same loop ({llm_transfs[i]}); reorder them.')
    return False

def has_tiling_above_parallelized_loops(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_any_tiling()
    first_tiled_level = new_optim.iterators[0][1]
    tiled_comps = new_optim.comps
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_parallelization():
            parallel_level = optim.iterator_id[1]
            parallel_comps = optim.comps
            # if it affects parallelized comps and and the tiling level is higher than the parallel one, this can cause a wrong legality evaluation since new loops will be created while the parallel tag stays in the same level
            if any(comp in tiled_comps for comp in parallel_comps) and (first_tiled_level<parallel_level):
                raise InvalidScheduleError(f'Tiling {llm_transfs[-1]} cannot target levels above a parallelized loop ({llm_transfs[i]}); reorder them.')
    return False

def validate_optim_and_add_to_schedule(prev_schedule, new_optim, llm_transfs):
    new_optim.initialize_action_for_tree(prev_schedule.tree)
    assert is_new_optim_valid(prev_schedule, new_optim, llm_transfs)
    prev_schedule.add_optimizations([new_optim])
    return prev_schedule

def extract_transformation_commands(message):
    if not message:
        return None
    match = re.search(r"<schedule>(.*?)</schedule>", message.replace('\n',''))

    if match:
        llm_sched_str = match.group(1)
        return llm_sched_str.replace(' ', '').replace('`','')
    else:
        return None
