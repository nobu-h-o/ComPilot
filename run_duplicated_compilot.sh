#!/bin/bash

# Usage: ./run_duplicated_compilot.sh <num_runs> <screen_name_prefix> [script_args]

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_runs> <screen_name_prefix> <config_file>"
    exit 1
fi

# Assign arguments to variables
NUM_RUNS=$1            # Number of duplicates to start
SCREEN_PREFIX=$2       # Screen session prefix
CONFIG_FILE=${3:-config.yaml} # Optional config YAML path (default: config.yaml)

PYTHON_SCRIPT="src/compilot.py"    # The Python script to run (run this launcher from the repo root)

# The name of your Conda environment
CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/mm12191/.conda/envs/collm_env
CONDA_CMD=". $CONDA_DIR/bin/activate && conda activate $CONDA_ENV"

# SLURM settings
SBATCH_COMMAND="srun -J ${SCREEN_PREFIX} --exclude=dn105 --pty -n1  --exclusive --tasks-per-node=1 --cpus-per-task=28 --mem=100G -t 7-0:00:00 --constraint='dalma&avx2&compute&dalmac'"

echo "Caution⚠: This script will rerun $PYTHON_SCRIPT indefinitely until it terminates properly. Make sure to not edit $PYTHON_SCRIPT while this script is running to avoid erroneous results."
sleep 2

# Loop to create screen sessions
for i in $(seq 1 $NUM_RUNS)
do
    # Create a unique screen session name
    SCREEN_NAME="${SCREEN_PREFIX}_${i}"
    COMPLETION_FILE="completed_${SCREEN_NAME}.flag"
    rm -f $COMPLETION_FILE  # Ensure old completion file is removed before starting

    # Start a new screen session, activate the Conda environment, and run the Python script
    screen -dmS $SCREEN_NAME bash -c "
        echo 'Starting experiment'
        while true; do
            
            $SBATCH_COMMAND bash -c '$CONDA_CMD && python $PYTHON_SCRIPT -i=$i -c=$COMPLETION_FILE --config=$CONFIG_FILE'
            # Check if the success file was created
            if [ -f \"$COMPLETION_FILE\" ]; then
                echo 'Experiment completed successfully.'
                rm -f \"$COMPLETION_FILE\"  # Cleanup after success
                break
            fi
            echo 'file \"$COMPLETION_FILE\" not found'
            
            echo 'Experiment terminated abnormally. Restarting in 5 seconds...'
            sleep 5
        done
        
        exec bash
    "
    
    echo "Started $SCREEN_NAME running: python $PYTHON_SCRIPT -i=$i --config=$CONFIG_FILE"
done


