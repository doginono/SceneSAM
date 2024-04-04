#!/bin/bash

# Define your commands in this array
declare -A commands
commands=(["command1"]="python -W ignore run.py configs/Own/room1.yaml" ["command2"]="python -W ignore run.py configs/Own/room2.yaml" ["command3"]="python -W ignore run.py configs/Own/room0.yaml" ["command4"]="python -W ignore run.py configs/Own/office0.yaml" ["command5"]="python -W ignore run.py configs/Own/office1.yaml" ["command6"]="python -W ignore run.py configs/Own/office2.yaml" ["command7"]="python -W ignore run.py configs/Own/office3.yaml" ["command8"]="python -W ignore run.py configs/Own/office4.yaml" ["command9"]="python -W ignore run.py configs/Own/room0_small.yaml" ["command10"]="python -W ignore run.py configs/Own/room1_small.yaml")
#commands=(["command1"]="python -W ignore run.py configs/Own/room1_small.yaml")
declare -A execution_times

for cmd in "${!commands[@]}"; do
    echo "Running command: ${commands[$cmd]}"
    start_time=$(date +%s)
    ${commands[$cmd]}
    end_time=$(date +%s)
    execution_time=$((end_time-start_time))
    execution_times[$cmd]=$execution_time
done

echo "Execution times:"
for cmd in "${!execution_times[@]}"; do
    echo "${commands[$cmd]}: ${execution_times[$cmd]} seconds"
done