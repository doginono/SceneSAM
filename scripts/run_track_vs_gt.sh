

#!/bin/bash


# Get the current working directory
current_dir=$(pwd)

# Define the log file path using the current directory
log_file="$current_dir/execution_times_track_vs_gt.txt"

start_time=$(date +%s)
python -W ignore run.py configs/Own/room0_gt_tracking.yaml
end_time=$(date +%s)
echo "Execution time for room 0 gt: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/room0.yaml
end_time=$(date +%s)
echo "Execution time for room0 tracking activated: $(($end_time - $start_time)) seconds" >> $log_file
