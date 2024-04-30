#!/bin/bash

# Timeout duration in seconds (3 hours = 10800 seconds)
timeout_duration=13500

# Log file to store execution times
log_file="/home/rozenberszki/project/wsnsl/execution_times.txt"

cd /home/rozenberszki/project/wsnsl/

# First command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0616_00.yaml
end_time=$(date +%s)
echo "Execution time for scene0616_00: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100

# Second command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0300_01.yaml
end_time=$(date +%s)
echo "Execution time for scene0300_01: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100

# Third command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0427_00.yaml
end_time=$(date +%s)
echo "Execution time for scene0427_00: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100

# Navigate to the project directory
cd /home/rozenberszki/D_Project/wsnsl/

timeout_duration= 20000
# Fourth command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/Scannet++/c413b34238.yaml
end_time=$(date +%s)
echo "Execution time for c413b34238: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100


# Fifth command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/Scannet++/56a0ec536c.yaml
end_time=$(date +%s)
echo "Execution time for 56a0ec536c: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100

# Sixth command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/Scannet++/5fb5d2dbf2.yaml
end_time=$(date +%s)
echo "Execution time for 5fb5d2dbf2: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100

# Print a message at the end of the script
echo "Script execution completed."
