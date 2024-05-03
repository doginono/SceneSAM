#!/bin/bash

# Timeout duration in seconds (3 hours = 10800 seconds)
timeout_duration=16000

# Log file to store execution times
log_file="/home/rozenberszki/project/wsnsl/execution_times.txt"

cd /home/rozenberszki/project/wsnsl/

# First command
#start_time=$(date +%s)
#timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0300_01.yaml
#end_time=$(date +%s)
#echo "Execution time for scene0300_01: $(($end_time - $start_time)) seconds" >> $log_file
#sleep 10

# Second command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0389_00.yaml
end_time=$(date +%s)
echo "Execution time for scene0389_00: $(($end_time - $start_time)) seconds" >> $log_file
sleep 10

# Third command
start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0645_02.yaml
end_time=$(date +%s)
echo "Execution time for scene0645_02: $(($end_time - $start_time)) seconds" >> $log_file
sleep 10

start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0423_02_panoptic.yaml
end_time=$(date +%s)
echo "Execution time for scene0423_02_panoptic: $(($end_time - $start_time)) seconds" >> $log_file
sleep 10

start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py configs/ScanNet/scene0693_00.yaml
end_time=$(date +%s)
echo "Execution time for scene0693_00: $(($end_time - $start_time)) seconds" >> $log_file
sleep 10

# Navigate to the project directory
./home/rozenberszki/D_Project/wsnsl/overnight.sh
