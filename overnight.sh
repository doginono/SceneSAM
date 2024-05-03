#!/bin/bash

# Timeout duration in seconds (3 hours = 10800 seconds)
timeout_duration=20000

# Log file to store execution times
log_file="/home/rozenberszki/D_Project/wsnsl/execution_times.txt"

# Navigate to the project directory
cd /home/rozenberszki/D_Project/wsnsl/

start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py /home/rozenberszki/D_Project/wsnsl/configs/Scannet++/39f36da05b.yaml
end_time=$(date +%s)
echo "Execution time for 5fb5d2dbf2: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100



start_time=$(date +%s)
timeout $timeout_duration python -W ignore run.py /home/rozenberszki/D_Project/wsnsl/configs/Scannet++/07f5b601ee.yaml
end_time=$(date +%s)
echo "Execution time for 5fb5d2dbf2: $(($end_time - $start_time)) seconds" >> $log_file
sleep 100
# Print a message at the end of the script
echo "Script execution completed."
