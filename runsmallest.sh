current_dir=$(pwd)

# Define the log file path using the current directory
log_file="$current_dir/execution_times.txt"


start_time=$(date +%s)
python -W ignore run.py configs/Own/office3Bm.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py cconfigs/Own/office3Sm.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file
