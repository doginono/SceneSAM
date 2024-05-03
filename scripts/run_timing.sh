#!/bin/bash


# Get the current working directory
current_dir=$(pwd)

# Define the log file path using the current directory
log_file="$current_dir/execution_times_david_full_slam.txt"

#git clone https://github.com/JuliusKoerner/wsnsl.git
#cd wsnsl
#git checkout full_slam
#conda env create -f environment_droplet.yml -n scenesam
#sleep 2

#source /home/rozenberszki/anaconda3/etc/profile.d/conda.sh

#conda activate scenesam
## Interrupt the run of the bash file here
#bash scripts/download_replica.sh
#bash scripts/download_sam.sh


start_time=$(date +%s)
python -W ignore run.py configs/Own/office0.yaml
end_time=$(date +%s)
echo "Execution time for office0: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/office1.yaml
end_time=$(date +%s)
echo "Execution time for office1: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/office2.yaml
end_time=$(date +%s)
echo "Execution time for office2: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/office3.yaml
end_time=$(date +%s)
echo "Execution time for office3: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/office4.yaml
end_time=$(date +%s)
echo "Execution time for office4: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/room0.yaml
end_time=$(date +%s)
echo "Execution time for room0: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/room1.yaml
end_time=$(date +%s)
echo "Execution time for room1: $(($end_time - $start_time)) seconds" >> $log_file

start_time=$(date +%s)
python -W ignore run.py configs/Own/room2.yaml
end_time=$(date +%s)
echo "Execution time for room2: $(($end_time - $start_time)) seconds" >> $log_file