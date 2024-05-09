#!/bin/bash


# Get the current working directory
current_dir=$(pwd)

# Define the log file path using the current directory
log_file="$current_dir/execution_times_ablation.txt"

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


#start_time=$(date +%s)
#python -W ignore run.py configs/Own/office0.yaml
#end_time=$(date +%s)
#echo "Execution time for office0: $(($end_time - $start_time)) seconds" >> $log_file
#start_time=$(date +%s)
#python -W ignore run.py configs/Own/office0_gt.yaml
#end_time=$(date +%s)
#echo "Execution time for office0_gt: $(($end_time - $start_time)) seconds" >> $log_file
#sleep 2
#start_time=$(date +%s)
#python -W ignore run.py configs/Own_post_process/office0.yaml
#end_time=$(date +%s)
#echo "Execution time for office0_post_process: $(($end_time - $start_time)) seconds" >> $log_file
#sleep 2
start_time=$(date +%s)
python -W ignore run.py configs/Own/room0.yaml
end_time=$(date +%s)
echo "Execution time for room0: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2
start_time=$(date +%s)
python -W ignore run.py configs/Own/room0_gt.yaml
end_time=$(date +%s)
echo "Execution time for room0_gt: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2
start_time=$(date +%s)
python -W ignore run.py configs/Own_post_process/room0.yaml
end_time=$(date +%s)
echo "Execution time for room0_post_process: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2

start_time=$(date +%s)
python -W ignore run.py configs/Own/office2.yaml
end_time=$(date +%s)
echo "Execution time for office2: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2
start_time=$(date +%s)
python -W ignore run.py configs/Own/office2_gt.yaml
end_time=$(date +%s)
echo "Execution time for office2_gt: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2
start_time=$(date +%s)
python -W ignore run.py configs/Own_post_process/office2.yaml
end_time=$(date +%s)
echo "Execution time for office2 post process: $(($end_time - $start_time)) seconds" >> $log_file
sleep 2



