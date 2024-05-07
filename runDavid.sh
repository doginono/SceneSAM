current_dir=$(pwd)

# Define the log file path using the current directory


#start_time=$(date +%s)
#python -W ignore run.py configs/Scannet++/07f5b601ee.yaml
#end_time=$(date +%s)
#echo "Execution time for 07f5b601ee WITH GROUNDTRUTH: $(($end_time - $start_time)) seconds" >> #$log_file
#
#start_time=$(date +%s)
#python -W ignore run.py configs/Scannet++NoTruth/07f5b601ee.yaml
#end_time=$(date +%s)
#echo "Execution time for 07f5b601ee WITHOUT GROUNDTRUTH: $(($end_time - $start_time)) seconds" #>> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/07f5b601ee.yaml
end_time=$(date +%s)


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++NoTruth/07f5b601ee.yaml
end_time=$(date +%s)



start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/7cd2ac43b4.yaml
end_time=$(date +%s)


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++NoTruth/7cd2ac43b4.yaml
end_time=$(date +%s)
