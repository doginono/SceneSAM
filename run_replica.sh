sbatch --job-name "neursam_office0" --output "logs/neursam_office0.log" base_job.job "configs/Own/office0.yaml"
sbatch --job-name "neursam_office1" --output "logs/neursam_office1.log" base_job.job "configs/Own/office1.yaml"
sbatch --job-name "neursam_office2" --output "logs/neursam_office2.log" base_job.job "configs/Own/office2.yaml"
sbatch --job-name "neursam_office3" --output "logs/neursam_office3.log" base_job.job "configs/Own/office3.yaml"
sbatch --job-name "neursam_office4" --output "logs/neursam_office4.log" base_job.job "configs/Own/office4.yaml"
sbatch --job-name "neursam_room0" --output "logs/neursam_room0.log" base_job.job "configs/Own/room0.yaml"
sbatch --job-name "neursam_room1" --output "logs/neursam_room1.log" base_job.job "configs/Own/room1.yaml"
sbatch --job-name "neursam_room2" --output "logs/neursam_room2.log" base_job.job "configs/Own/room2.yaml"



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