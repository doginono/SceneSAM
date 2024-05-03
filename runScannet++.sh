# Get the current working directory
current_dir=$(pwd)

# Define the log file path using the current directory
log_file="$current_dir/execution_times.txt"

git clone https://github.com/JuliusKoerner/wsnsl.git
cd wsnsl
git checkout AutomaticNew
conda env create -f environment_droplet.yml -n scenesam
sleep 2

source /home/rozenberszki/anaconda3/etc/profile.d/conda.sh

conda activate scenesam

mkdir Dataset



bash scripts/download_sam.sh

mkdir Dataset

scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/07f5b601ee ./Dataset/07f5b601ee

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/07f5b601ee.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/7cd2ac43b4 ./Dataset/7cd2ac43b4

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/7cd2ac43b4.yaml
end_time=$(date +%s)
echo "Execution time for 7cd2ac43b4: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/56a0ec536c ./Dataset/56a0ec536c

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/configs/Scannet++/56a0ec536c.yaml
end_time=$(date +%s)
echo "Execution time for 56a0ec536c: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/8d563fc2cc ./Dataset/8d563fc2cc

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/8d563fc2cc.yaml
end_time=$(date +%s)
echo "Execution time for 8d563fc2cc: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/39f36da05b ./Dataset/39f36da05b

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/39f36da05b.yaml
end_time=$(date +%s)
echo "Execution time for 39f36da05b: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/b20a261fdf ./Dataset/b20a261fdf

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/b20a261fdf.yaml
end_time=$(date +%s)
echo "Execution time for b20a261fdf: $(($end_time - $start_time)) seconds" >> $log_file


scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset/8b5caf3398 ./Dataset/8b5caf3398

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/8b5caf3398.yaml
end_time=$(date +%s)
echo "Execution time for 8b5caf3398: $(($end_time - $start_time)) seconds" >> $log_file
