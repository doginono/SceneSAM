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

scp -r rozenberszki@sws01.vc.cit.tum.de:/home/rozenberszki/D_Project/wsnsl/Dataset ./Dataset

bash scripts/download_sam.sh

start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/07f5b601ee.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/7cd2ac43b4.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/configs/Scannet++/56a0ec536c.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/8d563fc2cc.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/39f36da05b.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/b20a261fdf.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file


start_time=$(date +%s)
python -W ignore run.py configs/Scannet++/8b5caf3398.yaml
end_time=$(date +%s)
echo "Execution time for 07f5b601ee: $(($end_time - $start_time)) seconds" >> $log_file
