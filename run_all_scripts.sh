git pull 

# Big GPU experiments
#sbatch --job-name "full_slam_neursam_office0" --output "logs/full_slam_neursam_office0.log" base_job.job "configs/Own/office0.yaml"
#sbatch --job-name "full_slam_neursam_office1" --output "logs/full_slam_neursam_office1.log" base_job.job "configs/Own/office1.yaml"
#sbatch --job-name "full_slam_neursam_office2" --output "logs/full_slam_neursam_office2.log" base_job.job "configs/Own/office2.yaml"
#sbatch --job-name "full_slam_neursam_office3" --output "logs/full_slam_neursam_office3.log" base_job.job "configs/Own/office3.yaml"
#sbatch --job-name "full_slam_neursam_office4" --output "logs/full_slam_neursam_office4.log" base_job.job "configs/Own/office4.yaml"
#sbatch --job-name "full_slam_neursam_room0" --output "logs/full_slam_neursam_room0.log" base_job.job "configs/Own/room0.yaml"
#sbatch --job-name "full_slam_neursam_room1" --output "logs/full_slam_neursam_room1.log" base_job.job "configs/Own/room1.yaml"
#sbatch --job-name "full_slam_neursam_room2" --output "logs/full_slam_neursam_room2.log" base_job.job "configs/Own/room2.yaml"

sbatch --job-name "full_slam_neursam_07f5b601ee" --output "logs/full_slam_neursam_07f5b601ee.log" base_job.job "configs/Scannet++/07f5b601ee.yaml"
sbatch --job-name "full_slam_neursam_7cd2ac43b4" --output "logs/full_slam_neursam_7cd2ac43b4.log" base_job.job "configs/Scannet++/7cd2ac43b4.yaml"
sbatch --job-name "full_slam_neursam_56a0ec536c" --output "logs/full_slam_neursam_56a0ec536c.log" base_job.job "configs/Scannet++/56a0ec536c.yaml"
sbatch --job-name "full_slam_neursam_8d563fc2cc" --output "logs/full_slam_neursam_8d563fc2cc.log" base_job.job "configs/Scannet++/8d563fc2cc.yaml"
sbatch --job-name "full_slam_neursam_39f36da05b" --output "logs/full_slam_neursam_39f36da05b.log" base_job.job "configs/Scannet++/39f36da05b.yaml"
sbatch --job-name "full_slam_neursam_b20a261fdf" --output "logs/full_slam_neursam_b20a261fdf.log" base_job.job "configs/Scannet++/b20a261fdf.yaml"
sbatch --job-name "full_slam_neursam_8b5caf3398" --output "logs/full_slam_neursam_8b5caf3398.log" base_job.job "configs/Scannet++/8b5caf3398.yaml"

#sbatch --job-name "full_slam_neursam_scene0645_02_both_grid_sem_hs_c_dim" --output "logs/full_slam_neursam_scene0645_02_both_grid_sem_hs_c_dim.log" base_job.job "configs/ScanNet/scene0645_02_both_grid_sem_hs_c_dim.yaml"
#sbatch --job-name "full_slam_neursam_scene0645_02_grid_len" --output "logs/full_slam_neursam_scene0645_02_grid_len.log" base_job.job "configs/ScanNet/scene0645_02_grid_len.yaml"
#sbatch --job-name "full_slam_neursam_scene0645_02_hs_c_dim" --output "logs/full_slam_neursam_scene0645_02_hs_c_dim.log" base_job.job "configs/ScanNet/scene0645_02_hs_c_dim.yaml"
#sbatch --job-name "full_slam_neursam_scene0693_00" --output "logs/full_slam_neursam_scene0693_00.log" base_job.job "configs/ScanNet/scene0693_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0645_02" --output "logs/full_slam_neursam_scene0645_02.log" base_job.job "configs/ScanNet/scene0645_02.yaml"
#sbatch --job-name "full_slam_neursam_scene0389_00" --output "logs/full_slam_neursam_scene0389_00.log" base_job.job "configs/ScanNet/scene0389_00.yaml"

sbatch --job-name "full_slam_neursam_07f5b601ee_no_truth" --output "logs/full_slam_neursam_07f5b601ee_no_truth.log" base_job.job "configs/Scannet++NoTruth/07f5b601ee.yaml"
sbatch --job-name "full_slam_neursam_7cd2ac43b4_no_truth" --output "logs/full_slam_neursam_7cd2ac43b4_no_truth.log" base_job.job "configs/Scannet++NoTruth/7cd2ac43b4.yaml"
sbatch --job-name "full_slam_neursam_56a0ec536c_no_truth" --output "logs/full_slam_neursam_56a0ec536c_no_truth.log" base_job.job "configs/Scannet++NoTruth/56a0ec536c.yaml"
sbatch --job-name "full_slam_neursam_8d563fc2cc_no_truth" --output "logs/full_slam_neursam_8d563fc2cc_no_truth.log" base_job.job "configs/Scannet++NoTruth/8d563fc2cc.yaml"
sbatch --job-name "full_slam_neursam_39f36da05b_no_truth" --output "logs/full_slam_neursam_39f36da05b_no_truth.log" base_job.job "configs/Scannet++NoTruth/39f36da05b.yaml"
sbatch --job-name "full_slam_neursam_b20a261fdf_no_truth" --output "logs/full_slam_neursam_b20a261fdf_no_truth.log" base_job.job "configs/Scannet++NoTruth/b20a261fdf.yaml"
sbatch --job-name "full_slam_neursam_8b5caf3398_no_truth" --output "logs/full_slam_neursam_8b5caf3398_no_truth.log" base_job.job "configs/Scannet++NoTruth/8b5caf3398.yaml"

#sbatch --job-name "full_slam_neursam_scene0011_00" --output "logs/full_slam_neursam_scene0011_00.log" base_job.job "configs/ScanNet_orig/scene0011_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0025_00" --output "logs/full_slam_neursam_scene0025_00.log" base_job.job "configs/ScanNet_orig/scene0025_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0046_00" --output "logs/full_slam_neursam_scene0046_00.log" base_job.job "configs/ScanNet_orig/scene0046_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0131_01" --output "logs/full_slam_neursam_scene0131_01.log" base_job.job "configs/ScanNet_orig/scene0131_01.yaml"
#sbatch --job-name "full_slam_neursam_scene0671_00" --output "logs/full_slam_neursam_scene0671_00.log" base_job.job "configs/ScanNet_orig/scene0671_00.yaml"

#sbatch --job-name "full_slam_neursam_scene0011_00_every_fraame" --output "logs/full_slam_neursam_scene0011_00_every_frame.log" base_job.job "configs/ScanNet_orig_every/scene0011_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0025_00_every_fraame" --output "logs/full_slam_neursam_scene0025_00_every_frame.log" base_job.job "configs/ScanNet_orig_every/scene0025_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0046_00_every_fraame" --output "logs/full_slam_neursam_scene0046_00_every_frame.log" base_job.job "configs/ScanNet_orig_every/scene0046_00.yaml"
#sbatch --job-name "full_slam_neursam_scene0131_01_every_fraame" --output "logs/full_slam_neursam_scene0131_01_every_frame.log" base_job.job "configs/ScanNet_orig_every/scene0131_01.yaml"
#sbatch --job-name "full_slam_neursam_scene0671_00_every_fraame" --output "logs/full_slam_neursam_scene0671_00_every_frame.log" base_job.job "configs/ScanNet_orig_every/scene0671_00.yaml"



# Smaller GPU experiments
#sbatch --job-name "full_slam_neursam_office0_Own_post_process" --output "logs/full_slam_neursam_office0_Own_post_process.log" base_job_small_gpu.job "configs/Own/office0.yaml"
#sbatch --job-name "full_slam_neursam_office1_Own_post_process" --output "logs/full_slam_neursam_office1_Own_post_process.log" base_job_small_gpu.job "configs/Own/office1.yaml"
#sbatch --job-name "full_slam_neursam_office2_Own_post_process" --output "logs/full_slam_neursam_office2_Own_post_process.log" base_job_small_gpu.job "configs/Own/office2.yaml"
#sbatch --job-name "full_slam_neursam_office3_Own_post_process" --output "logs/full_slam_neursam_office3_Own_post_process.log" base_job_small_gpu.job "configs/Own/office3.yaml"
#sbatch --job-name "full_slam_neursam_office4_Own_post_process" --output "logs/full_slam_neursam_office4_Own_post_process.log" base_job_small_gpu.job "configs/Own/office4.yaml"
#sbatch --job-name "full_slam_neursam_room0_Own_post_process" --output "logs/full_slam_neursam_room0_Own_post_process.log" base_job_small_gpu.job "configs/Own/room0.yaml"
#sbatch --job-name "full_slam_neursam_room1_Own_post_process" --output "logs/full_slam_neursam_room1_Own_post_process.log" base_job_small_gpu.job "configs/Own/room1.yaml"
#sbatch --job-name "full_slam_neursam_room2_Own_post_process" --output "logs/full_slam_neursam_room2_Own_post_process.log" base_job_small_gpu.job "configs/Own/room2.yaml"








