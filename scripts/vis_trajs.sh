python vis_traj.py configs/ScanNet/scene0693_00.yaml output/scannet/track_scene0693_00/ckpts/v_00691.tar --dataset scannet

python vis_traj.py configs/Own/room0_panoptic.yaml output/Own/room0_panoptic/ckpts/ef2_ruunAuto_00674.tar --dataset room0_panoptic
python vis_traj.py configs/Own/room0_panoptic_gt.yaml output/Own/room0_panoptic/ckpts/ef2_gt_tracking_00674.tar --dataset room0_panoptic
python vis_traj.py configs/ScanNet/scene0423_02_panoptic_gt.yaml output/scannet/gt_track_scene0423_02_panoptic/ckpts/00683.tar --dataset scannet
python vis_traj.py configs/ScanNet/scene0423_02_panoptic.yaml output/scannet/track_scene0423_02_panoptic/ckpts/plot_paper_00683.tar --dataset scannet
python vis_traj.py configs/Own/room1.yaml output/Own/room1/ckpts/v01999.tar --dataset replica
python vis_traj.py configs/Own/room1_gt.yaml output_david/Own_gt/room1/ckpts/v01999.tar --dataset replica 