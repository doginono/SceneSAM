import kaolin as kal
import torch
import trimesh
from tqdm import tqdm

def load_point_cloud(file_name):
    """ Load a mesh and sample points on its surface. """
    mesh = trimesh.load(file_name)  # Sample 10,000 points on the surface
    #return mesh.vertices
    return torch.tensor(mesh.vertices).to('cuda')

def chamfer_distance(p1, p2):
    x = p1.unsqueeze(0)
    y = p2.unsqueeze(0)
    d1 = kal.metrics.pointcloud.chamfer_distance(x, y)
    d2 = kal.metrics.pointcloud.chamfer_distance(y, x)
    chamfer_dist = torch.mean(d1) + torch.mean(d2)
    return chamfer_dist

def main():
    log_file = '/home/rozenberszki/project/wsnsl/chamfer_dist.txt'
    with open(log_file, 'a') as f:
        f.write(f'{"="* 100}Chamfer distance between ground truth and rendered point clouds on\n           Replica dataset\n\n')
    scenes = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_1', 'room_2']
    basepath_gt  = '/home/rozenberszki/project/replica_v2/{scene}/habitat/mesh_semantic.ply'
    basepath_rendered = '/home/rozenberszki/project/wsnsl/output_replica_best_23_April/{scene}_final_mesh_color.ply'
    for scene in tqdm(scenes):
        gt = load_point_cloud(basepath_gt.format(scene=scene))
        rendered = load_point_cloud(basepath_rendered.format(scene=scene))
        chamfer_dist = chamfer_distance(gt, rendered)
        print(f'{scene}: {chamfer_dist.item()}')
        with open(log_file, 'a') as f:
            f.write(f'{scene}: {chamfer_dist.item()}\n')    
    with open(log_file, 'a') as f:
        f.write(f'{"="* 100}')
    

if __name__ == '__main__':
    main()