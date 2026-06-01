import copy
import gc
import logging as log
import os.path as osp
import random

import numpy as np
import torch
import trimesh
import utils3d
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from skimage import measure
from sklearn.neighbors import KDTree
import open3d_pycg as o3d
import argparse

import sys
sys.path.append('.')

from third_party.PartField.partfield.model_trainer_pvcnn_only_demo import Model
from lib.util import partfield, common, pointcloud
from lib.util.visualization import visualize_and_save, map_voxel_labels_to_vertices as _map_voxel_labels_to_vertices
from third_party.TRELLIS.trellis.pipelines import TrellisTextTo3DPipeline
from utils import merge_meshes

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

STEPS_SHAPE_GEN = 12
CFG_SHAPE_GEN = 7.5


# ---------------------------------------------------------------------------
# Superquadric helpers (copied from run.py)
# ---------------------------------------------------------------------------

def add_superquadric_compact_rot_mat(
        scalings=np.array([1.0, 1.0, 1.0]),
        exponents=np.array([2.0, 2.0, 2.0]),
        translation=np.array([0.0, 0.0, 0.0]),
        rotation=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        resolution=10,
        visible=True):
    def create_superquadric_mesh(A, B, C, e1, e2, N):
        def f(o, m): return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
        def g(o, m): return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m
        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = np.repeat(v, N)
        if np.linalg.det(rotation) < 0:
            u = u[::-1]
        x = A * g(v, e1) * g(u, e2)
        y = B * g(v, e1) * f(u, e2)
        z = C * f(v, e1)
        x[:N] = 0.0
        x[-N:] = 0.0
        vertices = np.stack([x, y, z], axis=1)
        vertices = (rotation @ vertices.T).T + translation
        triangles = []
        for i in range(N - 1):
            for j in range(N - 1):
                triangles.append([i * N + j, i * N + j + 1, (i + 1) * N + j])
                triangles.append([(i + 1) * N + j, i * N + j + 1, (i + 1) * N + (j + 1)])
        for i in range(N - 1):
            triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
            triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])
        triangles.append([(N - 1) * N + (N - 1), (N - 1) * N, (N - 1)])
        triangles.append([(N - 1), (N - 1) * N, 0])
        return vertices, triangles

    return create_superquadric_mesh(scalings[0], scalings[1], scalings[2],
                                    exponents[0], exponents[1], resolution)


def load_superquadric_from_file(file_path):
    par_dict = np.load(file_path)
    scale = par_dict['scales']
    rotate = par_dict['rotations']
    shapes = par_dict['shapes']
    trans = par_dict['translations']
    num_el = scale.shape[0]
    superquadrics = {}
    for k in range(num_el):
        superquadrics[k] = {
            'scale': scale[k, :],
            'shape': shapes[k],
            'rotation': rotate[k, :],
            'translation': trans[k, :],
        }
    return superquadrics


def load_superquadrics(path, spatial_control_mesh_path):
    superquadrics = load_superquadric_from_file(path)
    meshes = []
    for sq_id in superquadrics:
        vertices, triangles = add_superquadric_compact_rot_mat(
            superquadrics[sq_id]['scale'],
            superquadrics[sq_id]['shape'],
            superquadrics[sq_id]['translation'],
            superquadrics[sq_id]['rotation'],
            resolution=100)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        meshes.append(mesh)
    merged = merge_meshes(meshes)
    aabb = np.stack([np.asarray(merged.vertices).min(0), np.asarray(merged.vertices).max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = 1.0 / ((aabb[1] - aabb[0]).max())
    merged.translate(-center)
    merged.scale(scale, (0, 0, 0))
    o3d.io.write_triangle_mesh(spatial_control_mesh_path, merged)
    log.info(f"Spatial control mesh saved to {spatial_control_mesh_path}")


def sparse_voxels_to_glb(sparse_points, grid_size=64, output_filename="output.glb"):
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    sparse_points = np.round(sparse_points).astype(int)
    for x, y, z in sparse_points:
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            voxel_grid[x, y, z] = True
    padded_grid = np.pad(voxel_grid, pad_width=1, mode='constant', constant_values=False)
    verts, faces, normals, values = measure.marching_cubes(padded_grid, level=0.5)
    verts = verts - 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    trimesh.smoothing.filter_taubin(mesh, iterations=50)
    mesh.export(output_filename)


def predict_part(obj_path, output_dir):
    log.info("Extracting PartField feature planes...")
    partfield_config = 'third_party/PartField/config.yaml'
    partfield_cfg = OmegaConf.load(partfield_config)
    seed_everything(partfield_cfg.seed)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    pl_root = osp.join(output_dir, 'pl_partfield')
    common.ensure_dir(pl_root)

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        precision="16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=partfield_cfg.training_epochs,
        log_every_n_steps=1,
        limit_train_batches=3500,
        limit_val_batches=None,
        default_root_dir=pl_root,
        logger=False,
        enable_checkpointing=False,
    )
    partfield_model = Model(partfield_cfg, obj_path)
    output = trainer.predict(partfield_model, ckpt_path=partfield_cfg.continue_ckpt)
    part_planes, uid = output[0]
    np.save(f'{output_dir}/part_feat_{uid}_batch_part_plane.npy', part_planes)
    del partfield_model
    gc.collect()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate structure and visualize PartField clusters')
    parser.add_argument('--shape_superquadric_path', type=str, required=True,
                        help='Path to superquadrics .npz file')
    parser.add_argument('--text_prompt', type=str, required=True,
                        help='Text prompt for structure generation')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of PartField clusters (default: 10)')
    parser.add_argument('--shape_tau', type=float, default=6.0,
                        help='Tau value for spatial control (default: 6.0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for structure generation (default: 1)')
    parser.add_argument('--convert_yup_to_zup', action='store_true',
                        help='Convert Y-up coordinate system to Z-up')
    args = parser.parse_args()

    common.ensure_dir(args.output_dir)

    # 1. Generate spatial control mesh from superquadrics
    spatial_control_mesh_path = osp.join(args.output_dir, 'spatial_control_mesh.ply')
    load_superquadrics(args.shape_superquadric_path, spatial_control_mesh_path)

    # 2. Generate structure (sparse voxels) with Trellis
    log.info("Generating structure with TrellisTextTo3DPipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    coords = pipeline.gen_structure(args.text_prompt, seed=args.seed,
                                    sparse_structure_sampler_params={
                                        "steps": STEPS_SHAPE_GEN,
                                        "cfg_strength": CFG_SHAPE_GEN,
                                        "t0_idx_value": args.shape_tau,
                                        "spatial_control_mesh_path": spatial_control_mesh_path,
                                    })
    del pipeline
    gc.collect()

    # 3. Convert sparse voxels to mesh via marching cubes
    log.info("Converting sparse voxels to mesh...")
    coords_np = coords.detach().cpu().numpy()
    filtered_coords = coords_np[:, 1:]
    log.info(f"Number of valid voxels: {filtered_coords.shape[0]}")

    sample_glb = osp.join(args.output_dir, 'sample.glb')
    sparse_voxels_to_glb(filtered_coords, grid_size=64, output_filename=sample_glb)

    # 4. Normalize mesh to [-0.5, 0.5] (replicates Blender's normalize_scene)
    struct_mesh = trimesh.load(sample_glb, force='mesh')
    verts = struct_mesh.vertices
    bbox_min, bbox_max = verts.min(0), verts.max(0)
    center = (bbox_min + bbox_max) / 2
    scale = 1.0 / (bbox_max - bbox_min).max()
    struct_mesh.vertices = (verts - center) * scale

    # 5. Optionally convert Y-up to Z-up
    if args.convert_yup_to_zup:
        struct_mesh = pointcloud.convert_mesh_yup_to_zup(struct_mesh)

    struct_mesh_zup_path = osp.join(args.output_dir, 'struct_mesh_zup.glb')
    struct_mesh.export(struct_mesh_zup_path)

    # Export to PLY for voxelization (mesh already in [-0.5, 0.5])
    struct_mesh_ply = osp.join(args.output_dir, 'struct_mesh_normalized.ply')
    struct_mesh.export(struct_mesh_ply)

    # 6. Voxelize structure mesh
    voxel_dir = osp.join(args.output_dir, 'voxels')
    common.ensure_dir(voxel_dir)
    struct_voxels_path = osp.join(voxel_dir, 'struct_voxels.ply')
    log.info("Voxelizing structure mesh...")
    pointcloud.voxelize_mesh(struct_mesh_ply, save_path=struct_voxels_path)

    # 7. Extract PartField features
    partfield_dir = osp.join(args.output_dir, 'partfield')
    common.ensure_dir(partfield_dir)
    predict_part(struct_mesh_zup_path, partfield_dir)

    # 8. Load voxel coords and part planes
    struct_voxels_normalized = utils3d.io.read_ply(struct_voxels_path)[0]  # (M, 3) in [-0.5, 0.5]
    struct_coords = torch.from_numpy(struct_voxels_normalized).float().cuda()
    struct_coords_64 = ((struct_coords + 0.5) * 64).long()
    zeros = torch.zeros((struct_coords_64.size(0), 1),
                        dtype=struct_coords_64.dtype, device=struct_coords_64.device)
    struct_coords_4d = torch.cat([zeros, struct_coords_64], dim=1)  # (M, 4)

    part_feat_path = osp.join(partfield_dir,
                              'part_feat_struct_mesh_zup_batch_part_plane.npy')
    struct_part_planes = torch.from_numpy(
        np.load(part_feat_path, allow_pickle=True)).cuda()

    # 9. Cluster voxels via PartField
    log.info(f"Clustering structure into {args.num_clusters} parts with PartField...")
    voxel_labels = partfield.cluster_geoms(struct_coords_4d, struct_part_planes,
                                           num_clusters=args.num_clusters)

    # 10. Map per-voxel labels to per-vertex labels (for smooth mesh rendering)
    struct_mesh_vis = trimesh.load(struct_mesh_zup_path, force='mesh')
    vertex_labels = _map_voxel_labels_to_vertices(
        struct_mesh_vis.vertices, struct_voxels_normalized, voxel_labels)

    # 11. Render colored mesh as orbit video
    visualize_and_save(struct_mesh_vis, vertex_labels, args.output_dir)


if __name__ == '__main__':
    main()
