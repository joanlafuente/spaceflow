from html import parser
import os.path as osp
import gc
import trimesh
from PIL import Image
import logging as log
from omegaconf import OmegaConf
import argparse
import random
import numpy as np
from skimage import measure

import torch
from torchvision import transforms
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from pycg import vis, image
from pycg import render as pycg_render
import open3d_pycg as o3d

import sys
sys.path.append('.')

from third_party.PartField.partfield.model_trainer_pvcnn_only_demo import Model
from lib.opt import appearance, self_similarity
from lib.util import generation, common, render, pointcloud
import third_party.TRELLIS.trellis.models as models
from third_party.TRELLIS.trellis.pipelines import TrellisTextTo3DPipeline
from third_party.TRELLIS.trellis.utils import postprocessing_utils
from utils import merge_meshes

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

STEPS_SHAPE_GEN = 12
CFG_SHAPE_GEN = 7.5

def init_args():
    parser = argparse.ArgumentParser(description='GuideFlow3D - 3D Shape Generation')

    # Guidance mode selection
    parser.add_argument('--guidance_mode', type=str, required=True, choices=['appearance', 'similarity'],
                        help='Guidance mode: "appearance" or "similarity"')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--convert_yup_to_zup', action='store_true',
                        help='Convert Y-up coordinate system to Z-up')

    parser.add_argument('--appearance_mesh', type=str,
                        help='Path to appearance mesh (.glb format)')

    parser.add_argument('--appearance_image', type=str,
                        help='Path to appearance reference image')
    parser.add_argument('--appearance_text', type=str, default='',
                        help='Optional appearance text description')

    # SapceControl parameters
    parser.add_argument('--shape_superquadric_path', type=str, required=True,
                        help='Path to shape superquadrics file')
    parser.add_argument('--shape_tau', type=float, default=6.0, required=True,
                        help='Value of tau for superquadric control')
    parser.add_argument('--text_prompt', type=str, required=True,
                        help='Text prompt for 3D shape generation')

    args = parser.parse_args()

    if args.guidance_mode == 'appearance' and not args.appearance_mesh:
            parser.error("--appearance_mesh is required when using appearance guidance mode")

    elif args.guidance_mode == 'similarity':
        if args.appearance_text and args.appearance_image:
            parser.error("Provide either --appearance_image or --appearance_text for similarity guidance, not both.")

        if not args.appearance_text and not args.appearance_image:
            parser.error("Provide either --appearance_image or --appearance_text for similarity guidance.")

    return parser.parse_args()

def add_superquadric_compact_rot_mat(
    scalings: np.array=np.array([1.0, 1.0, 1.0]),
    exponents: np.array=np.array([2.0, 2.0, 2.0]),
    translation: np.array=np.array([0.0, 0.0, 0.0]),
    rotation: np.array=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0,1.0]]),
    resolution: int=10,
    visible: bool=True):
    """Adds a superqiadroc mesh to the scene."""

    def create_superquadric_mesh(A, B, C, e1, e2, N):
        def f(o, m):
            return np.sign(np.sin(o)) * np.abs(np.sin(o))**m
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o))**m
        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = (np.repeat(v, N))
        if np.linalg.det(rotation) < 0:
            u = u[::-1]
        triangles = []

        x = A * g(v, e1) * g(u, e2)
        y = B * g(v, e1) * f(u, e2)
        z = C * f(v, e1)
        # Set poles to zero to account for numerical instabilities in f and g due to ** operator
        x[:N] = 0.0
        x[-N:] = 0.0
        vertices =  np.concatenate([np.expand_dims(x, 1),
                                    np.expand_dims(y, 1),
                                    np.expand_dims(z, 1)], axis=1)
        vertices =  (rotation @ vertices.T).T +translation  # TODO verify left or right apply rotation

        triangles = []
        for i in range(N-1):
            for j in range(N-1):
                triangles.append([i*N+j, i*N+j+1, (i+1)*N+j])
                triangles.append([(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)])
        # Connect first and last vertex in each row
        for i in range(N - 1):
            triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
            triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])

        triangles.append([(N-1)*N+(N-1), (N-1)*N, (N-1)])
        triangles.append([(N-1), (N-1)*N, 0])

        return vertices, triangles


    vertices, triangles = create_superquadric_mesh(scalings[0], scalings[1], scalings[2],
                                                exponents[0], exponents[1],
                                                resolution)
    return vertices, triangles

def load_superquadric_from_file(file_path: str) -> list:
    par_dict = np.load(file_path)
    scale = par_dict['scales']        # 3 (3x1 vector)
    rotate = par_dict['rotations']    # 3 (3x3 rotation matrix)
    shapes = par_dict['shapes']       # 2 (2x1 vector)
    trans = par_dict['translations']  # 3 (3x1 vector)
    num_el = scale.shape[0]           # number of superquadrics

    superquadrics = {}
    for k in range(num_el):
        superquadric_dict = {}
        superquadric_dict['scale'] = scale[k, :]
        superquadric_dict['shape'] = shapes[k]
        superquadric_dict['rotation'] = rotate[k, :]
        superquadric_dict['translation'] = trans[k, :]
        superquadric_dict['color'] = [90, 200, 255]
        superquadrics[k] = superquadric_dict
    return superquadrics

def load_superquadrics(path, args):
    # Implementation for loading superquadrics

    superquadrics = load_superquadric_from_file(path)
    
        # Loading the spatial control mesh generated from the superquadrics provided and check that it all right
    if not osp.exists(args.spatial_control_mesh_path):
        log.error(f"Spatial control mesh not found: {args.spatial_control_mesh_path}")
        return
    else:
        log.info(f"Spatial control mesh found: {args.spatial_control_mesh_path}")
        mesh = o3d.io.read_triangle_mesh(args.spatial_control_mesh_path)
        if mesh.is_empty():
            log.error(f"Spatial control mesh is empty: {args.spatial_control_mesh_path}")
            return
        else:
            log.info(f"Spatial control mesh loaded successfully: {args.spatial_control_mesh_path}")


    meshes = []
    for superquadric_id in superquadrics.keys():
        vertices, triangles = add_superquadric_compact_rot_mat(
        superquadrics[superquadric_id]['scale'],
        superquadrics[superquadric_id]['shape'],
        superquadrics[superquadric_id]['translation'],
        superquadrics[superquadric_id]['rotation'], resolution=100)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        meshes.append(mesh)
    merged_mesh = merge_meshes(meshes)
    aabb = np.stack([np.asarray(merged_mesh.vertices).min(0), np.asarray(merged_mesh.vertices).max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = 1/((aabb[1] - aabb[0]).max())

    merged_mesh.translate(-center)
    merged_mesh.scale(scale, (0,0,0))
    spatial_control_mesh_path = osp.join(args.output_dir, 'spatial_control_mesh.ply')
    o3d.io.write_triangle_mesh(spatial_control_mesh_path, merged_mesh)

def sparse_voxels_to_glb(sparse_points, grid_size=64, output_filename="output.glb"):
    """
    Converts sparse voxel coordinates to a GLB mesh.

    :param sparse_points: List or array of (x, y, z) coordinates (integers).
    :param grid_size: The size of the voxel bounding box (e.g., 64).
    :param output_filename: The name of the output GLB file.
    """

    print(f"Creating {grid_size}x{grid_size}x{grid_size} grid...")
    # Init grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Populate grid with sparse points
    sparse_points = np.round(sparse_points).astype(int)
    for x, y, z in sparse_points:
        # Check bounds just in case
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            voxel_grid[x, y, z] = True

    # Padding the grid (Needed for marching cubes)
    padded_grid = np.pad(voxel_grid, pad_width=1, mode='constant', constant_values=False)

    print("Running Marching Cubes algorithm...")
    # Marching Cubes (Level set at 0.5 to extract the surface between occupied and empty voxels)
    verts, faces, normals, values = measure.marching_cubes(padded_grid, level=0.5)

    # Shift vertices back by 1 to account for the padding we added
    verts = verts - 1.0

    print("Generating mesh and exporting to GLB...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Smoothing of the mesh
    trimesh.smoothing.filter_taubin(mesh, iterations=50)

    # Export to GLB format
    mesh.export(output_filename)
    print(f"Successfully exported mesh to {output_filename}")

def predict_part(obj_path, output_dir):
    log.info("Extracting PartField feature planes...")
    partfield_config = 'third_party/PartField/config.yaml'
    partfield_cfg = OmegaConf.load(partfield_config)

    seed_everything(partfield_cfg.seed)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Lightning defaults to ./lightning_logs under cwd (repo root); team members often
    # cannot mkdir there. Keep all PL artifacts under this run's output_dir.
    pl_root = osp.join(output_dir, 'pl_partfield')
    common.ensure_dir(pl_root)

    trainer = Trainer(devices=-1,
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
    gc.collect() # Free up memory

def main():
    args = init_args()
    cfg = OmegaConf.load('config/default.yaml')

    common.ensure_dir(args.output_dir)

    # Load structure mesh
    log.info("Creating structure mesh with SpaceControl code...")

    load_superquadrics(args.shape_superquadric_path, args)

    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    text_prompt = args.text_prompt

    # Sparse voxels
    coords = pipeline.gen_structure(text_prompt, seed=1, sparse_structure_sampler_params={
        "steps": STEPS_SHAPE_GEN,
        "cfg_strength": CFG_SHAPE_GEN,
        "t0_idx_value": args.shape_tau,
        "spatial_control_mesh_path": osp.join(args.output_dir, 'spatial_control_mesh.ply')
    })

    # Convert sparse voxels to mesh
    log.info("Converting sparse voxels to mesh...")

    coords_np = coords.detach().cpu().numpy()

    print(coords_np)
    filtered_coords = coords_np[:, 1:]
    print(f"Number of valid voxels: {filtered_coords.shape[0]}")

    sparse_voxels_to_glb(filtered_coords, grid_size=64, output_filename=osp.join(args.output_dir, "sample.glb"))

    # glb.export("sample.glb")

    # log.info("Loading generated mesh...")

    struct_mesh = trimesh.load(osp.join(args.output_dir, "sample.glb"), force='mesh')
    # Generator / marching-cubes output in Y-up; keep a copy for debugging.
    struct_mesh.export(osp.join(args.output_dir, 'struct_mesh.glb'))

    del pipeline
    gc.collect() # Free up memory

    # Canonical mesh for renders, voxels, and PartField must share one frame (Z-up if converting).
    if args.convert_yup_to_zup:
        struct_mesh = pointcloud.convert_mesh_yup_to_zup(struct_mesh)
    struct_mesh.export(osp.join(args.output_dir, 'struct_mesh_zup.glb'))
    struct_mesh_for_pipeline = osp.join(args.output_dir, 'struct_mesh_zup.glb')

    log.info(f"Rendering structure mesh for {cfg.num_views // 10} views...")
    struct_render_dir = osp.join(args.output_dir, 'struct_renders')
    common.ensure_dir(struct_render_dir)
    out_renderviews = render.render_all_views(struct_mesh_for_pipeline, struct_render_dir, num_views=cfg.num_views // 10)

    voxel_dir = osp.join(args.output_dir, 'voxels')
    common.ensure_dir(voxel_dir)
    log.info("Voxelizing structure mesh...")
    pointcloud.voxelize_mesh(osp.join(struct_render_dir, 'mesh.ply'), save_path=osp.join(voxel_dir, 'struct_voxels.ply'))

    log.info("Extracting Structure Mesh PartField feature planes...")
    partfield_dir = osp.join(args.output_dir, 'partfield')
    common.ensure_dir(partfield_dir)
    predict_part(struct_mesh_for_pipeline, partfield_dir)

    if not out_renderviews:
        log.info("Structure rendering failed!")

    if args.guidance_mode == 'appearance':
        log.info("Running appearance-guided optimization...")

        # Load appearance mesh
        log.info("Loading appearance mesh...")

        if not args.appearance_mesh.endswith('.glb'):
            log.error("Meshes must be in .glb format")
            return

        if not osp.exists(args.appearance_mesh):
            log.error(f"Appearance mesh not found: {args.appearance_mesh}")
            return

        app_mesh = trimesh.load(args.appearance_mesh, force='mesh')
        app_mesh.export(osp.join(args.output_dir, 'app_mesh.glb'))

        # Convert Y-up to Z-up if needed
        if args.convert_yup_to_zup:
            app_mesh = pointcloud.convert_mesh_yup_to_zup(app_mesh)
        app_mesh.export(osp.join(args.output_dir, 'app_mesh_zup.glb'))

        # Load appearance image
        log.info("Loading appearance image...")
        if args.appearance_image:
            app_image = Image.open(args.appearance_image).convert('RGB')
            app_image.save(osp.join(args.output_dir, 'app_image.png'))
        else:
            mesh = vis.from_file(osp.join(args.output_dir, 'app_mesh.glb'), load_obj_textures=True)
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
            scene = pycg_render.Scene(up_axis='+Y')
            scene.add_object(mesh)
            scene.quick_camera(w=512, h=512, pitch_angle=30, plane_angle=-45.0, fov=40)
            pycg_render.ThemeDiffuseShadow(None, sun_tilt_right=0.0, sun_tilt_back=0.0, sun_angle=60.0).apply_to(scene)
            rendering = scene.render_blender(quality=512)
            rendering = image.alpha_compositing(rendering, image.solid(rendering.shape[1], rendering.shape[0]))
            image.write(osp.join(args.output_dir, 'app_image.png'), rendering)

        # Render views for DinoV2 feature extraction
        log.info(f"Rendering appearance mesh for {cfg.num_views} views...")
        app_render_dir = osp.join(args.output_dir, 'app_renders')
        common.ensure_dir(app_render_dir)
        out_renderviews = render.render_all_views(osp.join(args.output_dir, 'app_mesh.glb'), app_render_dir, num_views=cfg.num_views)
        if not out_renderviews:
            log.info("Appearance rendering failed!")
            return

        # Voxelise mesh
        log.info("Voxelizing appearance mesh...")
        pointcloud.voxelize_mesh(osp.join(app_render_dir, 'mesh.ply'), save_path=osp.join(voxel_dir, 'app_voxels.ply'))

        # Extract DinoV2 Features
        log.info("Extracting DinoV2 features...")
        dinov2_model = torch.hub.load(cfg.dinov2_repo, cfg.feature_name)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        common.ensure_dir(osp.join(args.output_dir, 'features', cfg.feature_name))
        generation.extract_feature(args.output_dir, dinov2_model, transform)
        torch.cuda.empty_cache()

        del dinov2_model
        gc.collect() # Free up memory

        # Extract SLAT Latent
        log.info("Extracting SLAT latent...")
        encoder = models.from_pretrained(cfg.enc_pretrained).eval().cuda()

        common.ensure_dir(osp.join(args.output_dir, 'latents', cfg.latent_name))
        generation.get_latent(args.output_dir, cfg.feature_name, cfg.latent_name, encoder)

        del encoder
        gc.collect() # Free up memory

        # Extract PartField features for appearance mesh
        log.info("Extracting Appearance Mesh PartField feature planes...")
        predict_part(osp.join(args.output_dir, 'app_mesh_zup.glb'), partfield_dir)

        # Appearance Optimization
        appearance.optimize_appearance(cfg, args.output_dir)

    elif args.guidance_mode == 'similarity':
        log.info("Running similarity-guided optimization...")

        if args.appearance_image:
            app_type = 'image'
            app = args.appearance_image

            app_image = Image.open(args.appearance_image).convert('RGB')
            app_image.save(osp.join(args.output_dir, 'app_image.png'))

        elif args.appearance_text:
            app_type = 'text'
            app = args.appearance_text

        log.info(f"Using {app_type} for self-similarity guidance...")


        # Self-Similarity Optimization
        self_similarity.optimize_self_similarity(cfg, app, app_type, args.output_dir)

    else:
        raise NotImplementedError(f"Guidance mode {args.guidance_mode} not implemented.")

if __name__ == "__main__":
    main()