import logging as log
import os.path as osp

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import numpy as np
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm

from third_party.TRELLIS.trellis.representations.mesh.cube2mesh import MeshExtractResult
from third_party.TRELLIS.trellis.renderers.mesh_renderer import MeshRenderer
from third_party.TRELLIS.trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


def map_voxel_labels_to_vertices(mesh_vertices, voxel_coords, voxel_labels):
    """Assign each mesh vertex the label of its nearest voxel via KD-tree lookup.

    Args:
        mesh_vertices: np.ndarray (V, 3) — mesh vertex positions
        voxel_coords:  np.ndarray (M, 3) — voxel center positions (same coordinate space)
        voxel_labels:  np.ndarray (M,)   — integer label per voxel

    Returns:
        np.ndarray (V,) — integer label per vertex
    """
    tree = KDTree(voxel_coords)
    _, idx = tree.query(mesh_vertices, k=1)
    return voxel_labels[idx.flatten()]


def visualize_and_save(structure, partfield_clusters, output_folder,
                       output_name='cluster_visualization.mp4',
                       num_frames=300, resolution=512):
    """Render a smooth rotating orbit video of the structure colored by cluster labels.

    Uses the TRELLIS MeshRenderer (nvdiffrast, ssaa=4) and the same yaw/pitch
    camera trajectory as the pipeline's Gaussian scene videos so all sides are visible.

    Args:
        structure:          trimesh.Trimesh — normalized structure mesh
                            (vertices in [-0.5, 0.5])
        partfield_clusters: np.ndarray (V,) — per-vertex integer cluster labels,
                            where V matches structure.vertices.shape[0]
        output_folder:      str  — directory in which to save the video
        output_name:        str  — filename for the saved video
                            (default 'cluster_visualization.mp4')
        num_frames:         int  — number of video frames (default 300)
        resolution:         int  — output resolution in pixels (default 512)
    """
    partfield_clusters = np.asarray(partfield_clusters, dtype=int)
    num_clusters = int(partfield_clusters.max()) + 1

    # Evenly-spaced HSV hues → maximally distinct colors
    hues = np.linspace(0, 1, num_clusters, endpoint=False)
    cluster_colors = np.array([mcolors.hsv_to_rgb([h, 0.90, 0.95]) for h in hues],
                               dtype=np.float32)  # (num_clusters, 3)
    vertex_colors = cluster_colors[partfield_clusters]  # (V, 3)

    # Build MeshExtractResult for the TRELLIS MeshRenderer
    verts_t = torch.tensor(structure.vertices, dtype=torch.float32, device='cuda')
    faces_t = torch.tensor(structure.faces, dtype=torch.long, device='cuda')
    colors_t = torch.tensor(vertex_colors, dtype=torch.float32, device='cuda')
    mesh_result = MeshExtractResult(vertices=verts_t, faces=faces_t, vertex_attrs=colors_t)

    # Camera trajectory: identical to render_utils.render_video
    yaws = torch.linspace(0, 2 * torch.pi, num_frames).tolist()
    pitch = (0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * torch.pi, num_frames))).tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)

    renderer = MeshRenderer(rendering_options={
        "resolution": resolution,
        "near": 1,
        "far": 100,
        "ssaa": 4,
    })

    frames = []
    log.info(f"Rendering {num_frames} frames → {output_name} ...")
    with torch.no_grad():
        for extr, intr in tqdm(zip(extrinsics, intrinsics), total=num_frames,
                               desc=f'Rendering {output_name}'):
            res = renderer.render(mesh_result, extr, intr,
                                  return_types=["color", "mask"])
            # composite cluster colors over black background
            frame = (res['color'] * res['mask']).permute(1, 2, 0).clamp(0.0, 1.0)
            frames.append((frame.cpu().numpy() * 255).astype(np.uint8))

    video_path = osp.join(output_folder, output_name)
    imageio.mimsave(video_path, frames, fps=30)
    log.info(f"Saved {video_path}")
