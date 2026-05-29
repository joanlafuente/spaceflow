import copy
import os.path as osp
from PIL import Image
import numpy as np
import torch
import utils3d
import logging
import open3d_pycg as o3d

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from lib.util import generation, partfield

# Global logger
log = logging.getLogger(__name__)


def _voxelize_mesh(mesh):
    """Voxelize an Open3D mesh into a (1,1,64,64,64) binary float32 tensor on CPU."""
    mesh = copy.deepcopy(mesh)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    voxel_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    coords_dense = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32)
    if len(voxel_indices) > 0:
        coords_dense[0, 0, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    return coords_dense


def compute_coords_dense_indices(struct_coords, individual_sq_meshes, device='cuda', vox_cluster_labels=None):
    """Map each 64^3 voxel to its nearest superquadric (1-indexed; 0=unassigned).

    When vox_cluster_labels is provided (PartField cluster label per voxel), assigns
    voxels via cluster centroids: each cluster centroid is matched to the nearest SQ,
    and all voxels in that cluster inherit the SQ assignment. This is more accurate
    than per-voxel geometric nearest neighbor because PartField clusters capture
    semantic part structure.

    struct_coords: (M, 4) int tensor [batch, x, y, z] in 64x64x64 space.
    vox_cluster_labels: (M,) int tensor of PartField cluster indices per voxel.
    Returns a (1, 1, 32, 32, 32) int32 tensor.
    """
    coords_dense_indices = torch.zeros(1, 1, 32, 32, 32, dtype=torch.int32, device=device)
    n_sq = len(individual_sq_meshes)
    if n_sq == 0:
        return coords_dense_indices

    generated_coords = struct_coords[:, 1:].float().to(device)  # (M, 3)

    # Precompute SQ surface voxel coords once
    sq_coords_list = []
    for mesh in individual_sq_meshes:
        sq_vox = _voxelize_mesh(mesh)  # (1,1,64,64,64)
        sq_c = torch.argwhere(sq_vox)[:, 2:].float().to(device)  # (K,3)
        sq_coords_list.append(sq_c)

    if vox_cluster_labels is not None:
        # PartField-based assignment: cluster centroid → nearest SQ → voxels
        labels = vox_cluster_labels.to(device)
        num_clusters = int(labels.max().item()) + 1
        cluster_to_sq = torch.zeros(num_clusters, dtype=torch.long, device=device)
        for c in range(num_clusters):
            mask = labels == c
            if not mask.any():
                continue
            cluster_voxels = generated_coords[mask]
            mean_pos = cluster_voxels.mean(0, keepdim=True)
            centroid = cluster_voxels[torch.cdist(cluster_voxels, mean_pos).argmin()].unsqueeze(0)  # (1, 3) actual voxel
            min_dists = [
                torch.cdist(sq_c, centroid).min().item() if sq_c.shape[0] > 0 else float('inf')
                for sq_c in sq_coords_list
            ]
            cluster_to_sq[c] = int(np.argmin(min_dists))
        min_distances_idx = cluster_to_sq[labels] + 1  # (M,), 1-indexed
    else:
        # Fallback: per-voxel geometric nearest neighbor
        min_distances = torch.zeros(n_sq, struct_coords.shape[0], device=device)
        for idx, sq_c in enumerate(sq_coords_list):
            if sq_c.shape[0] == 0:
                min_distances[idx] = float('inf')
            else:
                min_distances[idx] = torch.cdist(sq_c, generated_coords).min(0).values
        min_distances_idx = min_distances.argmin(0) + 1  # 1-indexed, shape (M,)

    for i in range(struct_coords.shape[0]):
        x = struct_coords[i, 1] // 2
        y = struct_coords[i, 2] // 2
        z = struct_coords[i, 3] // 2
        coords_dense_indices[0, 0, x, y, z] = int(min_distances_idx[i].item())

    return coords_dense_indices

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

def chunked_contrastive_loss(feats, labels, eps=1e-08, chunk_size=1024):
    """Supervised contrastive loss without materializing the full N×N similarity matrix.
    Peak memory is O(chunk_size × N) instead of O(N²).
    """
    # feats: (1, 1, N, C) -> (N, C)
    x = feats[0, 0]
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=eps)  # (N, C), L2-normalized
    labels = labels.view(-1)  # (N,)
    N = x_norm.shape[0]

    numerator = torch.zeros(N, device=x.device)
    denominator = torch.zeros(N, device=x.device)
    valid = torch.zeros(N, dtype=torch.bool, device=x.device)

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        sim_row = x_norm[i:i_end] @ x_norm.T  # (cs, N)

        labels_i = labels[i:i_end].unsqueeze(1)  # (cs, 1)
        same_label = (labels_i == labels.unsqueeze(0)).float()  # (cs, N)

        # Exclude self-similarity on diagonal
        diag_idx = torch.arange(i_end - i, device=x.device)
        same_label[diag_idx, i + diag_idx] = 0.0

        logits_mask = torch.ones_like(sim_row)
        logits_mask[diag_idx, i + diag_idx] = 0.0

        exp_sim = torch.exp(sim_row) * logits_mask
        numerator[i:i_end] = (exp_sim * same_label).sum(dim=1)
        denominator[i:i_end] = exp_sim.sum(dim=1)
        valid[i:i_end] = same_label.sum(dim=1) > 0

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss[valid].mean()

def optimize_self_similarity(cfg, app, app_type, output_dir,
                             local_prompts=None, local_prompt_type=None,
                             individual_sq_meshes=None):
    log.info("Starting self-similarity optimization...")

    if app_type == 'image':
        generation_pipeline = TrellisImageTo3DPipeline.from_pretrained(cfg.trellis_img_model_name)
        app = Image.open(osp.join(output_dir, 'app_image.png')).convert('RGB')
        app = generation_pipeline.preprocess_image(app)
    else:
        generation_pipeline = TrellisTextTo3DPipeline.from_pretrained(cfg.trellis_text_model_name)
    generation_pipeline.cuda()
    
    # Load Structure Data
    struct_coords = utils3d.io.read_ply(osp.join(output_dir, 'voxels', 'struct_voxels.ply'))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    
    zeros = torch.zeros((struct_coords.size(0), 1), dtype=struct_coords.dtype, device=struct_coords.device)
    struct_coords = torch.cat([zeros, struct_coords], dim=1)
    
    # Load partfield planes (extracted from Blender's struct_renders/mesh.ply — same mesh as voxels)
    path = osp.join(output_dir, "partfield", "part_feat_mesh_batch_part_plane.npy")
    struct_part_planes = torch.from_numpy(np.load(path, allow_pickle=True)).cuda()

    struct_labels = partfield.cluster_geoms(struct_coords, struct_part_planes, num_clusters=cfg.sim_guidance.num_part_clusters)

    # Optimization Starts...
    struct_labels = torch.from_numpy(struct_labels.flatten()).cuda()
    struct_feats_params = torch.nn.Parameter(torch.randn((struct_coords.shape[0], cfg.flow_model_in_channels)), requires_grad=True)
    
    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.sim_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    
    best_loss = float('inf')
    feats = None
    
    cond = generation_pipeline.get_cond_text([app]) if app_type != "image" else generation_pipeline.get_cond([app])

    # Build per-SQ local conditioning if local prompts were provided.
    cond_list = None
    coords_dense_indices = None
    if local_prompts and individual_sq_meshes:
        global_emb = cond['cond']  # (1, seq_len, dim)
        if local_prompt_type == 'text':
            local_embs = [
                generation_pipeline.encode_text([p]) if p and p.strip() else global_emb
                for p in local_prompts
            ]
        else:  # 'image'
            local_embs = [
                generation_pipeline.encode_image([Image.open(p).convert('RGB')]) if p else global_emb
                for p in local_prompts
            ]
        cond_list = [global_emb] + local_embs
        log.info(f"Built cond_list with {len(cond_list)} entries (1 global + {len(local_embs)} local)")
        coords_dense_indices = compute_coords_dense_indices(struct_coords, individual_sq_meshes, 'cuda',
                                                             vox_cluster_labels=struct_labels)
        log.info(f"coords_dense_indices: shape={coords_dense_indices.shape}, non-zero={int((coords_dense_indices > 0).sum())}")

        log.info("Skipping condition routing visualization; routing indices are still used for local conditioning.")
        torch.cuda.empty_cache()

    flow_model = generation_pipeline.models['slat_flow_model']

    # get_cond() is done — only slat_flow_model is needed for the loop. Offload everything else.
    for k, m in generation_pipeline.models.items():
        if k != 'slat_flow_model':
            m.cpu()
    torch.cuda.empty_cache()
    
    if app_type == 'image':
        sampler_params = {
            "cfg_strength": cfg.img_model.cfg_strength,
            "cfg_interval": cfg.img_model.cfg_interval,
        }
        rescale_t = cfg.img_model.rescale_t
    
    else:
        sampler_params = {
            "cfg_strength": cfg.text_model.cfg_strength,
            "cfg_interval": cfg.text_model.cfg_interval,
        }
        rescale_t = cfg.text_model.rescale_t

    t_seq = np.linspace(1, 0, cfg.sim_guidance.steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(cfg.sim_guidance.steps))
    
    std = torch.tensor(generation_pipeline.slat_normalization['std'])[None].cuda()
    mean = torch.tensor(generation_pipeline.slat_normalization['mean'])[None].cuda()

    log.info(f"Beginning self-similarity guidance + flow sampling loop for {len(t_pairs)} steps...")
    for iteration, (t, t_prev) in enumerate(t_pairs):
        optimizer.zero_grad()
        
        # Diffusion
        struct_feats_params_clone = struct_feats_params.clone().cuda()
        noise = sp.SparseTensor(
            feats = struct_feats_params_clone,
            coords = struct_coords.int(),
        ).cuda()
        
        local_kwargs = {}
        if cond_list is not None:
            local_kwargs['cond_list'] = cond_list
            local_kwargs['coords_dense_indices'] = coords_dense_indices

        with torch.no_grad():
            out = generation_pipeline.slat_sampler.sample_once(flow_model, noise, t, t_prev, **cond, **sampler_params, **local_kwargs)
            
        sample = out.pred_x_prev
        struct_feats_params.data = sample.feats
        
        # Optimization - Structure Loss
        if iteration < len(t_pairs) - 1:
            struct_loss = chunked_contrastive_loss(struct_feats_params[None, None, ...], struct_labels)

            total_loss = cfg.sim_guidance.loss_weight * struct_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iteration == 0) or (iteration + 1) % cfg.log_every == 0:
                message = f"Step: {iteration}, Structure Loss: {struct_loss.item():.4f}, Total Loss: {total_loss.item():.4f}"
                log.info(message)
                
            if total_loss < best_loss:
                best_loss = total_loss.item()
                feats = struct_feats_params.detach() * std + mean
    
    # Move SLAT decoders back to GPU for decoding
    for k in ['slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf']:
        if k in generation_pipeline.models:
            generation_pipeline.models[k].cuda()

    # Decode SLAT
    log.info("Decoding output SLAT...")
    out_meshpath = osp.join(output_dir,  'out_sim.glb')
    out_gspath = osp.join(output_dir,  'out_gaussian_sim.mp4')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, out_gspath)