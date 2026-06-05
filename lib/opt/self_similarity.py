import copy
import os.path as osp
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
import logging
import open3d_pycg as o3d
import time

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from lib.util import generation, partfield

# Global logger
log = logging.getLogger(__name__)


def _select_slat_flow_model(generation_pipeline, app_type):
    """Return the SLAT flow model key/model for old and split text/image pipeline configs."""
    if app_type == 'image':
        candidates = ['slat_flow_model_image', 'slat_flow_model']
    else:
        candidates = ['slat_flow_model_text', 'slat_flow_model']

    for key in candidates:
        if key in generation_pipeline.models:
            log.info(f"Using SLAT flow model: {key}")
            return key, generation_pipeline.models[key]

    available = ', '.join(sorted(generation_pipeline.models.keys()))
    raise KeyError(
        f"No compatible SLAT flow model found for {app_type} conditioning. "
        f"Tried {candidates}; available models: {available}"
    )


def _text_conditioner_to(generation_pipeline, device):
    text_cond_model = getattr(generation_pipeline, 'text_cond_model', None)
    if not text_cond_model:
        return
    if 'model' in text_cond_model:
        text_cond_model['model'].to(device)
    if 'null_cond' in text_cond_model:
        text_cond_model['null_cond'] = text_cond_model['null_cond'].to(device)


def _preprocess_condition_image(generation_pipeline, path):
    image = Image.open(path).convert('RGB')
    if hasattr(generation_pipeline, 'preprocess_image'):
        return generation_pipeline.preprocess_image(image)
    return image


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

def _dense_sq_counts(coords_dense_indices, n_sq):
    flat = coords_dense_indices.reshape(-1)
    return {
        sq_idx: int((flat == sq_idx + 1).sum().item())
        for sq_idx in range(n_sq)
    }

def _dense_condition_counts(coords_dense_indices, n_conditions):
    flat = coords_dense_indices.reshape(-1)
    return {
        cond_idx: int((flat == cond_idx).sum().item())
        for cond_idx in range(1, n_conditions)
    }

def _active_route_coverage(sq_counts, active_indices):
    counts = [sq_counts.get(idx, 0) for idx in active_indices]
    return sum(count > 0 for count in counts), sum(counts)

def _format_nonzero_counts(counts):
    nonzero = {idx: count for idx, count in counts.items() if count > 0}
    return nonzero if nonzero else {}

def _choose_local_routing(struct_coords, individual_sq_meshes, struct_labels, active_indices, device='cuda'):
    clustered = compute_coords_dense_indices(
        struct_coords, individual_sq_meshes, device, vox_cluster_labels=struct_labels)
    n_sq = len(individual_sq_meshes)
    clustered_counts = _dense_sq_counts(clustered, n_sq)
    log.info(f"PartField local routing SQ counts: {_format_nonzero_counts(clustered_counts)}")

    if not active_indices:
        return clustered, "partfield", clustered_counts

    clustered_coverage = _active_route_coverage(clustered_counts, active_indices)
    if clustered_coverage[0] == len(active_indices):
        return clustered, "partfield", clustered_counts

    geometric = compute_coords_dense_indices(
        struct_coords, individual_sq_meshes, device, vox_cluster_labels=None)
    geometric_counts = _dense_sq_counts(geometric, n_sq)
    geometric_coverage = _active_route_coverage(geometric_counts, active_indices)
    log.info(f"Geometric local routing SQ counts: {_format_nonzero_counts(geometric_counts)}")

    if geometric_coverage > clustered_coverage:
        missing = [idx for idx in active_indices if clustered_counts.get(idx, 0) == 0]
        log.warning(
            "PartField local routing missed active SQ(s) %s; using geometric routing "
            "for local texture conditions instead.",
            missing,
        )
        return geometric, "geometric", geometric_counts

    missing = [idx for idx in active_indices if clustered_counts.get(idx, 0) == 0]
    if missing:
        log.warning(
            "PartField local routing missed active SQ(s) %s, and geometric routing "
            "did not improve coverage; keeping PartField routing.",
            missing,
        )
    return clustered, "partfield", clustered_counts

def _dilate_condition_indices(coords_dense_indices, n_conditions, radius=1):
    if radius <= 0 or n_conditions <= 1:
        return coords_dense_indices

    dilated_indices = coords_dense_indices.clone()
    kernel_size = 2 * radius + 1
    for cond_idx in range(1, n_conditions):
        mask = (coords_dense_indices == cond_idx).float()
        dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=radius) > 0
        dilated_indices[(dilated_indices == 0) & dilated] = cond_idx
    return dilated_indices

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

def _auto_contrastive_chunk_size(num_voxels):
    if num_voxels <= 16_000:
        return 1024
    if num_voxels <= 28_000:
        return 512
    return 256


def chunked_contrastive_loss(feats, labels, eps=1e-08, chunk_size=None):
    """Supervised contrastive loss without materializing the full N×N similarity matrix.
    Peak memory is O(chunk_size × N) instead of O(N²).
    """
    # feats: (1, 1, N, C) -> (N, C)
    x = feats[0, 0]
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=eps)  # (N, C), L2-normalized
    labels = labels.view(-1)  # (N,)
    N = x_norm.shape[0]
    chunk_size = chunk_size or _auto_contrastive_chunk_size(N)

    numerator = torch.zeros(N, device=x.device)
    denominator = torch.zeros(N, device=x.device)
    valid = torch.zeros(N, dtype=torch.bool, device=x.device)

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        sim_row = x_norm[i:i_end] @ x_norm.T  # (cs, N)
        exp_sim = torch.exp(sim_row)

        labels_i = labels[i:i_end].unsqueeze(1)  # (cs, 1)
        same_label = labels_i == labels.unsqueeze(0)  # (cs, N)

        # Exclude self-similarity on diagonal
        diag_idx = torch.arange(i_end - i, device=x.device)
        self_mask = torch.zeros_like(same_label)
        self_mask[diag_idx, i + diag_idx] = True
        positive_mask = same_label & ~self_mask

        denominator[i:i_end] = exp_sim.masked_fill(self_mask, 0.0).sum(dim=1)
        valid[i:i_end] = positive_mask.any(dim=1)
        numerator[i:i_end] = exp_sim.masked_fill(~positive_mask, 0.0).sum(dim=1)

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss[valid].mean()

def optimize_self_similarity(cfg, app, app_type, output_dir,
                             local_prompts=None, local_prompt_type=None,
                             individual_sq_meshes=None,
                             generation_pipeline=None,
                             decode_texture=True):
    overall_start = time.perf_counter()
    log.info("Starting self-similarity optimization...")

    pipeline_start = time.perf_counter()
    if generation_pipeline is None:
        if app_type == 'image':
            generation_pipeline = TrellisImageTo3DPipeline.from_pretrained(cfg.trellis_img_model_name)
        else:
            generation_pipeline = TrellisTextTo3DPipeline.from_pretrained(cfg.trellis_text_model_name)
    else:
        log.info("Reusing preloaded TRELLIS pipeline for self-similarity guidance.")
    generation_pipeline.cuda()
    _text_conditioner_to(generation_pipeline, 'cuda')
    log.info("Prepared TRELLIS pipeline for self-similarity in %.2fs", time.perf_counter() - pipeline_start)

    if app_type == 'image':
        app = _preprocess_condition_image(generation_pipeline, osp.join(output_dir, 'app_image.png'))
    
    # Load Structure Data
    structure_start = time.perf_counter()
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
    log.info(
        "Loaded structure voxels, PartField planes, and labels in %.2fs (%d voxels)",
        time.perf_counter() - structure_start,
        struct_coords.shape[0],
    )
    
    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.sim_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    
    best_loss = float('inf')
    feats = None
    
    cond_start = time.perf_counter()
    if app_type == "image":
        cond = generation_pipeline.get_cond([app]) if hasattr(generation_pipeline, 'get_cond') else generation_pipeline.get_cond_image([app])
    else:
        cond = generation_pipeline.get_cond_text([app])
    log.info("Encoded global %s guidance condition in %.2fs", app_type, time.perf_counter() - cond_start)

    # Build per-SQ local conditioning if local prompts were provided.
    cond_list = None
    coords_dense_indices = None
    if local_prompts and individual_sq_meshes:
        local_start = time.perf_counter()
        global_emb = cond['cond']  # (1, seq_len, dim)
        n_sq = len(local_prompts)
        active_indices = [
            idx
            for idx, prompt in enumerate(local_prompts)
            if str(prompt or "").strip()
        ]
        log.info(
            f"Building local {local_prompt_type} conditioning for "
            f"{len(active_indices)} / {n_sq} superquadrics; active SQ indices: {active_indices}"
        )
        if local_prompt_type == 'text':
            local_embs = []
            for sq_idx, prompt in enumerate(local_prompts):
                prompt = str(prompt or "").strip()
                if prompt:
                    log.info(
                        "Local text condition for SQ %d: %r",
                        sq_idx,
                        prompt,
                    )
                local_embs.append(
                    generation_pipeline.encode_text([prompt])
                    if prompt
                    else global_emb
                )
        else:  # 'image'
            local_embs = []
            for path in local_prompts:
                if path:
                    local_image = _preprocess_condition_image(generation_pipeline, path)
                    local_embs.append(generation_pipeline.encode_image([local_image]))
                else:
                    local_embs.append(global_emb)
        # Only include SQs that have a real (non-global-fallback) local condition.
        conditioned_mask = [emb is not global_emb for emb in local_embs]
        conditioned_embs = [emb for emb, is_cond in zip(local_embs, conditioned_mask) if is_cond]
        cond_list = [global_emb] + conditioned_embs

        # Build remap: SQ 1-index → new condition index (0=global fallback, 1..n_conditioned).
        remap = torch.zeros(n_sq + 1, dtype=torch.long, device='cuda')
        new_idx = 1
        for sq_i, is_cond in enumerate(conditioned_mask):
            remap[sq_i + 1] = new_idx if is_cond else 0
            if is_cond:
                new_idx += 1
        sq_cond_map = remap[1:].tolist()  # condition index per SQ (0=global)
        log.info(f"SQ condition mapping (0=global): {sq_cond_map}")
        log.info(f"Built cond_list with {len(cond_list)} entries "
                 f"(1 global + {len(conditioned_embs)} real local out of {n_sq} SQs)")

        coords_dense_indices, routing_source, sq_route_counts = _choose_local_routing(
            struct_coords, individual_sq_meshes, struct_labels, active_indices, 'cuda')
        coords_dense_indices = remap[coords_dense_indices.long()]
        condition_counts_before_dilation = _dense_condition_counts(coords_dense_indices, len(cond_list))
        local_condition_dilation = int(getattr(cfg.sim_guidance, 'local_condition_dilation', 0))
        coords_dense_indices = _dilate_condition_indices(
            coords_dense_indices, len(cond_list), radius=local_condition_dilation)
        condition_counts = _dense_condition_counts(coords_dense_indices, len(cond_list))
        active_sq_counts = {idx: sq_route_counts.get(idx, 0) for idx in active_indices}
        log.info(
            f"Local routing source: {routing_source}; active SQ counts: {active_sq_counts}"
        )
        log.info(
            f"coords_dense_indices: shape={coords_dense_indices.shape}, "
            f"condition_counts_before_dilation={condition_counts_before_dilation}, "
            f"dilation_radius={local_condition_dilation}, "
            f"condition_counts={condition_counts}, non-zero={int((coords_dense_indices > 0).sum())}"
        )

        log.info("Skipping condition routing visualization; routing indices are still used for local conditioning.")
        torch.cuda.empty_cache()
        log.info("Prepared local condition routing in %.2fs", time.perf_counter() - local_start)
    else:
        log.info("No local SQ texture overrides provided; all voxels use the global condition.")

    _text_conditioner_to(generation_pipeline, 'cpu')

    flow_model_key, flow_model = _select_slat_flow_model(generation_pipeline, app_type)

    # get_cond() is done — only the selected SLAT flow model is needed for the loop. Offload everything else.
    for k, m in generation_pipeline.models.items():
        if k != flow_model_key:
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

    loop_start = time.perf_counter()
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
    log.info(
        "Completed self-similarity guidance + flow sampling loop in %.2fs",
        time.perf_counter() - loop_start,
    )
    
    # Move SLAT decoders back to GPU for decoding.
    decoder_load_start = time.perf_counter()
    decoder_keys = ['slat_decoder_mesh']
    if decode_texture:
        decoder_keys.append('slat_decoder_gs')
    for k in decoder_keys:
        if k in generation_pipeline.models:
            generation_pipeline.models[k].cuda()
        elif decode_texture:
            raise KeyError(f"Texture decode requested but required decoder is missing: {k}")
    log.info("Moved SLAT decoder(s) to CUDA in %.2fs: %s", time.perf_counter() - decoder_load_start, decoder_keys)

    # Decode SLAT
    log.info("Decoding output SLAT...")
    decode_start = time.perf_counter()
    out_meshpath = osp.join(output_dir,  'out_sim.glb')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, None, texture=decode_texture)
    log.info("Decoded output SLAT in %.2fs: %s", time.perf_counter() - decode_start, out_meshpath)
    log.info("Completed self-similarity optimization in %.2fs", time.perf_counter() - overall_start)
