import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
import logging

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from lib.util import generation
from lib.superdec import (
    build_correspondence,
    load_superdec_npz,
    save_segment_visualisations,
    save_summary,
    UNMATCHED,
)

# Global logger
log = logging.getLogger(__name__)

def optimize_appearance(cfg, output_dir):
    log.info("Starting appearance optimization...")
    
    generation_pipeline = TrellisImageTo3DPipeline.from_pretrained(cfg.trellis_img_model_name)
    generation_pipeline.cuda()
    
    # load appearance and structure data
    path = osp.join(output_dir, 'latents', cfg.latent_name, "appearance.npz")
    data = np.load(path)
    app_feats = torch.from_numpy(data['feats']).cuda()
    app_coords = torch.from_numpy(data['coords']).cuda()
    
    struct_coords = utils3d.io.read_ply(osp.join(output_dir, 'voxels', 'struct_voxels.ply'))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    
    app_image = Image.open(osp.join(output_dir, 'app_image.png')).convert('RGB')
    
    zeros = torch.zeros((struct_coords.size(0), 1), dtype=struct_coords.dtype, device=struct_coords.device)
    struct_coords = torch.cat([zeros, struct_coords], dim=1)

    # ------------------------------------------------------------------
    # SUPERDEC-driven correspondence (replaces PartField + k-means).
    # outdict_q.npz / outdict_a.npz are produced by run.py via
    # lib.superdec.predict_superdec on the active voxel point clouds.
    #
    # Voxel float positions in [-0.5, 0.5]^3 are reconstructed from the
    # already-loaded integer voxel indices: x_float = (x_int + 0.5)/64 - 0.5.
    # For voxels_a_xyz we deliberately use app_coords (the SLAT-latent
    # row order) so that the matched index m(i) can directly index
    # app_feats below.
    # ------------------------------------------------------------------
    superdec_q = load_superdec_npz(osp.join(output_dir, 'superdec', 'outdict_q.npz'))
    superdec_a = load_superdec_npz(osp.join(output_dir, 'superdec', 'outdict_a.npz'))
    voxels_q_xyz = (struct_coords[:, 1:].float() + 0.5) / 64.0 - 0.5      # (N_q, 3)
    voxels_a_xyz = (app_coords[:, 1:].float() + 0.5) / 64.0 - 0.5         # (N_a, 3)

    sd_cfg = cfg.app_guidance
    correspondence = build_correspondence(
        voxels_q_xyz, voxels_a_xyz, superdec_q, superdec_a,
        exist_threshold=float(sd_cfg.get('superdec_exist_threshold', 0.5)),
        refine_boundary=bool(sd_cfg.get('superdec_boundary_refine', True)),
        boundary_softmax_beta=float(sd_cfg.get('superdec_boundary_softmax_beta', 50.0)),
        sinkhorn_eps=float(sd_cfg.get('superdec_match_eps', 0.05)),
        sinkhorn_iters=int(sd_cfg.get('superdec_match_iters', 30)),
        conf_threshold=float(sd_cfg.get('superdec_conf_threshold', 0.15)),
        inside_threshold=float(sd_cfg.get('superdec_inside_threshold', 0.0)),
        include_translation_descriptor=bool(sd_cfg.get('superdec_include_translation', False)),
    )

    valid_idx = correspondence.valid.nonzero(as_tuple=False).flatten()
    m_valid = correspondence.m[valid_idx]
    n_pq = int(superdec_q['scale'].shape[0])
    n_pa = int(superdec_a['scale'].shape[0])
    n_q = int(struct_coords.shape[0])
    n_unmatched_seg = int((correspondence.tau == UNMATCHED).sum().item())
    log.info(
        "[SUPERDEC] P_q=%d P_a=%d  unmatched_segments=%d  |L_q|=%d/%d (%.1f%%)  mean_conf=%.3f",
        n_pq, n_pa, n_unmatched_seg, int(valid_idx.numel()), n_q,
        100.0 * float(valid_idx.numel()) / max(n_q, 1),
        float(correspondence.conf.mean().item()),
    )
    # Diagnostic dumps for visual QC. These are cheap (write 4 small PLYs +
    # one JSON) and independent of the optimisation loop, so always on.
    save_segment_visualisations(
        voxels_q_xyz, voxels_a_xyz,
        correspondence.s_q, correspondence.s_a,
        correspondence.tau,
        output_dir,
    )
    save_summary(
        output_dir,
        n_pq=n_pq,
        n_pa=n_pa,
        n_voxels_q=n_q,
        n_voxels_a=int(voxels_a_xyz.shape[0]),
        tau=correspondence.tau,
        conf=correspondence.conf,
        valid_mask=correspondence.valid,
        nn_dist=correspondence.nn_dist,
    )
    log.info("[SUPERDEC] diagnostics written to %s/superdec/", output_dir)

    if int(valid_idx.numel()) == 0:
        raise RuntimeError(
            "SUPERDEC matcher produced |L_q|=0; lower app_guidance.superdec_conf_threshold "
            "or inspect superdec/superdec_summary.json + segment_correspondence_*.ply."
        )

    struct_feats_params = torch.nn.Parameter(torch.randn((struct_coords.shape[0], cfg.flow_model_in_channels)), requires_grad=True)

    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.app_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    best_loss = float('inf')
    feats = None

    image = generation_pipeline.preprocess_image(app_image)
    cond = generation_pipeline.get_cond([image])

    flow_model = generation_pipeline.models['slat_flow_model']

    sampler_params = {
            "cfg_strength": cfg.img_model.cfg_strength,
            "cfg_interval": cfg.img_model.cfg_interval,
        }
    rescale_t = cfg.img_model.rescale_t

    t_seq = np.linspace(1, 0, cfg.app_guidance.steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(cfg.app_guidance.steps))

    std = torch.tensor(generation_pipeline.slat_normalization['std'])[None].cuda()
    mean = torch.tensor(generation_pipeline.slat_normalization['mean'])[None].cuda()
    
    log.info(f"Beginning guidance + flow sampling loop for {len(t_pairs)} steps...")
    for iteration, (t, t_prev) in enumerate(t_pairs):
        optimizer.zero_grad()

        # Diffusion
        struct_feats_params_clone = struct_feats_params.clone().cuda()
        noise = sp.SparseTensor(
                    feats = struct_feats_params_clone,
                    coords = struct_coords.int(),
                    ).cuda()
    
        with torch.no_grad():
            out = generation_pipeline.slat_sampler.sample_once(flow_model, noise, t, t_prev, **cond, **sampler_params)

        sample = out.pred_x_prev
        struct_feats_params.data = sample.feats

        # Optimization — SUPERDEC-driven appearance loss:
        #
        #     L_app^SUPERDEC = (1 / |L_q|) Σ_{i ∈ L_q} || z_q(i) - z_a(m(i)) ||²
        #
        # where m(i) was computed once via build_correspondence() above. Voxels
        # whose input primitive is unmatched (tau(s_q[i]) == UNMATCHED) are
        # outside L_q and contribute zero gradient — no global-NN fallback.
        if iteration < len(t_pairs) - 1:
            matched_z_a = app_feats[m_valid]
            app_loss_raw = F.mse_loss(struct_feats_params[valid_idx], matched_z_a)
            app_loss = cfg.app_guidance.loss_weight * app_loss_raw

            total_loss = app_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (iteration == 0) or (iteration + 1) % cfg.log_every == 0:
                message = (
                    f"Step: {iteration}, |L_q|={int(valid_idx.numel())}, "
                    f"Appearance Loss: {app_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}"
                )
                log.info(message)

            if total_loss < best_loss:
                best_loss = total_loss.item()
                feats = struct_feats_params.detach() * std + mean

    # Decode SLAT
    log.info("Decoding output SLAT...")
    out_meshpath = osp.join(output_dir,  'out_app.glb')
    out_gspath = osp.join(output_dir,  'out_gaussian_app.mp4')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, out_gspath)