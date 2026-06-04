import os
import torch
import numpy as np
import imageio
import trimesh

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.utils import render_utils, postprocessing_utils

def _white_geometry_mesh(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name='geometry_white',
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.85,
        )
    )
    mesh.vertex_normals
    mesh.face_normals
    return mesh


def _postprocess_mesh_geometry(mesh, simplify=0.95):
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices,
        faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=True,
        fill_holes_max_hole_size=0.04,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=False,
        verbose=False,
    )
    return vertices, faces


def decode_slat(generation_pipeline, feats, coords, out_meshpath, out_gspath, texture=True):
    # Decode Output SLAT
    slat = sp.SparseTensor(
            feats = feats.float(),
            coords = coords.int(),
        ).cuda()            
    formats = ['mesh', 'gaussian'] if texture else ['mesh']
    with torch.no_grad():
        outputs = generation_pipeline.decode_slat(slat, formats)

    out_geometry_path = f"{os.path.splitext(out_meshpath)[0]}_geometry.glb"

    if texture:
        mesh_textured = postprocessing_utils.to_glb(
                        outputs['gaussian'][0],
                        outputs['mesh'][0],
                        # Optional parameters
                        simplify=0.95,          # Ratio of triangles to remove in the simplification process
                        texture_size=1024,      # Size of the texture used for the GLB
                        verbose=False,          # Print logs
                    )
        mesh_textured.export(out_meshpath)
        mesh_geometry = _white_geometry_mesh(
            mesh_textured.vertices.copy(),
            mesh_textured.faces.copy(),
        )
        mesh_geometry.export(out_geometry_path)

        if out_gspath:
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(out_gspath, video, fps=30)
        return

    vertices, faces = _postprocess_mesh_geometry(outputs['mesh'][0], simplify=0.95)
    mesh_geometry = _white_geometry_mesh(vertices, faces)
    mesh_geometry.export(out_meshpath)
    mesh_geometry.export(out_geometry_path)
