import numpy as np
from plyfile import (PlyData, PlyElement)
import os
import sys
import subprocess
import time
import threading
import copy
import viser
import viser.transforms as tf
from PIL import Image
from io import BytesIO

os.environ['SPCONV_ALGO'] = 'native'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RESOLUTION = 32
steps = 12

scene_elements = {}
gui_elements = {}
superquadrics = {}
active_superquadric = -1
active_template_id = 1
generated_mesh = None

server = viser.ViserServer(up_axis=2)
server.scene.set_up_direction([0.0, 0.0, 1.0])
server.scene.set_environment_map('studio', background=False, environment_intensity=0.5)
server.scene.add_light_ambient('light_a', color=(255, 255, 255), intensity=10000.0)

server.gui.configure_theme(dark_mode=True)

@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, -0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)


# ---------------------------------------------------------------------------
# Superquadric geometry
# ---------------------------------------------------------------------------

def add_superquadric_compact_rot_mat(
        scalings=np.array([1.0, 1.0, 1.0]),
        exponents=np.array([2.0, 2.0, 2.0]),
        translation=np.array([0.0, 0.0, 0.0]),
        rotation=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        resolution=10,
        visible=True):

    def create_superquadric_mesh(A, B, C, e1, e2, N):
        def f(o, m):
            return np.sign(np.sin(o)) * np.abs(np.sin(o))**m
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o))**m
        u = np.tile(np.linspace(-np.pi, np.pi, N, endpoint=True), N)
        v = np.repeat(np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True), N)
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
                triangles.append([i*N+j, i*N+j+1, (i+1)*N+j])
                triangles.append([(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)])
        for i in range(N - 1):
            triangles.append([i*N+(N-1), i*N, (i+1)*N+(N-1)])
            triangles.append([(i+1)*N+(N-1), i*N, (i+1)*N])
        triangles.append([(N-1)*N+(N-1), (N-1)*N, (N-1)])
        triangles.append([(N-1), (N-1)*N, 0])
        return vertices, triangles

    return create_superquadric_mesh(scalings[0], scalings[1], scalings[2],
                                    exponents[0], exponents[1], resolution)


# ---------------------------------------------------------------------------
# Superquadric file I/O
# ---------------------------------------------------------------------------

def load_superquadric_from_file(file_path):
    par_dict = np.load(file_path)
    scale = par_dict['scales']
    rotate = par_dict['rotations']
    shapes = par_dict['shapes']
    trans = par_dict['translations']
    sqs = {}
    for k in range(scale.shape[0]):
        sqs[k] = {
            'scale': scale[k, :],
            'shape': shapes[k],
            'rotation': rotate[k, :],
            'translation': trans[k, :],
            'color': [90, 200, 255],
        }
    return sqs


def save_superquadric_to_file(sqs, file_path):
    np.savez(file_path,
             scales=np.array([sqs[k]['scale'] for k in sqs]),
             rotations=np.array([sqs[k]['rotation'] for k in sqs]),
             shapes=np.array([sqs[k]['shape'] for k in sqs]),
             translations=np.array([sqs[k]['translation'] for k in sqs]))


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def get_all_templates():
    return {i: f.split('_')[0]
            for i, f in enumerate(sorted(os.listdir('gui/superquadrics/')))
            if f.endswith('_sq.npz')}


def get_all_appearance_meshes():
    return {i: f
            for i, f in enumerate(sorted(os.listdir('gui/appearance_meshes/')))
            if f.endswith('.glb')}


# ---------------------------------------------------------------------------
# Generation — calls run.py as a subprocess
# ---------------------------------------------------------------------------

def _run_pipeline(cmd, on_done):
    """Run run.py in a background thread; call on_done(returncode) when finished."""
    print(f"=== run.py starting: {' '.join(cmd)} ===", flush=True)
    proc = subprocess.run(cmd)  # inherits parent stdout/stderr → goes to .out/.err
    print(f"=== run.py finished (returncode={proc.returncode}) ===", flush=True)
    on_done(proc.returncode)


def generate(superquadrics, gui_state) -> None:
    import json as _json
    btn = gui_elements['generate_button']
    btn.label = "Generating..."
    btn.icon = viser.Icon.LOADER
    btn.color = 'orange'
    btn.disabled = True

    # Save current superquadric state to a temp .npz so run.py can read it
    sq_npz_path = 'gui/tmp_superquadrics.npz'
    save_superquadric_to_file(superquadrics, sq_npz_path)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    text_prompt = gui_state['text_prompt']
    output_dir = f'outputs/{timestamp}_{text_prompt.replace(" ", "_")}'
    os.makedirs(output_dir, exist_ok=True)

    mode = gui_state['guidance_mode']
    run_mode = 'appearance' if mode == 'appearance' else 'similarity'

    python = sys.executable
    cmd = [
        python, 'run.py',
        '--guidance_mode', run_mode,
        '--output_dir', output_dir,
        '--shape_superquadric_path', sq_npz_path,
        '--shape_tau', str(gui_state['shape_tau']),
        '--text_prompt', text_prompt,
    ]

    if gui_state.get('convert_yup_to_zup'):
        cmd.append('--convert_yup_to_zup')

    if mode == 'appearance':
        cmd += ['--appearance_mesh', gui_state['appearance_mesh']]
        if gui_state.get('appearance_image_path'):
            cmd += ['--appearance_image', gui_state['appearance_image_path']]
    elif mode == 'image-similarity':
        if gui_state.get('appearance_image_path'):
            cmd += ['--appearance_image', gui_state['appearance_image_path']]
        local_image_paths = gui_state.get('local_image_paths', [])
        if any(p for p in local_image_paths):
            cmd += ['--local_image_paths', _json.dumps(local_image_paths)]
    elif mode == 'text-similarity':
        if gui_state.get('global_texture'):
            cmd += ['--appearance_text', gui_state['global_texture']]
        local_text_prompts = gui_state.get('local_text_prompts', [])
        if any(p.strip() for p in local_text_prompts):
            cmd += ['--local_text_prompts', _json.dumps(local_text_prompts)]

    print(f"Running: {' '.join(cmd)}")

    # Collect temp local image paths for cleanup
    local_image_paths_for_cleanup = gui_state.get('local_image_paths', []) or []

    def on_done(returncode):
        global generated_mesh
        # Clean up temp files used by this run
        cleanup_paths = [sq_npz_path, gui_state.get('appearance_image_path')] + local_image_paths_for_cleanup
        for tmp in cleanup_paths:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        gui_elements.pop('_appearance_image_path', None)
        # Clear stored local image paths from superquadrics
        for sq_id in superquadrics:
            superquadrics[sq_id].pop('local_image_path', None)

        btn = gui_elements['generate_button']
        if returncode == 0:
            candidate = 'out_app.glb' if gui_state['guidance_mode'] == 'appearance' else 'out_sim.glb'
            glb_path = os.path.join(output_dir, candidate)
            if os.path.exists(glb_path):
                try:
                    with open(glb_path, 'rb') as f:
                        glb_data = f.read()
                    generated_mesh = server.scene.add_glb("generated_mesh", glb_data, visible=True)
                    for key in scene_elements:
                        if key.startswith('sq_'):
                            scene_elements[key].visible = False
                    if active_superquadric != -1:
                        scene_elements[f'sqc_{active_superquadric}'].visible = False
                    server.scene.set_environment_map('studio', background=False, environment_intensity=2.0)
                    print(f"Displaying {glb_path}", flush=True)
                except Exception as e:
                    print(f"Could not display mesh: {e}", flush=True)
            else:
                print(f"Output mesh not found: {glb_path}", flush=True)
            btn.label = "Generate"
            btn.color = 'green'
        else:
            btn.label = "FAILED — retry?"
            btn.color = 'red'
        btn.icon = viser.Icon.PLAYER_PLAY
        btn.disabled = False

    threading.Thread(target=_run_pipeline, args=(cmd, on_done), daemon=True).start()


# ---------------------------------------------------------------------------
# GUI setup
# ---------------------------------------------------------------------------

def _save_and_preview_image(upload_handle, folder_key, preview_key):
    path = 'gui/tmp_appearance_image.png'
    img = Image.open(BytesIO(upload_handle.value.content)).convert('RGB')
    img.save(path)
    gui_elements['_appearance_image_path'] = path
    with gui_elements[folder_key]:
        try:
            gui_elements[preview_key].remove()
        except Exception:
            pass
        gui_elements[preview_key] = server.gui.add_image(np.array(img))
    print(f"Appearance image saved to {path}")


def handle_upload_texture_image(event):
    _save_and_preview_image(texture_image_handle, 'folder_image_sim', 'texture_image_preview')


def handle_upload_app_image(event):
    _save_and_preview_image(app_image_handle, 'folder_appearance', 'app_image_preview')


def setup_gui(server, superquadrics):
    global gui_elements, scene_elements, active_template_id, active_superquadric
    global texture_image_handle, app_image_handle

    gui_elements = {}
    active_superquadric = -1
    server.gui.reset()
    server.scene.reset()
    scene_elements = {}

    server.gui.set_panel_label("SpaceFlow")

    # --- Shape controls ---
    templates = get_all_templates()
    select_template_dropdown = server.gui.add_dropdown(
        label="Object Template",
        options=list(templates.values()),
        order=0,
        initial_value=templates[active_template_id])
    select_template_dropdown.on_update(
        lambda _: select_template_from_id(
            [k for k, v in get_all_templates().items() if v == select_template_dropdown.value][0]))
    gui_elements['select_template_dropdown'] = select_template_dropdown

    t0_idx = server.gui.add_slider(
        "Control strength (tau)", order=1,
        min=0, max=steps, step=1.0, initial_value=6.0,
        marks=((0, "0"), (steps // 3, f"{steps // 3}"), (2 * steps // 3, f"{2 * steps // 3}")))

    convert_checkbox = server.gui.add_checkbox("Convert Y-up \u2192 Z-up", initial_value=True, order=2)
    gui_elements['convert_checkbox'] = convert_checkbox

    # Per-superquadric folders
    def make_sq_image_upload_handler(sq_id):
        def handler(event):
            # Access image data from the upload button handle, not the event object.
            handle = gui_elements[f'sq_{sq_id}']['local_image_upload']
            path = f'gui/tmp_local_image_{sq_id}.png'
            img = Image.open(BytesIO(handle.value.content)).convert('RGB')
            img.save(path)
            superquadrics[sq_id]['local_image_path'] = path
            # Show preview inside the SQ folder
            with gui_elements[f'sq_{sq_id}']['folder']:
                if 'local_image_preview' in gui_elements[f'sq_{sq_id}']:
                    try:
                        gui_elements[f'sq_{sq_id}']['local_image_preview'].remove()
                    except Exception:
                        pass
                gui_elements[f'sq_{sq_id}']['local_image_preview'] = server.gui.add_image(
                    np.array(img.resize((128, 128))))
            print(f"Local image for SQ {sq_id} saved to {path}")
        return handler

    for id, sq in superquadrics.items():
        per = {}
        per['folder'] = server.gui.add_folder(f'Superquadric {id}', order=3, expand_by_default=True, visible=False)
        with per['folder']:
            per['shape_1'] = server.gui.add_slider("Shape 1", min=0, max=2, step=0.01, initial_value=sq['shape'][0], marks=((0,"0"),(1,"1"),(2,"2")))
            per['shape_2'] = server.gui.add_slider("Shape 2", min=0, max=2, step=0.01, initial_value=sq['shape'][1], marks=((0,"0"),(1,"1"),(2,"2")))
            per['scale_x'] = server.gui.add_slider("Scale X", min=0, max=1, step=0.002, initial_value=sq['scale'][0], marks=((0,"0"),(1,"1")))
            per['scale_y'] = server.gui.add_slider("Scale Y", min=0, max=1, step=0.002, initial_value=sq['scale'][1], marks=((0,"0"),(1,"1")))
            per['scale_z'] = server.gui.add_slider("Scale Z", min=0, max=1, step=0.002, initial_value=sq['scale'][2], marks=((0,"0"),(1,"1")))
            for k in per:
                try:
                    per[k].on_update(lambda _: update_sq(superquadrics, active_superquadric, RESOLUTION))
                except Exception:
                    pass
            per['local_text_prompt'] = server.gui.add_text(
                "Local texture", initial_value=sq.get('local_text_prompt', ''), visible=False)
            per['local_image_upload'] = server.gui.add_upload_button(
                "Upload local texture", color='gray', visible=False)
            per['local_image_upload'].on_upload(make_sq_image_upload_handler(id))
            per['duplicate_button'] = server.gui.add_button("Duplicate", color='blue', icon=viser.Icon.COPY)
            per['duplicate_button'].on_click(lambda _: duplicate_active_superquadric())
            per['delete_button'] = server.gui.add_button("Delete", color='red', icon=viser.Icon.CROSS)
            per['delete_button'].on_click(lambda _: delete_active_superquadric())
        gui_elements[f'sq_{id}'] = per

    gui_elements['save_sq_button'] = server.gui.add_button("Save as Template", color='gray', icon=viser.Icon.WRITING, order=4)

    # --- Guidance mode ---
    guidance_dropdown = server.gui.add_dropdown(
        label="Guidance mode",
        options=['text-similarity', 'image-similarity', 'appearance'],
        order=6,
        initial_value='text-similarity')
    gui_elements['guidance_dropdown'] = guidance_dropdown

    global_shape = server.gui.add_text("Global shape", "chair", order=7)
    gui_elements['global_shape'] = global_shape

    gui_elements['save_sq_button'].on_click(
        lambda _: save_superquadric_to_file(
            superquadrics, f'gui/superquadrics/{global_shape.value}_sq.npz'))

    # --- text-similarity: Global texture text field (visible by default) ---
    gui_elements['folder_text_sim'] = server.gui.add_folder(
        "Text similarity", order=8, expand_by_default=True, visible=True)
    with gui_elements['folder_text_sim']:
        global_texture = server.gui.add_text("Global texture", "", order=0)
        gui_elements['global_texture'] = global_texture

    # --- image-similarity: upload button only (hidden by default) ---
    gui_elements['folder_image_sim'] = server.gui.add_folder(
        "Image similarity", order=9, expand_by_default=True, visible=False)
    with gui_elements['folder_image_sim']:
        texture_image_handle = server.gui.add_upload_button("Upload texture image", color='gray', order=0)
        texture_image_handle.on_upload(handle_upload_texture_image)
        gui_elements['texture_image_handle'] = texture_image_handle

    # --- Appearance: mesh dropdown + optional texture upload (hidden by default) ---
    app_meshes = get_all_appearance_meshes()
    gui_elements['folder_appearance'] = server.gui.add_folder(
        "Appearance guidance", order=10, expand_by_default=True, visible=False)
    with gui_elements['folder_appearance']:
        appearance_mesh_dropdown = server.gui.add_dropdown(
            label="Select appearance mesh",
            options=list(app_meshes.values()),
            order=0,
            initial_value=list(app_meshes.values())[0] if app_meshes else "")
        gui_elements['appearance_mesh_dropdown'] = appearance_mesh_dropdown
        app_image_handle = server.gui.add_upload_button("Upload texture image (optional)", color='gray', order=1)
        app_image_handle.on_upload(handle_upload_app_image)

    # Show/hide exactly one guidance section when dropdown changes;
    # also toggle per-SQ local controls (text-similarity ↔ image-similarity only).
    def _on_guidance_change(_):
        mode = guidance_dropdown.value
        gui_elements['folder_text_sim'].visible = (mode == 'text-similarity')
        gui_elements['folder_image_sim'].visible = (mode == 'image-similarity')
        gui_elements['folder_appearance'].visible = (mode == 'appearance')
        gui_elements['global_shape'].visible = (mode == 'text-similarity')
        for sq_id in superquadrics:
            per = gui_elements.get(f'sq_{sq_id}')
            if per is None:
                continue
            per['local_text_prompt'].visible = (mode == 'text-similarity')
            per['local_image_upload'].visible = (mode == 'image-similarity')
    guidance_dropdown.on_update(_on_guidance_change)

    # --- Generate buttons ---
    def _collect_state():
        mode = guidance_dropdown.value
        selected_mesh = gui_elements['appearance_mesh_dropdown'].value
        sq_ids = sorted(superquadrics.keys())
        text_prompt = global_shape.value
        if mode == 'image-similarity':
            img_path = gui_elements.get('_appearance_image_path')
            text_prompt = os.path.splitext(os.path.basename(img_path))[0] if img_path else 'image_similarity'
        return {
            'text_prompt': text_prompt,
            'shape_tau': t0_idx.value,
            'guidance_mode': mode,
            'convert_yup_to_zup': convert_checkbox.value,
            'global_texture': global_texture.value,
            'appearance_image_path': gui_elements.get('_appearance_image_path'),
            'appearance_mesh': os.path.join('gui/appearance_meshes', selected_mesh) if selected_mesh else '',
            'local_text_prompts': [
            gui_elements[f'sq_{i}']['local_text_prompt'].value for i in sq_ids
            ],
            'local_image_paths': [
                superquadrics[i].get('local_image_path') for i in sq_ids
            ],
        }

    gui_elements['generate_button'] = server.gui.add_button(
        "Generate", color='green', icon=viser.Icon.PLAYER_PLAY, order=11)
    gui_elements['generate_button'].on_click(
        lambda _: generate(superquadrics, _collect_state()))

    toggle_button = server.gui.add_button("Toggle mesh/primitives", color='gray', order=13)
    toggle_button.on_click(lambda _: toggle_sq_mesh())

    return gui_elements


# ---------------------------------------------------------------------------
# Scene manipulation helpers
# ---------------------------------------------------------------------------

def duplicate_active_superquadric():
    global superquadrics, scene_elements, gui_elements, active_superquadric
    new_id = max(superquadrics.keys()) + 1
    superquadrics[new_id] = copy.deepcopy(superquadrics[active_superquadric])
    superquadrics[new_id]['translation'] += np.array([0.02, 0.02, 0.02])
    copied = active_superquadric
    gui_elements = setup_gui(server, superquadrics)
    for sid in superquadrics:
        add_superquadric(superquadrics, sid, gui_elements, RESOLUTION)
    active_superquadric = copied
    sq_on_click(new_id)


def delete_active_superquadric():
    global superquadrics, scene_elements, gui_elements, active_superquadric
    if active_superquadric == -1:
        return
    superquadrics.pop(active_superquadric)
    scene_elements[f'sq_{active_superquadric}'].remove()
    scene_elements[f'sqc_{active_superquadric}'].remove()
    gui_elements[f'sq_{active_superquadric}']['folder'].visible = False
    del scene_elements[f'sq_{active_superquadric}']
    del scene_elements[f'sqc_{active_superquadric}']
    active_superquadric = -1


def toggle_sq_mesh():
    global generated_mesh
    if generated_mesh is None:
        return
    generated_mesh.visible = not generated_mesh.visible
    intensity = 2.0 if generated_mesh.visible else 0.5
    server.scene.set_environment_map('studio', background=False, environment_intensity=intensity)
    for key in scene_elements:
        if key.startswith('sq_'):
            scene_elements[key].visible = not generated_mesh.visible
    if active_superquadric != -1:
        scene_elements[f'sqc_{active_superquadric}'].visible = not generated_mesh.visible


def update_sq(superquadrics, superquadric_id, resolution):
    if superquadric_id == -1:
        return
    per = gui_elements[f'sq_{superquadric_id}']
    superquadrics[superquadric_id]['shape'][0] = per['shape_1'].value
    superquadrics[superquadric_id]['shape'][1] = per['shape_2'].value
    superquadrics[superquadric_id]['scale'][0] = per['scale_x'].value
    superquadrics[superquadric_id]['scale'][1] = per['scale_y'].value
    superquadrics[superquadric_id]['scale'][2] = per['scale_z'].value
    add_superquadric(superquadrics, superquadric_id, gui_elements, resolution)


def add_superquadric(superquadrics, superquadric_id, gui_elements, resolution):
    global scene_elements

    vertices, triangles = add_superquadric_compact_rot_mat(
        superquadrics[superquadric_id]['scale'],
        superquadrics[superquadric_id]['shape'],
        superquadrics[superquadric_id]['translation'],
        superquadrics[superquadric_id]['rotation'],
        resolution)

    scene_elements[f'sq_{superquadric_id}'] = server.scene.add_mesh_simple(
        name=f"/sq/{superquadric_id}",
        vertices=vertices,
        color=superquadrics[superquadric_id]['color'],
        faces=np.array(triangles))

    scene_elements[f'sqc_{superquadric_id}'] = server.scene.add_transform_controls(
        f'sqc_{superquadric_id}', scale=0.2, line_width=2.5, fixed=False,
        visible=superquadric_id == active_superquadric,
        active_axes=[True, True, True], depth_test=False,
        position=superquadrics[superquadric_id]['translation'],
        wxyz=tf.SO3.from_matrix(superquadrics[superquadric_id]['rotation']).wxyz)

    @scene_elements[f'sqc_{superquadric_id}'].on_update
    def _(_):
        superquadrics[superquadric_id]['translation'] = scene_elements[f'sqc_{superquadric_id}'].position
        superquadrics[superquadric_id]['rotation'] = tf.SO3.as_matrix(scene_elements[f'sqc_{superquadric_id}'])
        update_sq(superquadrics, superquadric_id, RESOLUTION)

    if active_superquadric != superquadric_id:
        scene_elements[f'sq_{superquadric_id}'].on_click(lambda _: sq_on_click(superquadric_id))


def sq_on_click(superquadric_id):
    global active_superquadric
    if active_superquadric != -1:
        scene_elements[f'sq_{active_superquadric}'].on_click(lambda _: sq_on_click(active_superquadric))
        superquadrics[active_superquadric]['color'] = [90, 200, 255]
    active_superquadric = superquadric_id
    scene_elements[f'sq_{active_superquadric}'].remove_click_callback('all')
    superquadrics[active_superquadric]['color'] = [255, 0, 255]
    for i in superquadrics:
        gui_elements[f'sq_{i}']['folder'].visible = (i == active_superquadric)
        scene_elements[f'sqc_{i}'].visible = (i == active_superquadric)
    for i in superquadrics:
        update_sq(superquadrics, i, RESOLUTION)


def select_template_from_id(template_id):
    global active_template_id, superquadrics
    active_template_id = template_id
    templates = get_all_templates()
    input_path = f'gui/superquadrics/{templates[template_id]}_sq.npz'
    print(f"Loading superquadrics from {input_path}")
    superquadrics = load_superquadric_from_file(input_path)
    els = setup_gui(server, superquadrics)
    for sid in range(len(superquadrics)):
        add_superquadric(superquadrics, sid, els, RESOLUTION)


def main():
    select_template_from_id(0)
    while True:
        time.sleep(10.0)

if __name__ == '__main__':
    main()
