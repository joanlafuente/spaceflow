#!/usr/bin/env python3
"""
Validate that the NPZ writer in the browser produces files compatible
with run.py's load_superquadric_from_file.

Usage: python test_npz_compat.py <path_to_npz>
  or: python test_npz_compat.py --generate  (creates a test .npz for comparison)
"""
import sys
import numpy as np


def load_superquadric_from_file(file_path: str) -> dict:
    """Exact copy from run.py for standalone testing."""
    par_dict = np.load(file_path)
    scale = par_dict['scales']
    rotate = par_dict['rotations']
    shapes = par_dict['shapes']
    trans = par_dict['translations']
    num_el = scale.shape[0]

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


def validate_npz(path: str):
    print(f"Loading: {path}")
    data = np.load(path)

    required_keys = {'scales', 'shapes', 'translations', 'rotations'}
    actual_keys = set(data.files)
    missing = required_keys - actual_keys
    assert not missing, f"Missing keys: {missing}"
    print(f"  Keys: {sorted(data.files)}")

    N = data['scales'].shape[0]
    print(f"  N (primitives): {N}")

    assert data['scales'].shape == (N, 3), f"scales shape {data['scales'].shape} != ({N}, 3)"
    assert data['shapes'].shape == (N, 2), f"shapes shape {data['shapes'].shape} != ({N}, 2)"
    assert data['translations'].shape == (N, 3), f"translations shape {data['translations'].shape} != ({N}, 3)"
    assert data['rotations'].shape == (N, 3, 3), f"rotations shape {data['rotations'].shape} != ({N}, 3, 3)"

    for key in required_keys:
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    # Validate rotations
    for i in range(N):
        R = data['rotations'][i]
        RtR = R.T @ R
        err = np.abs(RtR - np.eye(3)).max()
        det = np.linalg.det(R)
        ok = err < 1e-3 and abs(abs(det) - 1) < 1e-3
        status = "OK" if ok else "WARN"
        print(f"  R[{i}]: det={det:.6f}, orthogonality_err={err:.2e} [{status}]")

    # Validate scales
    for i in range(N):
        s = data['scales'][i]
        assert np.all(s > 0), f"scales[{i}] has non-positive value: {s}"

    # Test pipeline compat
    sq = load_superquadric_from_file(path)
    assert len(sq) == N
    print(f"\n  Pipeline compatibility: PASS ({N} superquadrics loaded)")
    print("  All checks passed!")


def generate_test(path: str = "test_export.npz"):
    N = 3
    scales = np.array([[1.0, 0.5, 0.8], [0.3, 0.3, 1.2], [0.6, 0.6, 0.1]], dtype=np.float64)
    shapes = np.array([[2.0, 2.0], [0.3, 0.3], [2.0, 0.3]], dtype=np.float64)
    translations = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    rotations = np.array([np.eye(3)] * N, dtype=np.float64)
    np.savez(path, scales=scales, shapes=shapes, translations=translations, rotations=rotations)
    print(f"Generated: {path}")
    validate_npz(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_npz_compat.py <path.npz>")
        print("       python test_npz_compat.py --generate")
        sys.exit(1)

    if sys.argv[1] == "--generate":
        generate_test()
    else:
        validate_npz(sys.argv[1])
