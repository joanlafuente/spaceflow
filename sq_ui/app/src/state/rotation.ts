/**
 * Euler angles (ZYX intrinsic, degrees) ↔ 3×3 rotation matrix conversions.
 */

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export function eulerToMatrix(eulerDeg: [number, number, number]): number[][] {
  const [rx, ry, rz] = eulerDeg.map(d => d * DEG);
  const cx = Math.cos(rx), sx = Math.sin(rx);
  const cy = Math.cos(ry), sy = Math.sin(ry);
  const cz = Math.cos(rz), sz = Math.sin(rz);

  // ZYX: R = Rz * Ry * Rx
  return [
    [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
    [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
    [-sy,     sx * cy,                cx * cy],
  ];
}

export function matrixToEuler(R: number[][]): [number, number, number] {
  let ry = Math.asin(-clamp(R[2][0], -1, 1));
  let rx: number, rz: number;

  if (Math.abs(R[2][0]) < 0.99999) {
    rx = Math.atan2(R[2][1], R[2][2]);
    rz = Math.atan2(R[1][0], R[0][0]);
  } else {
    // Gimbal lock
    rx = Math.atan2(-R[1][2], R[1][1]);
    rz = 0;
  }

  return [rx * RAD, ry * RAD, rz * RAD];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function isOrthogonal(R: number[][], tol = 1e-4): boolean {
  // Check R^T R ≈ I
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let dot = 0;
      for (let k = 0; k < 3; k++) dot += R[k][i] * R[k][j];
      const expected = i === j ? 1 : 0;
      if (Math.abs(dot - expected) > tol) return false;
    }
  }
  return true;
}

export function det3(R: number[][]): number {
  return (
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
  );
}

/** 3×3 matrix product (row-major), C = A B. */
export function matMul3(a: number[][], b: number[][]): number[][] {
  const out: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

/** Column vector v → M v (world-space linear map). */
export function matVec3(m: number[][], v: readonly [number, number, number]): [number, number, number] {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}
