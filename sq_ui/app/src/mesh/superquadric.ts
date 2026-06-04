/**
 * Superquadric mesh generation — exact port of run_local_tau.py's
 * add_superquadric_compact_rot_mat / create_superquadric_mesh.
 */

export interface SuperquadricParams {
  scales: [number, number, number];     // A, B, C half-axes
  shapes: [number, number];             // e1, e2 exponents
  translation: [number, number, number];
  rotation: number[][];                 // 3×3
}

/** Optional tapering and bending applied in local space before world rotation. */
export interface SuperquadricDeform {
  tapering: [number, number];
  /** [k_z, α_z, k_x, α_x, k_y, α_y] — matches Python visualization order. */
  bending: [number, number, number, number, number, number];
}

function f(o: number, m: number): number {
  return Math.sign(Math.sin(o)) * Math.pow(Math.abs(Math.sin(o)), m);
}

function g(o: number, m: number): number {
  return Math.sign(Math.cos(o)) * Math.pow(Math.abs(Math.cos(o)), m);
}

function det3(R: number[][]): number {
  return (
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
  );
}

function applyTaperToArrays(
  x: Float64Array,
  y: Float64Array,
  z: Float64Array,
  C: number,
  kx: number,
  ky: number,
): void {
  for (let k = 0; k < x.length; k++) {
    const zNorm = z[k] / C;
    const fx = kx * zNorm + 1;
    const fy = ky * zNorm + 1;
    x[k] *= fx;
    y[k] *= fy;
  }
}

function applyBendingAxisArrays(
  x: Float64Array,
  y: Float64Array,
  z: Float64Array,
  valKb: number,
  valAlpha: number,
  axis: 'x' | 'y' | 'z',
): void {
  if (Math.abs(valKb) < 1e-3) return;
  const sinAlpha = Math.sin(valAlpha);
  const cosAlpha = Math.cos(valAlpha);
  for (let k = 0; k < x.length; k++) {
    let u: number;
    let vCoord: number;
    let w: number;
    const xi = x[k];
    const yi = y[k];
    const zi = z[k];
    if (axis === 'z') {
      u = xi;
      vCoord = yi;
      w = zi;
    } else if (axis === 'x') {
      u = yi;
      vCoord = zi;
      w = xi;
    } else {
      u = zi;
      vCoord = xi;
      w = yi;
    }
    const beta = Math.atan2(vCoord, u);
    const r = Math.sqrt(u * u + vCoord * vCoord) * Math.cos(valAlpha - beta);
    const invKb = 1.0 / valKb;
    const gamma = w * valKb;
    const rho = invKb - r;
    const Rb = invKb - rho * Math.cos(gamma);
    const expr = Rb - r;
    u += expr * cosAlpha;
    vCoord += expr * sinAlpha;
    w = rho * Math.sin(gamma);
    if (axis === 'z') {
      x[k] = u;
      y[k] = vCoord;
      z[k] = w;
    } else if (axis === 'x') {
      x[k] = w;
      y[k] = u;
      z[k] = vCoord;
    } else {
      x[k] = vCoord;
      y[k] = w;
      z[k] = u;
    }
  }
}

export function createSuperquadricMesh(
  A: number, B: number, C: number,
  e1: number, e2: number,
  rotation: number[][],
  translation: [number, number, number],
  N: number,
  deform?: SuperquadricDeform,
): { vertices: Float64Array; indices: Uint32Array } {
  const u = new Float64Array(N);
  const v = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    u[i] = -Math.PI + (2 * Math.PI * i) / (N - 1);
    v[i] = -Math.PI / 2 + (Math.PI * i) / (N - 1);
  }

  // Tile u N times, repeat v N times (matching np.tile / np.repeat)
  const totalVerts = N * N;
  const uFull = new Float64Array(totalVerts);
  const vFull = new Float64Array(totalVerts);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      uFull[i * N + j] = u[j];
      vFull[i * N + j] = v[i];
    }
  }

  // det(R) < 0 → reverse u
  if (det3(rotation) < 0) {
    uFull.reverse();
  }

  const x = new Float64Array(totalVerts);
  const y = new Float64Array(totalVerts);
  const z = new Float64Array(totalVerts);

  for (let k = 0; k < totalVerts; k++) {
    x[k] = A * g(vFull[k], e1) * g(uFull[k], e2);
    y[k] = B * g(vFull[k], e1) * f(uFull[k], e2);
    z[k] = C * f(vFull[k], e1);
  }

  // Pole fix: first N and last N vertices have x=0
  for (let k = 0; k < N; k++) {
    x[k] = 0.0;
    x[totalVerts - N + k] = 0.0;
  }

  if (deform) {
    const [kx, ky] = deform.tapering;
    const b = deform.bending;
    applyTaperToArrays(x, y, z, C, kx, ky);
    applyBendingAxisArrays(x, y, z, b[4], b[5], 'y');
    applyBendingAxisArrays(x, y, z, b[2], b[3], 'x');
    applyBendingAxisArrays(x, y, z, b[0], b[1], 'z');
  }

  // Apply rotation and translation: vertices = (R @ verts.T).T + translation
  const vertices = new Float64Array(totalVerts * 3);
  for (let k = 0; k < totalVerts; k++) {
    const vx = x[k], vy = y[k], vz = z[k];
    vertices[k * 3]     = rotation[0][0] * vx + rotation[0][1] * vy + rotation[0][2] * vz + translation[0];
    vertices[k * 3 + 1] = rotation[1][0] * vx + rotation[1][1] * vy + rotation[1][2] * vz + translation[1];
    vertices[k * 3 + 2] = rotation[2][0] * vx + rotation[2][1] * vy + rotation[2][2] * vz + translation[2];
  }

  // Build triangles (matching run_local_tau.py exactly)
  const triangles: number[] = [];
  for (let i = 0; i < N - 1; i++) {
    for (let j = 0; j < N - 1; j++) {
      triangles.push(i * N + j, i * N + j + 1, (i + 1) * N + j);
      triangles.push((i + 1) * N + j, i * N + j + 1, (i + 1) * N + (j + 1));
    }
  }
  // Wrap-around: connect last and first vertex in each row
  for (let i = 0; i < N - 1; i++) {
    triangles.push(i * N + (N - 1), i * N, (i + 1) * N + (N - 1));
    triangles.push((i + 1) * N + (N - 1), i * N, (i + 1) * N);
  }
  // Final two triangles
  triangles.push((N - 1) * N + (N - 1), (N - 1) * N, N - 1);
  triangles.push(N - 1, (N - 1) * N, 0);

  return { vertices, indices: new Uint32Array(triangles) };
}

/**
 * Pipeline-style normalization: center AABB and uniform-scale to fit max extent = 1.
 */
export function normalizeMergedVertices(
  allVertices: Float64Array[],
): { center: [number, number, number]; scale: number } {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const verts of allVertices) {
    for (let i = 0; i < verts.length; i += 3) {
      minX = Math.min(minX, verts[i]);
      minY = Math.min(minY, verts[i + 1]);
      minZ = Math.min(minZ, verts[i + 2]);
      maxX = Math.max(maxX, verts[i]);
      maxY = Math.max(maxY, verts[i + 1]);
      maxZ = Math.max(maxZ, verts[i + 2]);
    }
  }
  const center: [number, number, number] = [
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2,
  ];
  const scale = 1 / Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  return { center, scale };
}
