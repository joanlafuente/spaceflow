/**
 * Load superquadric .npz (ZIP of .npy) produced by the SQ Editor export or run.py.
 */
import JSZip from 'jszip';
import type { Primitive } from '../state/store';
import { matrixToEuler } from '../state/rotation';
import type { PrimitiveExport } from './npzExport';

/**
 * Maps a Z-up world frame to the editor's Y-up frame (Three.js):
 * x' = x, y' = z, z' = -y. Use when pipeline assets were authored with vertical Z.
 * World pose: p' = S @ p, so R' = S @ R, t' = S @ t for column vectors.
 */
const S_ZUP_TO_YUP: number[][] = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

function matMul3x3(A: number[][], B: number[][]): number[][] {
  const C: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

function matVec3(R: number[][], v: [number, number, number]): [number, number, number] {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

export function applyWorldBasisZUpToYUp(
  R: number[][],
  t: [number, number, number],
): { rotation: number[][]; translation: [number, number, number] } {
  return {
    rotation: matMul3x3(S_ZUP_TO_YUP, R),
    translation: matVec3(S_ZUP_TO_YUP, t),
  };
}

interface ParsedNpy {
  shape: number[];
  /** C-order row-major values as float64 */
  values: Float64Array;
}

function squeezeSingletonAxis(
  parsed: ParsedNpy,
  trailingShape: number[],
  label: string,
): ParsedNpy {
  let shape = [...parsed.shape];
  while (shape.length > trailingShape.length + 1 && shape[1] === 1) {
    shape.splice(1, 1);
  }
  if (shape.length === trailingShape.length + 2 && shape[1] === 1) {
    shape.splice(1, 1);
  }
  const expected = [shape[0], ...trailingShape];
  if (shape.length !== expected.length || !expected.every((v, i) => shape[i] === v)) {
    throw new Error(`${label} expected shape (N,${trailingShape.join(',')}), got (${parsed.shape.join(',')})`);
  }
  return { ...parsed, shape };
}

function parseNpyHeader(headerText: string): { descr: string; shape: number[]; fortran: boolean } {
  const descrM = /'descr':\s*'([^']+)'/.exec(headerText);
  if (!descrM) throw new Error('NPY header missing descr');
  const descr = descrM[1];
  const shapeM = /'shape':\s*\(([^)]*)\)/.exec(headerText);
  if (!shapeM) throw new Error('NPY header missing shape');
  const raw = shapeM[1].trim();
  const shape =
    raw === ''
      ? []
      : raw.split(',')
          .map(s => s.trim())
          .filter(s => s.length > 0)
          .map(s => {
            const n = Number(s);
            if (!Number.isFinite(n)) throw new Error(`Invalid shape segment: ${s}`);
            return n;
          });
  const fortran = /'fortran_order':\s*True/.test(headerText);
  return { descr, shape, fortran };
}

function readValues(
  data: Uint8Array,
  offset: number,
  descr: string,
  n: number,
): Float64Array {
  const out = new Float64Array(n);
  if (descr === '<f8') {
    const view = new DataView(data.buffer, data.byteOffset + offset, n * 8);
    for (let i = 0; i < n; i++) out[i] = view.getFloat64(i * 8, true);
    return out;
  }
  if (descr === '<f4') {
    const view = new DataView(data.buffer, data.byteOffset + offset, n * 4);
    for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
    return out;
  }
  throw new Error(`Unsupported NPY descr ${descr} (need <f8 or <f4)`);
}

export function parseNpyBuffer(buffer: ArrayBuffer): ParsedNpy {
  const u8 = new Uint8Array(buffer);
  if (u8.length < 10 || u8[0] !== 0x93 || u8[1] !== 0x4e || u8[2] !== 0x55 || u8[3] !== 0x4d || u8[4] !== 0x50 || u8[5] !== 0x59) {
    throw new Error('Not a NumPy .npy file');
  }
  if (u8[6] !== 1 || u8[7] !== 0) {
    throw new Error('Only NumPy format v1.0 is supported');
  }
  const headerLen = u8[8] | (u8[9] << 8);
  const headerText = new TextDecoder('latin1').decode(u8.slice(10, 10 + headerLen));
  const { descr, shape, fortran } = parseNpyHeader(headerText);
  if (fortran) {
    throw new Error('Fortran-ordered arrays are not supported');
  }
  let numEl = 1;
  for (const s of shape) numEl *= s;
  const offset = 10 + headerLen;
  const values = readValues(u8, offset, descr, numEl);
  return { shape, values };
}

function reshape2(values: Float64Array, rows: number, cols: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) row.push(values[i * cols + j]);
    out.push(row);
  }
  return out;
}

function reshape3(values: Float64Array, a: number, b: number, c: number): number[][][] {
  const out: number[][][] = [];
  let idx = 0;
  for (let i = 0; i < a; i++) {
    const plane: number[][] = [];
    for (let j = 0; j < b; j++) {
      const row: number[] = [];
      for (let k = 0; k < c; k++) row.push(values[idx++]);
      plane.push(row);
    }
    out.push(plane);
  }
  return out;
}

/** Parse arrays from an exported / pipeline-compatible .npz into primitive data. */
export function npzArraysToExports(
  scales: ParsedNpy,
  shapes: ParsedNpy,
  translations: ParsedNpy,
  rotations: ParsedNpy,
): PrimitiveExport[] {
  const scalesSq = squeezeSingletonAxis(scales, [3], 'scales.npy');
  const shapesSq = squeezeSingletonAxis(shapes, [2], 'shapes.npy');
  const translationsSq = squeezeSingletonAxis(translations, [3], 'translations.npy');
  const rotationsSq = squeezeSingletonAxis(rotations, [3, 3], 'rotations.npy');

  const N = scalesSq.shape[0];
  if (shapesSq.shape[0] !== N || translationsSq.shape[0] !== N || rotationsSq.shape[0] !== N) {
    throw new Error(`Array length mismatch: scales ${N}, shapes ${shapesSq.shape[0]}, translations ${translationsSq.shape[0]}, rotations ${rotationsSq.shape[0]}`);
  }

  const scalesM = reshape2(scalesSq.values, N, 3);
  const shapesM = reshape2(shapesSq.values, N, 2);
  const transM = reshape2(translationsSq.values, N, 3);
  const rotM = reshape3(rotationsSq.values, N, 3, 3);

  const out: PrimitiveExport[] = [];
  for (let i = 0; i < N; i++) {
    out.push({
      scales: [scalesM[i][0], scalesM[i][1], scalesM[i][2]],
      shapes: [shapesM[i][0], shapesM[i][1]],
      translation: [transM[i][0], transM[i][1], transM[i][2]],
      rotation: rotM[i],
    });
  }
  return out;
}

export interface ImportNpzOptions {
  /** If true, apply S @ R and S @ t so Z-up pipeline assets sit upright in Y-up Three.js. */
  basisZUpToYUp?: boolean;
}

export async function importNpzToPrimitives(
  blob: Blob,
  namePrefix = 'npz',
  options?: ImportNpzOptions,
): Promise<Primitive[]> {
  const zip = await JSZip.loadAsync(blob);
  const readNpy = async (name: string): Promise<ParsedNpy> => {
    const f = zip.file(name);
    if (!f) throw new Error(`Missing ${name} in archive`);
    const buf = await f.async('arraybuffer');
    return parseNpyBuffer(buf);
  };

  const [scales, shapes, translations, rotations] = await Promise.all([
    readNpy('scales.npy'),
    readNpy('shapes.npy'),
    readNpy('translations.npy'),
    readNpy('rotations.npy'),
  ]);

  let exports = npzArraysToExports(scales, shapes, translations, rotations);
  if (options?.basisZUpToYUp) {
    exports = exports.map(e => {
      const { rotation, translation } = applyWorldBasisZUpToYUp(e.rotation, e.translation);
      return { ...e, rotation, translation };
    });
  }

  const t = Date.now();
  return exports.map((e, i): Primitive => {
    const euler = matrixToEuler(e.rotation);
    return {
      id: `npz_${i}_${t}`,
      name: `${namePrefix}_${i}`,
      visible: true,
      scales: e.scales,
      shapes: e.shapes,
      translation: e.translation,
      rotation: e.rotation,
      eulerDeg: euler,
    };
  });
}
