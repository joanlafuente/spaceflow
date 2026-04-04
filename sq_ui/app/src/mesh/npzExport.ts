/**
 * Minimal NPZ writer — produces numpy-compatible .npz files in the browser.
 * A .npz is a ZIP of .npy files. Each .npy has a small header + raw data.
 */
import JSZip from 'jszip';

function createNpyArray(data: Float64Array, shape: number[]): Uint8Array {
  // NumPy .npy format v1.0
  // Header: \x93NUMPY\x01\x00 + 2-byte LE header_len + header_str (padded to 64-byte alignment)
  const descrStr = "'descr': '<f8'";
  const fortranStr = "'fortran_order': False";
  const shapeStr = `'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  let headerContent = `{${descrStr}, ${fortranStr}, ${shapeStr}, }`;

  // Pad header so (10 + headerLen) is divisible by 64
  const magicLen = 10; // \x93NUMPY\x01\x00 + 2 bytes for length
  let headerLen = headerContent.length + 1; // +1 for newline
  const remainder = (magicLen + headerLen) % 64;
  if (remainder !== 0) {
    headerLen += 64 - remainder;
  }
  headerContent = headerContent.padEnd(headerLen - 1, ' ') + '\n';

  const headerBytes = new TextEncoder().encode(headerContent);
  const totalSize = magicLen + headerBytes.length + data.byteLength;
  const result = new Uint8Array(totalSize);

  // Magic: \x93NUMPY
  result[0] = 0x93;
  result[1] = 0x4e; // N
  result[2] = 0x55; // U
  result[3] = 0x4d; // M
  result[4] = 0x50; // P
  result[5] = 0x59; // Y
  // Version 1.0
  result[6] = 0x01;
  result[7] = 0x00;
  // Header length (little-endian uint16)
  result[8] = headerBytes.length & 0xff;
  result[9] = (headerBytes.length >> 8) & 0xff;
  // Header
  result.set(headerBytes, 10);
  // Data
  result.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), magicLen + headerBytes.length);

  return result;
}

export interface PrimitiveExport {
  scales: [number, number, number];
  shapes: [number, number];
  translation: [number, number, number];
  rotation: number[][]; // 3×3
}

export async function exportNpz(primitives: PrimitiveExport[]): Promise<Blob> {
  const N = primitives.length;
  if (N === 0) throw new Error('No primitives to export');

  const scales = new Float64Array(N * 3);
  const shapes = new Float64Array(N * 2);
  const translations = new Float64Array(N * 3);
  const rotations = new Float64Array(N * 9);

  for (let i = 0; i < N; i++) {
    const p = primitives[i];
    scales[i * 3] = p.scales[0];
    scales[i * 3 + 1] = p.scales[1];
    scales[i * 3 + 2] = p.scales[2];
    shapes[i * 2] = p.shapes[0];
    shapes[i * 2 + 1] = p.shapes[1];
    translations[i * 3] = p.translation[0];
    translations[i * 3 + 1] = p.translation[1];
    translations[i * 3 + 2] = p.translation[2];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        rotations[i * 9 + r * 3 + c] = p.rotation[r][c];
      }
    }
  }

  const zip = new JSZip();
  zip.file('scales.npy', createNpyArray(scales, [N, 3]));
  zip.file('shapes.npy', createNpyArray(shapes, [N, 2]));
  zip.file('translations.npy', createNpyArray(translations, [N, 3]));
  zip.file('rotations.npy', createNpyArray(rotations, [N, 3, 3]));

  return zip.generateAsync({ type: 'blob' });
}
