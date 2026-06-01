/**
 * The WebGL canvas registers a snapshot function here so AI Edit can attach a viewport image.
 */

type CaptureFn = () => string | null;
type RenderExportFn = () => Promise<Blob | null>;

let capture: CaptureFn | null = null;
let renderExport: RenderExportFn | null = null;

export function setViewportCapture(fn: CaptureFn | null): void {
  capture = fn;
}

export function setViewportRenderExport(fn: RenderExportFn | null): void {
  renderExport = fn;
}

/** Latest frame as data URL (image/png), or null if the viewport is not mounted. */
export function captureViewportDataUrl(): string | null {
  try {
    return capture?.() ?? null;
  } catch {
    return null;
  }
}

/** Clean PNG render of the SQ meshes with export colors, or null if the viewport is not mounted. */
export async function captureSuperquadricRenderBlob(): Promise<Blob | null> {
  try {
    return await (renderExport?.() ?? null);
  } catch {
    return null;
  }
}

export function stripDataUrlToBase64(dataUrl: string): string {
  const i = dataUrl.indexOf('base64,');
  if (i >= 0) return dataUrl.slice(i + 7);
  return dataUrl;
}

/** Downscale so the longest side is at most `maxSide` (reduces request size for Ollama). */
export async function downscalePngDataUrlToBase64(
  dataUrl: string,
  maxSide: number
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (w === 0 || h === 0) {
        resolve(stripDataUrlToBase64(dataUrl));
        return;
      }
      const scale = Math.min(1, maxSide / Math.max(w, h));
      const nw = Math.max(1, Math.round(w * scale));
      const nh = Math.max(1, Math.round(h * scale));
      const canvas = document.createElement('canvas');
      canvas.width = nw;
      canvas.height = nh;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(stripDataUrlToBase64(dataUrl));
        return;
      }
      ctx.drawImage(img, 0, 0, nw, nh);
      const out = canvas.toDataURL('image/png');
      resolve(stripDataUrlToBase64(out));
    };
    img.onerror = () => reject(new Error('Could not read viewport image'));
    img.src = dataUrl;
  });
}

/** PNG as raw base64 for Ollama `images: [...]` (no data: prefix). */
export async function captureViewportImageForLlm(maxSide = 768): Promise<string | null> {
  const dataUrl = captureViewportDataUrl();
  if (!dataUrl) return null;
  return downscalePngDataUrlToBase64(dataUrl, maxSide);
}

/** Small `data:image/png;base64,...` for UI thumbnails (Edit panel preview). */
export async function captureViewportPreviewDataUrl(maxSide = 240): Promise<string | null> {
  const dataUrl = captureViewportDataUrl();
  if (!dataUrl) return null;
  const b64 = await downscalePngDataUrlToBase64(dataUrl, maxSide);
  return `data:image/png;base64,${b64}`;
}
