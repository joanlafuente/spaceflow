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
