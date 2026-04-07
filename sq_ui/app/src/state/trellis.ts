import { serviceBaseUrl } from './devServiceUrl';

function trellisBase(): string {
  return serviceBaseUrl(import.meta.env.VITE_TRELLIS_URL, 'http://localhost:11437');
}

export interface TrellisGenerateOptions {
  prompt: string;
  name?: string;
  seed?: number;
  pointCount?: number;
  normalize?: boolean;
  preferMesh?: boolean;
  sparseSteps?: number;
  slatSteps?: number;
  cfgStrength?: number;
  slatCfgStrength?: number;
}

export interface TrellisGenerateResult {
  file: File;
  pointCount: number;
  runId: string;
  metadata?: Record<string, unknown>;
}

interface TrellisGenerateResponse {
  status: 'ok';
  run_id: string;
  point_count: number;
  filename?: string;
  download_url: string;
  metadata?: Record<string, unknown>;
}

function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const base = trellisBase().replace(/\/$/, '');
  const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  return base ? `${base}${path}` : path;
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `TRELLIS returned ${res.status}`;
    try {
      const err = await res.json() as { error?: { message?: string; log_tail?: string } };
      if (err.error?.message) msg = err.error.message;
      if (err.error?.log_tail) msg += `\n${err.error.log_tail.slice(-2000)}`;
    } catch {
      const text = await res.text().catch(() => '');
      if (text) msg += `: ${text.slice(0, 400)}`;
    }
    throw new Error(msg);
  }
  return await res.json() as T;
}

function sanitizeFilename(name: string): string {
  const trimmed = name.trim();
  return trimmed.replace(/[^a-zA-Z0-9_-]/g, '_') || 'trellis_pointcloud';
}

export async function generatePointCloudFromText(
  options: TrellisGenerateOptions
): Promise<TrellisGenerateResult> {
  const body = {
    prompt: options.prompt,
    name: options.name,
    seed: options.seed,
    pointCount: options.pointCount,
    normalize: options.normalize,
    preferMesh: options.preferMesh,
    sparseSteps: options.sparseSteps,
    slatSteps: options.slatSteps,
    cfgStrength: options.cfgStrength,
    slatCfgStrength: options.slatCfgStrength,
  };

  const res = await fetch(resolveUrl('/trellis/generate'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await parseJson<TrellisGenerateResponse>(res);

  const plyRes = await fetch(resolveUrl(payload.download_url));
  if (!plyRes.ok) {
    const text = await plyRes.text().catch(() => '');
    throw new Error(`Could not download TRELLIS output: ${plyRes.status} ${text.slice(0, 200)}`);
  }

  const blob = await plyRes.blob();
  const baseName = sanitizeFilename(options.name || options.prompt || 'trellis_pointcloud');
  const filename = payload.filename || `${baseName}.ply`;
  const file = new File([blob], filename, {
    type: blob.type || 'application/octet-stream',
  });

  return {
    file,
    pointCount: payload.point_count,
    runId: payload.run_id,
    metadata: payload.metadata,
  };
}
