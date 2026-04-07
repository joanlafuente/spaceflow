import { importNpzToPrimitives } from '../mesh/npzImport';
import type { Primitive } from './store';
import { serviceBaseUrl } from './devServiceUrl';

function superdecBase(): string {
  return serviceBaseUrl(import.meta.env.VITE_SUPERDEC_URL, 'http://localhost:11435');
}

export interface SuperdecGenerateOptions {
  file: File;
  name?: string;
  zUp?: boolean;
  normalize?: boolean;
  lmOptimization?: boolean;
  maxPrimitives?: number;
  existThreshold?: number;
}

export interface SuperdecGenerateResult {
  primitives: Primitive[];
  names: string[];
  primitiveCount: number;
  runId: string;
  metadata?: Record<string, unknown>;
}

interface SuperdecGenerateResponse {
  status: 'ok';
  run_id: string;
  primitive_count: number;
  names?: string[];
  download_url: string;
  metadata?: Record<string, unknown>;
}

function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const base = superdecBase().replace(/\/$/, '');
  const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  return base ? `${base}${path}` : path;
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `SuperDec returned ${res.status}`;
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

export async function generateWithSuperdec(
  options: SuperdecGenerateOptions
): Promise<SuperdecGenerateResult> {
  const form = new FormData();
  form.append('file', options.file);
  if (options.name) form.append('name', options.name);
  form.append('zUp', String(!!options.zUp));
  form.append('normalize', String(options.normalize ?? true));
  form.append('lmOptimization', String(!!options.lmOptimization));
  if (typeof options.maxPrimitives === 'number' && Number.isFinite(options.maxPrimitives)) {
    form.append('maxPrimitives', String(Math.max(0, Math.floor(options.maxPrimitives))));
  }
  if (typeof options.existThreshold === 'number' && Number.isFinite(options.existThreshold)) {
    form.append('existThreshold', String(options.existThreshold));
  }

  const res = await fetch(resolveUrl('/superdec/generate'), {
    method: 'POST',
    body: form,
  });
  const payload = await parseJson<SuperdecGenerateResponse>(res);

  const npzRes = await fetch(resolveUrl(payload.download_url));
  if (!npzRes.ok) {
    const text = await npzRes.text().catch(() => '');
    throw new Error(`Could not download SuperDec output: ${npzRes.status} ${text.slice(0, 200)}`);
  }
  const blob = await npzRes.blob();
  const primitives = await importNpzToPrimitives(blob, options.name || options.file.name.replace(/\.[^.]+$/, '') || 'superdec');

  const names = payload.names ?? [];
  const renamed = primitives.map((primitive, index) => ({
    ...primitive,
    name: names[index] ?? primitive.name,
  }));

  return {
    primitives: renamed,
    names,
    primitiveCount: payload.primitive_count,
    runId: payload.run_id,
    metadata: payload.metadata,
  };
}
