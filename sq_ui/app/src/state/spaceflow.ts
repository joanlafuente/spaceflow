import { importNpzToPrimitives } from '../mesh/npzImport';
import {
  buildSpaceflowSqBundleBlobs,
  buildSpaceflowSqBundleData,
  type SpaceflowSqBundleData,
} from '../mesh/spaceflowExport';
import type { Primitive } from './store';
import {
  clampLowControlBBoxMargin,
  DEFAULT_LOW_CONTROL_BBOX_MARGIN,
} from './spaceflowConfig';
import { serviceBaseUrl } from './devServiceUrl';

function spaceflowBase(): string {
  return serviceBaseUrl(import.meta.env.VITE_SPACEFLOW_URL, 'http://localhost:11438');
}

function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const base = spaceflowBase().replace(/\/$/, '');
  const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  return base ? `${base}${path}` : path;
}

export function resolveSpaceflowUrl(pathOrUrl: string): string {
  return resolveUrl(pathOrUrl);
}

export interface SpaceflowHistoryEntry {
  id: string;
  project_name: string;
  saved_at: string;
  asset_dir: string;
  manifest_path: string;
  paths: {
    all: string;
    high_control: string;
    low_control_bbox: string;
  };
  counts?: {
    all?: number;
    high?: number;
    low?: number;
  };
}

export interface SpaceflowRunConfig {
  textPrompt: string;
  appearanceMode: 'text' | 'image';
  appearanceText?: string;
  appearanceImagePath?: string;
  lowTau: number;
  highTau: number;
  polyakTau: number;
  outputName: string;
  convertYupToZup: boolean;
  lowControlBBoxMargin: number;
  dryRun?: boolean;
}

export interface SpaceflowOutputFile {
  name: string;
  path: string;
  relative_path: string;
  kind: 'mesh' | 'data' | 'image' | 'video' | 'log' | 'file' | string;
  size: number;
  mtime: number;
  url: string;
}

export interface SpaceflowRunStatus {
  run_id: string;
  status: 'running' | 'succeeded' | 'failed' | 'dry_run' | string;
  project_name?: string;
  output_dir?: string;
  log_path?: string;
  command?: string[];
  returncode?: number;
  pipeline_stage?: string;
  output_files?: SpaceflowOutputFile[];
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `SpaceFlow service returned ${res.status}`;
    try {
      const payload = await res.json() as { error?: { message?: string; log_tail?: string } };
      if (payload.error?.message) message = payload.error.message;
      if (payload.error?.log_tail) message += `\n${payload.error.log_tail.slice(-2000)}`;
    } catch {
      const text = await res.text().catch(() => '');
      if (text) message += `: ${text.slice(0, 400)}`;
    }
    throw new Error(message);
  }
  return await res.json() as T;
}

function manifestFor(
  projectName: string,
  primitives: Primitive[],
  bundle: SpaceflowSqBundleData,
  options: { lowTau: number; highTau: number; lowControlBBoxMargin: number },
) {
  return {
    project_name: projectName,
    counts: bundle.counts,
    low_tau: options.lowTau,
    high_tau: options.highTau,
    bbox_margin_fraction: options.lowControlBBoxMargin,
    bbox: bundle.bbox,
    primitives: primitives.map((p, index) => ({
      index,
      name: p.name,
      controlLevel: p.controlLevel,
      visible: p.visible,
    })),
  };
}

async function buildBundleForm(
  projectName: string,
  primitives: Primitive[],
  options: { lowTau: number; highTau: number; lowControlBBoxMargin?: number },
): Promise<{ form: FormData; bundle: SpaceflowSqBundleData }> {
  const lowControlBBoxMargin = clampLowControlBBoxMargin(
    options.lowControlBBoxMargin ?? DEFAULT_LOW_CONTROL_BBOX_MARGIN,
  );
  const bundle = buildSpaceflowSqBundleData(primitives, { lowControlBBoxMargin });
  const blobs = await buildSpaceflowSqBundleBlobs(bundle);
  const form = new FormData();
  form.append('projectName', projectName);
  form.append('manifest', JSON.stringify(manifestFor(projectName, primitives, bundle, {
    lowTau: options.lowTau,
    highTau: options.highTau,
    lowControlBBoxMargin,
  })));
  form.append('all', blobs.all, 'all.npz');
  form.append('high_control', blobs.highControl, 'high_control.npz');
  form.append('low_control_bbox', blobs.lowControlBbox, 'low_control_bbox.npz');
  return { form, bundle };
}

export async function saveSpaceflowAsset(options: {
  projectName: string;
  primitives: Primitive[];
  lowTau?: number;
  highTau?: number;
  lowControlBBoxMargin?: number;
}): Promise<{ entry: SpaceflowHistoryEntry; bundle: SpaceflowSqBundleData }> {
  const { form, bundle } = await buildBundleForm(options.projectName, options.primitives, {
    lowTau: options.lowTau ?? 3,
    highTau: options.highTau ?? 10,
    lowControlBBoxMargin: options.lowControlBBoxMargin,
  });
  const res = await fetch(resolveUrl('/spaceflow/assets/save'), {
    method: 'POST',
    body: form,
  });
  const payload = await parseJson<{ status: 'ok'; entry: SpaceflowHistoryEntry }>(res);
  return { entry: payload.entry, bundle };
}

export async function fetchSpaceflowHistory(limit = 50): Promise<SpaceflowHistoryEntry[]> {
  const res = await fetch(resolveUrl(`/spaceflow/assets/history?limit=${encodeURIComponent(String(limit))}`));
  const payload = await parseJson<{ status: 'ok'; entries: SpaceflowHistoryEntry[] }>(res);
  return payload.entries;
}

export async function openSpaceflowAsset(entry: SpaceflowHistoryEntry): Promise<Primitive[]> {
  const res = await fetch(resolveUrl(`/spaceflow/assets/open?path=${encodeURIComponent(entry.paths.all)}`));
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Could not open saved NPZ: ${res.status} ${text.slice(0, 200)}`);
  }
  const blob = await res.blob();
  return importNpzToPrimitives(blob, entry.project_name || 'spaceflow_asset', {
    inferSuperflex: true,
    basisZUpToYUp: false,
  });
}

export async function startSpaceflowRun(options: {
  projectName: string;
  primitives: Primitive[];
  runConfig: SpaceflowRunConfig;
  appearanceImageFile?: File | null;
}): Promise<{ run: SpaceflowRunStatus; bundle: SpaceflowSqBundleData }> {
  const { form, bundle } = await buildBundleForm(options.projectName, options.primitives, {
    lowTau: options.runConfig.lowTau,
    highTau: options.runConfig.highTau,
    lowControlBBoxMargin: options.runConfig.lowControlBBoxMargin,
  });
  if (bundle.counts.high === 0) {
    throw new Error('Mark at least one primitive as high control before running SpaceFlow.');
  }
  if (bundle.counts.low === 0) {
    throw new Error('Mark at least one primitive as low control before running SpaceFlow.');
  }
  form.append('runConfig', JSON.stringify(options.runConfig));
  if (options.appearanceImageFile) {
    form.append('appearance_image', options.appearanceImageFile, options.appearanceImageFile.name);
  }
  const res = await fetch(resolveUrl('/spaceflow/runs/start'), {
    method: 'POST',
    body: form,
  });
  const payload = await parseJson<{ status: 'ok'; run_id: string; run: SpaceflowRunStatus }>(res);
  return { run: payload.run, bundle };
}

export async function getSpaceflowRunStatus(runId: string): Promise<{
  run: SpaceflowRunStatus;
  logTail: string;
}> {
  const res = await fetch(resolveUrl(`/spaceflow/runs/status?run_id=${encodeURIComponent(runId)}`));
  const payload = await parseJson<{ status: 'ok'; run: SpaceflowRunStatus; log_tail: string }>(res);
  return { run: payload.run, logTail: payload.log_tail };
}
