import type { ImportNpzOptions } from '../mesh/npzImport';

export interface NpzUrlRequest {
  source: string;
  namePrefix: string;
  importOptions: ImportNpzOptions;
}

function sanitizeStem(stem: string): string {
  return stem.replace(/[^a-zA-Z0-9_-]/g, '_') || 'npz';
}

function stemFromSource(source: string): string {
  let pathname = source;
  try {
    pathname = new URL(source, window.location.href).pathname;
  } catch {
    pathname = source.split(/[?#]/, 1)[0] ?? source;
  }
  const parts = pathname.split(/[\\/]/).filter(Boolean);
  const filename = parts[parts.length - 1] ?? 'npz';
  if (/^superflex\.npz$/i.test(filename) && parts.length >= 2) {
    return sanitizeStem(parts[parts.length - 2] ?? 'superflex');
  }
  return sanitizeStem(filename.replace(/\.npz$/i, ''));
}

function parseBasis(params: URLSearchParams): ImportNpzOptions['basisZUpToYUp'] {
  const rawBasis = (params.get('basis') ?? params.get('npzBasis') ?? '').toLowerCase();
  const rawZUp = (params.get('zup') ?? params.get('z_up') ?? params.get('basisZUpToYUp') ?? '').toLowerCase();
  const value = rawBasis || rawZUp;
  if (['stored', 'raw', 'none', 'false', '0', 'no'].includes(value)) return false;
  if (['zup', 'z-up', 'zup-to-yup', 'z-up-to-y-up', 'true', '1', 'yes'].includes(value)) return true;
  return false;
}

export function getNpzUrlRequest(location: Location = window.location): NpzUrlRequest | null {
  const params = new URLSearchParams(location.search);
  const source = params.get('npz') ?? params.get('npzPath') ?? params.get('file');
  if (!source) return null;
  const namePrefix = sanitizeStem(params.get('name') ?? stemFromSource(source));
  return {
    source,
    namePrefix,
    importOptions: {
      basisZUpToYUp: parseBasis(params),
      inferSuperflex: true,
    },
  };
}

export function npzFetchUrl(source: string, location: Location = window.location): string {
  try {
    const url = new URL(source, location.href);
    if (url.protocol === 'http:' || url.protocol === 'https:') {
      if (url.origin !== location.origin) return source;
      return `/_sq/npz?path=${encodeURIComponent(`${url.pathname}${url.search}`)}`;
    }
  } catch {
    /* fall through to local dev-server file endpoint */
  }
  return `/_sq/npz?path=${encodeURIComponent(source)}`;
}

export function npzEditorUrl(source: string, location: Location = window.location): string {
  const url = new URL(location.href);
  url.pathname = '/';
  url.search = '';
  url.hash = '';
  url.searchParams.set('npz', source);
  return url.toString();
}
