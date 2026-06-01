import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const appDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(appDir, '../..');

function allowedRoots(): string[] {
  const configured = process.env.SQ_UI_NPZ_ROOTS;
  if (!configured) return [projectRoot];
  return configured
    .split(path.delimiter)
    .map(root => path.resolve(root))
    .filter(Boolean);
}

function isInsideRoot(filePath: string, root: string): boolean {
  const rel = path.relative(root, filePath);
  return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel));
}

function decodeRequestedPath(raw: string): string {
  if (raw.startsWith('file://')) {
    return fileURLToPath(raw);
  }
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function pathCandidates(raw: string): string[] {
  const requested = decodeRequestedPath(raw.split('?', 1)[0] ?? raw);
  const candidates: string[] = [];
  if (path.isAbsolute(requested)) {
    candidates.push(path.resolve(requested));
    candidates.push(path.resolve(projectRoot, requested.replace(/^\/+/, '')));
  } else {
    candidates.push(path.resolve(projectRoot, requested));
  }
  return [...new Set(candidates)];
}

function resolveNpzPath(raw: string): { filePath?: string; status?: number; message?: string } {
  const roots = allowedRoots();
  const candidates = pathCandidates(raw);
  const filePath = candidates.find(candidate => fs.existsSync(candidate)) ?? candidates[0];
  if (!filePath) return { status: 400, message: 'Missing NPZ path' };
  if (path.extname(filePath).toLowerCase() !== '.npz') {
    return { status: 400, message: 'Only .npz files can be opened by the editor' };
  }
  if (!roots.some(root => isInsideRoot(filePath, root))) {
    return { status: 403, message: `NPZ path is outside SQ_UI_NPZ_ROOTS (${roots.join(path.delimiter)})` };
  }
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    return { status: 404, message: `NPZ file not found: ${filePath}` };
  }
  return { filePath };
}

function sendText(res: import('node:http').ServerResponse, status: number, message: string): void {
  res.statusCode = status;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end(message);
}

export function npzOpenPlugin(): Plugin {
  return {
    name: 'sq-ui-npz-open',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || (req.method !== 'GET' && req.method !== 'HEAD')) {
          next();
          return;
        }

        const url = new URL(req.url, 'http://sq-ui.local');
        if (url.pathname === '/_sq/npz') {
          const rawPath = url.searchParams.get('path');
          if (!rawPath) {
            sendText(res, 400, 'Missing path query parameter');
            return;
          }
          const resolved = resolveNpzPath(rawPath);
          if (!resolved.filePath) {
            sendText(res, resolved.status ?? 500, resolved.message ?? 'Could not resolve NPZ path');
            return;
          }
          const stat = fs.statSync(resolved.filePath);
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Length', String(stat.size));
          res.setHeader('Content-Disposition', `inline; filename="${path.basename(resolved.filePath)}"`);
          res.setHeader('Cache-Control', 'no-store');
          if (req.method === 'HEAD') {
            res.end();
            return;
          }
          fs.createReadStream(resolved.filePath).pipe(res);
          return;
        }

        if (url.pathname.toLowerCase().endsWith('.npz')) {
          const resolved = resolveNpzPath(url.pathname);
          if (!resolved.filePath) {
            next();
            return;
          }
          res.statusCode = 302;
          res.setHeader('Location', `/?npz=${encodeURIComponent(resolved.filePath)}`);
          res.end();
          return;
        }

        next();
      });
    },
  };
}
