/**
 * In dev, Vite proxies /superdec, /superflex, /trellis, /api to 127.0.0.1 on the host running Vite.
 * If .env.local sets VITE_* to another host:port (e.g. a guessed cluster IP), the browser
 * loads the UI from one origin and tries to call another — often "Failed to fetch"
 * (firewall, wrong IP, or CORS). Same-origin + proxy avoids that.
 */
export function serviceBaseUrl(
  envValue: string | undefined,
  prodFallback: string,
): string {
  const trimmed = envValue?.trim();
  if (!import.meta.env.DEV) {
    return (trimmed || prodFallback).replace(/\/$/, '');
  }
  if (!trimmed) return '';
  if (typeof window === 'undefined') return trimmed.replace(/\/$/, '');
  try {
    const u = new URL(trimmed.includes('://') ? trimmed : `http://${trimmed}`);
    if (u.origin !== window.location.origin) {
      console.warn(
        `[sq-ui] Ignoring service base URL in dev (not same origin as this page): ${trimmed}\n` +
          '→ using same-origin paths so Vite can proxy. Remove or fix VITE_SUPERDEC_URL / VITE_SUPERFLEX_URL / VITE_TRELLIS_URL / VITE_SPACEFLOW_URL in .env.local.',
      );
      return '';
    }
  } catch {
    return trimmed.replace(/\/$/, '');
  }
  return trimmed.replace(/\/$/, '');
}

export function ollamaChatUrl(prodFallback: string): string {
  const trimmed = import.meta.env.VITE_OLLAMA_URL?.trim();
  if (!import.meta.env.DEV) {
    return trimmed || prodFallback;
  }
  if (!trimmed) return '/api/chat';
  if (typeof window === 'undefined') return trimmed;
  try {
    const u = new URL(trimmed);
    if (u.origin !== window.location.origin) {
      console.warn(
        `[sq-ui] Ignoring VITE_OLLAMA_URL in dev (not same origin as this page): ${trimmed}\n` +
          '→ using /api/chat + Vite proxy. Remove or fix VITE_OLLAMA_URL in .env.local.',
      );
      return '/api/chat';
    }
  } catch {
    /* keep trimmed */
  }
  return trimmed;
}
