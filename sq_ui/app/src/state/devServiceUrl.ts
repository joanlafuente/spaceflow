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
          'using same-origin paths so Vite can proxy. Remove or fix VITE_SPACEFLOW_URL in .env.local.',
      );
      return '';
    }
  } catch {
    return trimmed.replace(/\/$/, '');
  }
  return trimmed.replace(/\/$/, '');
}
