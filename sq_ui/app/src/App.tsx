import { useEffect, useRef, useState } from 'react';
import TopBar from './ui/TopBar';
import PrimitiveList from './ui/PrimitiveList';
import Viewport from './ui/Viewport';
import ParameterPanel from './ui/ParameterPanel';
import { useStore } from './state/store';
import { importNpzToPrimitives } from './mesh/npzImport';
import { getNpzUrlRequest, npzFetchUrl } from './state/npzUrl';
import './App.css';

export default function App() {
  const undo = useStore(s => s.undo);
  const redo = useStore(s => s.redo);
  const selectedId = useStore(s => s.selectedId);
  const removePrimitive = useStore(s => s.removePrimitive);
  const duplicatePrimitive = useStore(s => s.duplicatePrimitive);
  const loadPreset = useStore(s => s.loadPreset);
  const [urlToast, setUrlToast] = useState<string | null>(null);
  const loadedNpzRequestRef = useRef<string | null>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;

      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'Z') {
        e.preventDefault();
        redo();
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        e.preventDefault();
        removePrimitive(selectedId);
      } else if (e.key === 'd' && selectedId && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        duplicatePrimitive(selectedId);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo, selectedId, removePrimitive, duplicatePrimitive]);

  useEffect(() => {
    const request = getNpzUrlRequest();
    const requestKey = request
      ? JSON.stringify({ source: request.source, importOptions: request.importOptions })
      : null;
    if (!request || loadedNpzRequestRef.current === requestKey) return;

    let cancelled = false;
    const loadFromUrl = async () => {
      setUrlToast(`Loading ${request.namePrefix}.npz...`);
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 12000);
      let blob: Blob;
      try {
        const res = await fetch(npzFetchUrl(request.source), { signal: controller.signal });
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new Error(`${res.status} ${res.statusText}${text ? `: ${text.slice(0, 160)}` : ''}`);
        }
        blob = await res.blob();
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          throw new Error('Timed out fetching NPZ. Check that the Vite port in the URL is still running.');
        }
        throw err;
      } finally {
        window.clearTimeout(timeout);
      }
      const prims = await importNpzToPrimitives(blob, request.namePrefix, request.importOptions);
      if (cancelled) return;
      loadPreset(prims);
      loadedNpzRequestRef.current = requestKey;
      setUrlToast(`Loaded ${prims.length} primitives from ${request.namePrefix}.npz`);
      window.setTimeout(() => {
        if (!cancelled) setUrlToast(null);
      }, 3500);
    };

    loadFromUrl().catch((err) => {
      if (cancelled) return;
      setUrlToast(`NPZ URL load failed: ${err instanceof Error ? err.message : err}`);
    });

    return () => {
      cancelled = true;
    };
  }, [loadPreset]);

  return (
    <div className="app">
      <TopBar />
      <div className="workspace">
        <PrimitiveList />
        <Viewport />
        <ParameterPanel />
      </div>
      {urlToast && <div className="toast" onClick={() => setUrlToast(null)}>{urlToast}</div>}
    </div>
  );
}
