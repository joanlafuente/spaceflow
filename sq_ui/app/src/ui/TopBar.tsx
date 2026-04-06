import { useCallback, useState, useRef, useEffect } from 'react';
import { useStore } from '../state/store';
import type { Primitive } from '../state/store';
import { exportNpz } from '../mesh/npzExport';
import type { PrimitiveExport } from '../mesh/npzExport';
import { importNpzToPrimitives } from '../mesh/npzImport';
import { isOrthogonal } from '../state/rotation';
import { eulerToMatrix, matrixToEuler } from '../state/rotation';
import { editFromText, generateFromText } from '../state/generate';
import { generateWithSuperdec } from '../state/superdec';
import {
  captureViewportDataUrl,
  captureViewportImageForLlm,
  captureViewportPreviewDataUrl,
} from '../state/viewportCapture';

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

let nextPresetId = 0;

export default function TopBar() {
  const primitives = useStore(s => s.primitives);
  const selectedId = useStore(s => s.selectedId);
  const selectPrimitive = useStore(s => s.selectPrimitive);
  const loadPreset = useStore(s => s.loadPreset);
  const undo = useStore(s => s.undo);
  const redo = useStore(s => s.redo);
  const undoStack = useStore(s => s.undoStack);
  const redoStack = useStore(s => s.redoStack);
  const [toast, setToast] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [genPrompt, setGenPrompt] = useState('');
  const [showSuperdec, setShowSuperdec] = useState(false);
  const [superdecGenerating, setSuperdecGenerating] = useState(false);
  const [superdecFile, setSuperdecFile] = useState<File | null>(null);
  const [superdecName, setSuperdecName] = useState('');
  const [superdecZUp, setSuperdecZUp] = useState(false);
  const [superdecNormalize, setSuperdecNormalize] = useState(true);
  const [superdecLmOptimization, setSuperdecLmOptimization] = useState(false);
  const [superdecMaxPrimitives, setSuperdecMaxPrimitives] = useState('16');
  const [superdecExistThreshold, setSuperdecExistThreshold] = useState('0.5');
  const [projectName, setProjectName] = useState('superquadrics');
  const [genMode, setGenMode] = useState<'create' | 'edit'>('create');
  const [editFocusNames, setEditFocusNames] = useState<string[]>([]);
  const [includeViewportInEdit, setIncludeViewportInEdit] = useState(true);
  const [viewportPreviewUrl, setViewportPreviewUrl] = useState<string | null>(null);
  const [viewportPreviewModal, setViewportPreviewModal] = useState(false);
  const [viewportModalUrl, setViewportModalUrl] = useState<string | null>(null);
  const genInputRef = useRef<HTMLInputElement>(null);
  const superdecNameRef = useRef<HTMLInputElement>(null);

  const refreshViewportPreview = useCallback(async () => {
    if (!includeViewportInEdit) {
      setViewportPreviewUrl(null);
      return;
    }
    try {
      const url = await captureViewportPreviewDataUrl(240);
      setViewportPreviewUrl(url);
    } catch {
      setViewportPreviewUrl(null);
    }
  }, [includeViewportInEdit]);

  useEffect(() => {
    if (!showGenerate || genMode !== 'edit' || !includeViewportInEdit) {
      setViewportPreviewUrl(null);
      return;
    }
    void refreshViewportPreview();
  }, [showGenerate, genMode, includeViewportInEdit, primitives, refreshViewportPreview]);

  useEffect(() => {
    if (!showGenerate) {
      setViewportPreviewModal(false);
      setViewportModalUrl(null);
    }
  }, [showGenerate]);

  useEffect(() => {
    if (showSuperdec) {
      setTimeout(() => superdecNameRef.current?.focus(), 50);
    }
  }, [showSuperdec]);

  useEffect(() => {
    if (!viewportPreviewModal) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setViewportPreviewModal(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [viewportPreviewModal]);

  const openViewportPreviewModal = useCallback(() => {
    const full = captureViewportDataUrl();
    setViewportModalUrl(full ?? viewportPreviewUrl);
    setViewportPreviewModal(true);
  }, [viewportPreviewUrl]);

  const allOrtho = primitives.every(p => isOrthogonal(p.rotation));
  const hasWarnings = !allOrtho;

  const showToast = (msg: string, durationMs = 3000) => {
    setToast(msg);
    setTimeout(() => setToast(null), durationMs);
  };

  const toggleEditFocus = useCallback((name: string) => {
    setEditFocusNames(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );
  }, []);

  const addSelectionToEditFocus = useCallback(() => {
    if (!selectedId) return;
    const name = primitives.find(p => p.id === selectedId)?.name;
    if (!name) return;
    setEditFocusNames(prev => (prev.includes(name) ? prev : [...prev, name]));
  }, [selectedId, primitives]);

  const resetSuperdec = useCallback(() => {
    setShowSuperdec(false);
    setSuperdecFile(null);
    setSuperdecName('');
    setSuperdecZUp(false);
    setSuperdecNormalize(true);
    setSuperdecLmOptimization(false);
    setSuperdecMaxPrimitives('16');
    setSuperdecExistThreshold('0.5');
  }, []);

  const handleGenerate = useCallback(async () => {
    const prompt = genPrompt.trim();
    if (!prompt || generating) return;
    if (genMode === 'edit' && primitives.length === 0) {
      showToast('Add or generate primitives before editing.', 4000);
      return;
    }
    setGenerating(true);
    const prevName = primitives.find(p => p.id === selectedId)?.name ?? null;
    try {
      if (genMode === 'create') {
        const prims = await generateFromText(prompt);
        loadPreset(prims);
        setProjectName(prompt.replace(/^a\s+/i, '').trim() || 'superquadrics');
        showToast(`Generated ${prims.length} primitives`);
        setShowGenerate(false);
        setGenPrompt('');
      } else {
        const focus =
          editFocusNames.length > 0 ? editFocusNames : undefined;
        let viewportImages: string[] | undefined;
        if (includeViewportInEdit) {
          try {
            const b64 = await captureViewportImageForLlm(768);
            if (b64) viewportImages = [b64];
          } catch {
            /* fall back to text-only edit */
          }
        }
        const editResult = await editFromText(prompt, primitives, {
          focusNames: focus,
          viewportImagesBase64: viewportImages,
        });
        const vpNote =
          includeViewportInEdit && viewportImages
            ? ' · viewport image sent'
            : includeViewportInEdit && !viewportImages
              ? ' · text only (screenshot unavailable)'
              : '';
        if (editResult.unchanged) {
          showToast(
            `No geometry changed — the model returned the same numbers as your current scene (common with vague prompts like "fix this").${vpNote} Try naming a part and a concrete change, or turn off the viewport screenshot if the model keeps echoing JSON.`,
            10000
          );
        } else {
          const prims = editResult.primitives;
          loadPreset(prims);
          const newSel =
            (prevName && prims.find(p => p.name === prevName)?.id) ??
            prims[0]?.id ??
            null;
          selectPrimitive(newSel);
          showToast(`Updated scene (${prims.length} primitives)${vpNote}`);
          setShowGenerate(false);
          setGenPrompt('');
        }
      }
    } catch (err) {
      showToast(`Generation failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setGenerating(false);
    }
  }, [
    genPrompt,
    generating,
    genMode,
    primitives,
    editFocusNames,
    includeViewportInEdit,
    loadPreset,
    selectPrimitive,
    selectedId,
  ]);

  const handleSuperdecGenerate = useCallback(async () => {
    if (!superdecFile || superdecGenerating) return;
    setSuperdecGenerating(true);
    const baseName =
      superdecName.trim() ||
      superdecFile.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_') ||
      'superdec';
    try {
      const maxPrimitives = Math.max(0, Number.parseInt(superdecMaxPrimitives || '0', 10) || 0);
      const existThreshold = Math.min(
        1,
        Math.max(0, Number.parseFloat(superdecExistThreshold || '0.5') || 0.5)
      );
      const result = await generateWithSuperdec({
        file: superdecFile,
        name: baseName,
        zUp: superdecZUp,
        normalize: superdecNormalize,
        lmOptimization: superdecLmOptimization,
        maxPrimitives,
        existThreshold,
      });
      loadPreset(result.primitives);
      setProjectName(baseName);
      showToast(`Generated ${result.primitiveCount} primitives with SuperDec`);
      resetSuperdec();
    } catch (err) {
      showToast(`SuperDec failed: ${err instanceof Error ? err.message : err}`, 10000);
    } finally {
      setSuperdecGenerating(false);
    }
  }, [
    loadPreset,
    resetSuperdec,
    superdecExistThreshold,
    superdecFile,
    superdecGenerating,
    superdecLmOptimization,
    superdecMaxPrimitives,
    superdecName,
    superdecNormalize,
    superdecZUp,
  ]);

  const handleDownloadNpz = useCallback(async () => {
    try {
      const exports: PrimitiveExport[] = primitives.map(p => ({
        scales: p.scales,
        shapes: p.shapes,
        translation: p.translation,
        rotation: p.rotation,
      }));
      const blob = await exportNpz(exports);
      const filename = `${projectName.replace(/[^a-zA-Z0-9_-]/g, '_') || 'superquadrics'}.npz`;
      downloadBlob(blob, filename);
      showToast(`Downloaded ${filename}`);
    } catch (err) {
      showToast(`Export failed: ${err}`);
    }
    setShowExport(false);
  }, [primitives]);

  const handleCopyJson = useCallback(() => {
    const data = primitives.map(p => ({
      name: p.name,
      scales: p.scales,
      shapes: p.shapes,
      translation: p.translation,
      eulerDeg: p.eulerDeg,
    }));
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    showToast('Copied JSON preset to clipboard');
    setShowExport(false);
  }, [primitives]);

  const handleImportJson = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text) as Array<{
          name?: string;
          scales: [number, number, number];
          shapes: [number, number];
          translation: [number, number, number];
          eulerDeg?: [number, number, number];
          rotation?: number[][];
        }>;
        const prims: Primitive[] = data.map((d) => {
          const euler: [number, number, number] = d.eulerDeg ?? (d.rotation ? matrixToEuler(d.rotation) : [0, 0, 0]);
          const rotation = d.rotation ?? eulerToMatrix(euler);
          return {
            id: `imported_${++nextPresetId}_${Date.now()}`,
            name: d.name ?? `Imported ${nextPresetId}`,
            visible: true,
            scales: d.scales,
            shapes: d.shapes,
            translation: d.translation,
            rotation,
            eulerDeg: euler,
          };
        });
        loadPreset(prims);
        showToast(`Loaded ${prims.length} primitives from JSON`);
      } catch (err) {
        showToast(`Import failed: ${err}`);
      }
    };
    input.click();
    setShowExport(false);
  }, [loadPreset]);

  const handleImportNpz = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.npz,application/octet-stream';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const stem = file.name.replace(/\.npz$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_') || 'npz';
        const prims = await importNpzToPrimitives(file, stem, { basisZUpToYUp: false });
        loadPreset(prims);
        showToast(`Loaded ${prims.length} primitives from ${file.name}`);
      } catch (err) {
        showToast(`NPZ import failed: ${err instanceof Error ? err.message : err}`, 6000);
      }
    };
    input.click();
    setShowExport(false);
  }, [loadPreset]);

  const handleImportNpzZUp = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.npz,application/octet-stream';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const stem = file.name.replace(/\.npz$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_') || 'npz';
        const prims = await importNpzToPrimitives(file, stem, { basisZUpToYUp: true });
        loadPreset(prims);
        showToast(`Loaded ${prims.length} primitives (Z-up → Y-up) from ${file.name}`);
      } catch (err) {
        showToast(`NPZ import failed: ${err instanceof Error ? err.message : err}`, 6000);
      }
    };
    input.click();
    setShowExport(false);
  }, [loadPreset]);

  const handleLoadTemplate = useCallback((template: string) => {
    const templates: Record<string, () => Primitive[]> = {
      'Single Ellipsoid': () => {
        const euler: [number, number, number] = [0, 0, 0];
        return [{
          id: `t_${++nextPresetId}_${Date.now()}`, name: 'Ellipsoid', visible: true,
          scales: [0.5, 0.5, 1], shapes: [1, 1], translation: [0, 0, 0],
          rotation: eulerToMatrix(euler), eulerDeg: euler,
        }];
      },
      'Table (5 parts)': () => {
        const leg = (x: number, z: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`, name: `Leg ${idx}`, visible: true,
          scales: [0.06, 0.4, 0.06], shapes: [0.4, 0.4], translation: [x, -0.44, z],
          rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0],
        });
        return [
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Top', visible: true,
            scales: [0.8, 0.04, 0.5], shapes: [0.3, 0.3], translation: [0, 0, 0],
            rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0] },
          leg(-0.65, -0.4, 1), leg(0.65, -0.4, 2),
          leg(-0.65, 0.4, 3), leg(0.65, 0.4, 4),
        ];
      },
      'Chair (6 parts)': () => {
        const leg = (x: number, z: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`, name: `Leg ${idx}`, visible: true,
          scales: [0.05, 0.35, 0.05], shapes: [0.4, 0.4], translation: [x, -0.39, z],
          rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0],
        });
        return [
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Seat', visible: true,
            scales: [0.5, 0.04, 0.45], shapes: [0.3, 0.3], translation: [0, 0, 0],
            rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0] },
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Backrest', visible: true,
            scales: [0.5, 0.35, 0.04], shapes: [0.3, 0.3], translation: [0, 0.39, -0.4],
            rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0] },
          leg(-0.42, -0.38, 1), leg(0.42, -0.38, 2),
          leg(-0.42, 0.38, 3), leg(0.42, 0.38, 4),
        ];
      },
    };
    const factory = templates[template];
    if (factory) {
      loadPreset(factory());
      showToast(`Loaded "${template}" template`);
    }
    setShowExport(false);
  }, [loadPreset]);

  return (
    <div className="top-bar">
      <div className="top-left">
        <span className="app-name">SQ Editor</span>
        <span className="app-sep">/</span>
        <input
          type="text"
          className="project-name-input"
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          title="Project name (used as export filename)"
        />
      </div>

      <div className="top-center">
        <button className="toolbar-btn" onClick={undo} disabled={undoStack.length === 0} title="Undo (Ctrl+Z)">↩</button>
        <button className="toolbar-btn" onClick={redo} disabled={redoStack.length === 0} title="Redo (Ctrl+Shift+Z)">↪</button>
        <span className="separator" />
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Single Ellipsoid')} title="Single Ellipsoid">⊙</button>
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Table (5 parts)')} title="Table template">⊞</button>
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Chair (6 parts)')} title="Chair template">⊟</button>
        <span className="separator" />
        <div className={`generate-group ${showGenerate ? 'is-open' : ''}`}>
          {!showGenerate ? (
            <button
              className="btn-generate"
              onClick={() => { setShowGenerate(true); setTimeout(() => genInputRef.current?.focus(), 50); }}
              disabled={generating}
              title="Generate from text description (requires Ollama)"
            >
              {generating ? '...' : 'AI Generate'}
            </button>
          ) : (
            <>
              <div
                className="generate-popover-backdrop"
                onClick={() => {
                  if (!generating) {
                    setShowGenerate(false);
                    setGenPrompt('');
                  }
                }}
                aria-hidden
              />
              <div className="generate-open-bar">
                <div className="gen-mode-row" role="group" aria-label="AI mode">
                  <button
                    type="button"
                    className={`gen-mode-btn ${genMode === 'create' ? 'active' : ''}`}
                    onClick={() => setGenMode('create')}
                    disabled={generating}
                  >
                    Create
                  </button>
                  <button
                    type="button"
                    className={`gen-mode-btn ${genMode === 'edit' ? 'active' : ''}`}
                    onClick={() => setGenMode('edit')}
                    disabled={generating}
                    title="Edit current scene with a text instruction"
                  >
                    Edit
                  </button>
                </div>
                <button
                  className="toolbar-btn"
                  onClick={() => { setShowGenerate(false); setGenPrompt(''); }}
                  disabled={generating}
                  title="Close (Esc)"
                  type="button"
                >
                  ✕
                </button>
              </div>
              <div
                className="generate-popover"
                role="dialog"
                aria-label={genMode === 'create' ? 'Create from prompt' : 'Edit scene with prompt'}
                onClick={(e) => e.stopPropagation()}
              >
                {genMode === 'edit' && primitives.length > 0 && (
                  <div className="edit-focus-block">
                    <div className="edit-focus-head">
                      <span className="edit-focus-title">Focus parts</span>
                      <button
                        type="button"
                        className="edit-focus-add-sel"
                        onClick={addSelectionToEditFocus}
                        disabled={!selectedId || generating}
                        title="Add currently selected primitive to focus"
                      >
                        + selection
                      </button>
                    </div>
                    <div className="edit-focus-list">
                      {primitives.map(p => (
                        <label key={p.id} className="edit-focus-item">
                          <input
                            type="checkbox"
                            checked={editFocusNames.includes(p.name)}
                            onChange={() => toggleEditFocus(p.name)}
                            disabled={generating}
                          />
                          <span>{p.name}</span>
                        </label>
                      ))}
                    </div>
                    <p className="edit-focus-hint">
                      Optional: check parts to steer the edit. Leave all unchecked to let the model choose.
                    </p>
                    <label className="edit-viewport-include">
                      <input
                        type="checkbox"
                        checked={includeViewportInEdit}
                        onChange={(e) => setIncludeViewportInEdit(e.target.checked)}
                        disabled={generating}
                      />
                      <span>
                        Include viewport screenshot (multimodal models: aligns the edit with what you see)
                      </span>
                    </label>
                    {includeViewportInEdit && (
                      <div className="edit-viewport-preview-row">
                        <button
                          type="button"
                          className="edit-viewport-thumb"
                          onClick={openViewportPreviewModal}
                          disabled={generating}
                          title="Click to view full size"
                        >
                          {viewportPreviewUrl ? (
                            <img src={viewportPreviewUrl} alt="" />
                          ) : (
                            <span className="edit-viewport-thumb-placeholder">No preview</span>
                          )}
                        </button>
                        <button
                          type="button"
                          className="edit-viewport-refresh"
                          onClick={() => void refreshViewportPreview()}
                          disabled={generating}
                          title="Refresh preview from viewport"
                        >
                          Refresh
                        </button>
                      </div>
                    )}
                  </div>
                )}
                <label className="generate-popover-label" htmlFor="sq-gen-prompt-input">
                  {genMode === 'create' ? 'Describe what to create' : 'Describe what to change'}
                </label>
                <div className="generate-popover-footer">
                  <input
                    id="sq-gen-prompt-input"
                    ref={genInputRef}
                    type="text"
                    className="generate-input generate-input-popover"
                    placeholder={
                      genMode === 'create'
                        ? 'e.g. a wooden desk lamp'
                        : 'e.g. make the backrest taller, rounder wheels'
                    }
                    value={genPrompt}
                    onChange={(e) => setGenPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleGenerate();
                      if (e.key === 'Escape') {
                        setShowGenerate(false);
                        setGenPrompt('');
                      }
                    }}
                    disabled={generating}
                  />
                  <button
                    className="btn-generate-go"
                    type="button"
                    onClick={handleGenerate}
                    disabled={
                      generating ||
                      !genPrompt.trim() ||
                      (genMode === 'edit' && primitives.length === 0)
                    }
                  >
                    {generating ? (
                      <svg
                        className="spinner-svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        aria-hidden={true}
                      >
                        <circle cx="8" cy="8" r="6.5" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2" />
                        <circle
                          cx="8"
                          cy="8"
                          r="6.5"
                          fill="none"
                          stroke="#fff"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeDasharray="10 31"
                        />
                      </svg>
                    ) : (
                      'Go'
                    )}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
        <div className={`generate-group ${showSuperdec ? 'is-open' : ''}`}>
          {!showSuperdec ? (
            <button
              className="btn-generate btn-superdec"
              onClick={() => setShowSuperdec(true)}
              disabled={superdecGenerating}
              title="Generate superquadrics from a point cloud with SuperDec"
            >
              {superdecGenerating ? '...' : 'SuperDec'}
            </button>
          ) : (
            <>
              <div
                className="generate-popover-backdrop"
                onClick={() => {
                  if (!superdecGenerating) resetSuperdec();
                }}
                aria-hidden
              />
              <div className="generate-open-bar">
                <div className="gen-mode-row" role="group" aria-label="SuperDec mode">
                  <span className="superdec-open-label">Point Cloud</span>
                </div>
                <button
                  className="toolbar-btn"
                  onClick={resetSuperdec}
                  disabled={superdecGenerating}
                  title="Close (Esc)"
                  type="button"
                >
                  ✕
                </button>
              </div>
              <div
                className="generate-popover superdec-popover"
                role="dialog"
                aria-label="Generate superquadrics from a point cloud"
                onClick={(e) => e.stopPropagation()}
              >
                <label className="generate-popover-label" htmlFor="sq-superdec-name-input">
                  Scene name
                </label>
                <input
                  id="sq-superdec-name-input"
                  ref={superdecNameRef}
                  type="text"
                  className="generate-input generate-input-popover"
                  placeholder="e.g. chair_scan"
                  value={superdecName}
                  onChange={(e) => setSuperdecName(e.target.value)}
                  disabled={superdecGenerating}
                />

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Input point cloud</span>
                  </div>
                  <label className="superdec-file-picker">
                    <input
                      type="file"
                      accept=".ply,.pcd,.xyz,.xyzn,.xyzrgb,.pts,.obj,.stl"
                      onChange={(e) => setSuperdecFile(e.target.files?.[0] ?? null)}
                      disabled={superdecGenerating}
                    />
                    <span>{superdecFile ? superdecFile.name : 'Choose point cloud or mesh file'}</span>
                  </label>
                  <p className="edit-focus-hint">
                    Supports `.ply` directly and also common formats such as `.pcd`, `.xyz`, `.pts`, `.obj`, and `.stl` via the service-side loader.
                  </p>
                </div>

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Inference options</span>
                  </div>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superdecZUp}
                      onChange={(e) => setSuperdecZUp(e.target.checked)}
                      disabled={superdecGenerating}
                    />
                    <span>Treat input as Z-up and convert it into the editor&apos;s Y-up frame</span>
                  </label>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superdecNormalize}
                      onChange={(e) => setSuperdecNormalize(e.target.checked)}
                      disabled={superdecGenerating}
                    />
                    <span>Normalize point cloud before inference</span>
                  </label>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superdecLmOptimization}
                      onChange={(e) => setSuperdecLmOptimization(e.target.checked)}
                      disabled={superdecGenerating}
                    />
                    <span>Enable LM optimization for a slower but potentially better fit</span>
                  </label>
                  <div className="superdec-number-grid">
                    <label className="superdec-number-field">
                      <span>Max primitives</span>
                      <input
                        type="number"
                        min="0"
                        step="1"
                        className="num-input"
                        value={superdecMaxPrimitives}
                        onChange={(e) => setSuperdecMaxPrimitives(e.target.value)}
                        disabled={superdecGenerating}
                      />
                    </label>
                    <label className="superdec-number-field">
                      <span>Exist threshold</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        className="num-input"
                        value={superdecExistThreshold}
                        onChange={(e) => setSuperdecExistThreshold(e.target.value)}
                        disabled={superdecGenerating}
                      />
                    </label>
                  </div>
                </div>

                <div className="generate-popover-footer">
                  <button
                    className="btn-generate-go"
                    type="button"
                    onClick={handleSuperdecGenerate}
                    disabled={superdecGenerating || !superdecFile}
                  >
                    {superdecGenerating ? (
                      <svg
                        className="spinner-svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        aria-hidden={true}
                      >
                        <circle cx="8" cy="8" r="6.5" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2" />
                        <circle
                          cx="8"
                          cy="8"
                          r="6.5"
                          fill="none"
                          stroke="#fff"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeDasharray="10 31"
                        />
                      </svg>
                    ) : (
                      'Generate'
                    )}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="top-right">
        {hasWarnings && (
          <span className="validation-warn" title="Some rotation matrices are not orthogonal">⚠</span>
        )}
        {!hasWarnings && primitives.length > 0 && (
          <span className="validation-ok" title="All rotations valid">✓</span>
        )}
        <div className="export-dropdown">
          <button
            className="btn-accent"
            onClick={() => setShowExport(!showExport)}
            title="Export, copy, or import presets"
          >
            Export ▾
          </button>
          {showExport && (
            <div className="dropdown-menu">
              <button
                type="button"
                className="dropdown-item"
                onClick={handleDownloadNpz}
                disabled={primitives.length === 0}
              >
                Download .npz
              </button>
              <button
                type="button"
                className="dropdown-item"
                onClick={handleCopyJson}
                disabled={primitives.length === 0}
              >
                Copy JSON preset
              </button>
              <button type="button" className="dropdown-item" onClick={handleImportJson}>
                Import JSON preset
              </button>
              <button type="button" className="dropdown-item" onClick={handleImportNpz}>
                Import .npz (as stored)
              </button>
              <button type="button" className="dropdown-item" onClick={handleImportNpzZUp} title="Use if the object looks sideways: applies x,y,z → x,z,-y">
                Import .npz (Z-up → Y-up)
              </button>
            </div>
          )}
        </div>
      </div>

      {viewportPreviewModal && (
        <div
          className="viewport-preview-modal-backdrop"
          role="presentation"
          onClick={() => setViewportPreviewModal(false)}
        >
          <div
            className="viewport-preview-modal"
            role="dialog"
            aria-label="Viewport screenshot"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className="viewport-preview-modal-close"
              onClick={() => setViewportPreviewModal(false)}
              aria-label="Close"
            >
              ✕
            </button>
            {viewportModalUrl ? (
              <img src={viewportModalUrl} alt="Viewport" className="viewport-preview-modal-img" />
            ) : (
              <p className="viewport-preview-modal-empty">Viewport is not available.</p>
            )}
          </div>
        </div>
      )}

      {toast && <div className="toast" onClick={() => setToast(null)}>{toast}</div>}
    </div>
  );
}
