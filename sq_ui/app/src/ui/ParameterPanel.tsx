import { useCallback, useMemo } from 'react';
import { useStore } from '../state/store';
import { eulerToMatrix, isOrthogonal, det3 } from '../state/rotation';
import { useTextureUploadStore } from '../state/textureUploads';

/** Half-axis minimum; must match scale slider min for normalized fits. */
const SCALE_MIN = 0.0001;
const SCALE_SLIDER_MAX = 5;
const SCALE_STEP = 0.0001;
const LOG_SLIDER_STEPS = 1000;

interface SliderRowProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  tooltip?: string;
  /** Decimal places for the number box (range uses `step`). Use more for very small normalized fits. */
  inputDecimals?: number;
  rangeMode?: 'linear' | 'log';
}

function SliderRow({ label, value, onChange, min, max, step, tooltip, inputDecimals = 4, rangeMode = 'linear' }: SliderRowProps) {
  const safeMin = rangeMode === 'log' ? Math.max(min, Number.EPSILON) : min;
  const safeMax = Math.max(max, safeMin);
  const clamped = Math.min(safeMax, Math.max(safeMin, value));
  const logMin = Math.log(safeMin);
  const logMax = Math.log(safeMax);
  const logDenom = logMax - logMin || 1;
  // Range inputs require value in [min,max]; clamp only for the thumb, keep number box on the real value.
  const forRange = rangeMode === 'log'
    ? ((Math.log(clamped) - logMin) / logDenom) * LOG_SLIDER_STEPS
    : clamped;
  const numDisplay = Number(value.toFixed(inputDecimals));
  return (
    <div className="slider-row">
      <label className="slider-label" title={tooltip}>
        {label}
        {tooltip && <span className="help-badge" title={tooltip}>?</span>}
      </label>
      <input
        type="range"
        className="slider"
        min={rangeMode === 'log' ? 0 : min}
        max={rangeMode === 'log' ? LOG_SLIDER_STEPS : max}
        step={rangeMode === 'log' ? 1 : step}
        value={forRange}
        onChange={(e) => {
          const raw = parseFloat(e.target.value);
          const next = rangeMode === 'log'
            ? Math.exp(logMin + (raw / LOG_SLIDER_STEPS) * logDenom)
            : raw;
          onChange(next);
        }}
      />
      <input
        type="number"
        className="num-input"
        value={numDisplay}
        step={step}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) onChange(v);
        }}
      />
    </div>
  );
}

function niceScaleMax(value: number): number {
  const candidates = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
  return candidates.find(candidate => candidate >= value) ?? 10;
}

export default function ParameterPanel() {
  const primitives = useStore(s => s.primitives);
  const selectedId = useStore(s => s.selectedId);
  const updatePrimitive = useStore(s => s.updatePrimitive);
  const previewResolution = useStore(s => s.previewResolution);
  const setPreviewResolution = useStore(s => s.setPreviewResolution);
  const showNormalized = useStore(s => s.showNormalized);
  const setShowNormalized = useStore(s => s.setShowNormalized);
  const showControlPreview = useStore(s => s.showControlPreview);
  const setShowControlPreview = useStore(s => s.setShowControlPreview);
  const localTextureImageFile = useTextureUploadStore(s =>
    selectedId ? (s.localTextureImageFiles[selectedId] ?? null) : null
  );
  const setLocalTextureImageFile = useTextureUploadStore(s => s.setLocalTextureImageFile);
  const clearLocalTextureImageFile = useTextureUploadStore(s => s.clearLocalTextureImageFile);

  const prim = primitives.find(p => p.id === selectedId);

  const updateScales = useCallback((idx: number, val: number) => {
    if (!prim) return;
    const s: [number, number, number] = [...prim.scales];
    s[idx] = Math.max(SCALE_MIN, val);
    updatePrimitive(prim.id, { scales: s });
  }, [prim, updatePrimitive]);

  const updateShapes = useCallback((idx: number, val: number) => {
    if (!prim) return;
    const s: [number, number, number] = [...prim.shapes, 0] as [number, number, number];
    const shapes: [number, number] = [s[0], s[1]];
    shapes[idx] = Math.max(0.05, val);
    updatePrimitive(prim.id, { shapes });
  }, [prim, updatePrimitive]);

  const updateTranslation = useCallback((idx: number, val: number) => {
    if (!prim) return;
    const t: [number, number, number] = [...prim.translation];
    t[idx] = val;
    updatePrimitive(prim.id, { translation: t });
  }, [prim, updatePrimitive]);

  const updateEuler = useCallback((idx: number, val: number) => {
    if (!prim) return;
    const e: [number, number, number] = [...prim.eulerDeg];
    e[idx] = val;
    updatePrimitive(prim.id, { eulerDeg: e });
  }, [prim, updatePrimitive]);

  const updateName = useCallback((name: string) => {
    if (!prim) return;
    updatePrimitive(prim.id, { name });
  }, [prim, updatePrimitive]);

  const updateControlLevel = useCallback((controlLevel: 'high' | 'low') => {
    if (!prim) return;
    updatePrimitive(prim.id, { controlLevel });
  }, [prim, updatePrimitive]);

  const updateLocalTextureText = useCallback((value: string) => {
    if (!prim) return;
    updatePrimitive(prim.id, { localTextureText: value });
  }, [prim, updatePrimitive]);

  const updateLocalTextureImagePath = useCallback((value: string) => {
    if (!prim) return;
    updatePrimitive(prim.id, { localTextureImagePath: value });
  }, [prim, updatePrimitive]);

  const clearLocalTextureOverride = useCallback(() => {
    if (!prim) return;
    updatePrimitive(prim.id, { localTextureText: '', localTextureImagePath: '' });
    clearLocalTextureImageFile(prim.id);
  }, [clearLocalTextureImageFile, prim, updatePrimitive]);

  const scaleSliderMax = useMemo(() => {
    const maxScale = Math.max(...primitives.flatMap(p => p.scales), 0.05);
    return Math.min(SCALE_SLIDER_MAX, niceScaleMax(maxScale * 1.5));
  }, [primitives]);

  if (!prim) {
    return (
      <div className="panel right-panel">
        <div className="panel-header">
          <span className="panel-title">Properties</span>
        </div>
        <div className="empty-state">
          <div className="empty-text">Select a primitive</div>
          <div className="empty-sub">Click in the list or viewport</div>
        </div>
        <div className="section" style={{ marginTop: 'auto' }}>
          <div className="section-title">Preview</div>
          <SliderRow
            label="Resolution"
            value={previewResolution}
            onChange={setPreviewResolution}
            min={16} max={128} step={4}
            tooltip="Mesh quality for preview only (does not affect export)"
          />
          <div className="checkbox-row">
            <label>
              <input
                type="checkbox"
                checked={showNormalized}
                onChange={(e) => setShowNormalized(e.target.checked)}
              />
              <span>Show pipeline-normalized view</span>
            </label>
          </div>
          <div className="checkbox-row">
            <label>
              <input
                type="checkbox"
                checked={showControlPreview}
                onChange={(e) => setShowControlPreview(e.target.checked)}
              />
              <span>Show control preview</span>
            </label>
          </div>
        </div>
      </div>
    );
  }

  const ortho = isOrthogonal(prim.rotation);
  const determinant = det3(prim.rotation);

  return (
    <div className="panel right-panel">
      <div className="panel-header">
        <span className="panel-title">Properties</span>
      </div>

      <div className="section">
        <div className="section-title">Name</div>
        <input
          type="text"
          className="name-input"
          value={prim.name}
          onChange={(e) => updateName(e.target.value)}
        />
      </div>

      <div className="section">
        <div className="section-title">
          Control
          <span
            className="help-badge"
            title="High-control primitives are preserved more strongly. Low-control primitives define the area where SpaceFlow is freer to imagine."
          >
            ?
          </span>
        </div>
        <div className="control-level-toggle" role="group" aria-label="Control level">
          <button
            type="button"
            className={`control-level-btn high ${prim.controlLevel === 'high' ? 'active' : ''}`}
            onClick={() => updateControlLevel('high')}
          >
            High
          </button>
          <button
            type="button"
            className={`control-level-btn low ${prim.controlLevel === 'low' ? 'active' : ''}`}
            onClick={() => updateControlLevel('low')}
          >
            Low
          </button>
        </div>
      </div>

      <div className="section local-texture-section">
        <div className="section-title">
          Local texture
          <span
            className="help-badge"
            title="Overrides the global SpaceFlow texture condition for this selected superquadric. Empty fields use the global texture."
          >
            ?
          </span>
        </div>
        <label className="local-texture-field">
          <span>Text override</span>
          <input
            type="text"
            className="name-input local-texture-input"
            value={prim.localTextureText ?? ''}
            onChange={(e) => updateLocalTextureText(e.target.value)}
            placeholder="Empty uses global text"
          />
        </label>
        <label className="local-texture-field">
          <span>Image path override</span>
          <input
            type="text"
            className="name-input local-texture-input"
            value={prim.localTextureImagePath ?? ''}
            onChange={(e) => updateLocalTextureImagePath(e.target.value)}
            placeholder="Empty uses global image"
          />
        </label>
        <div className="local-texture-actions">
          <label className="local-texture-file-picker">
            <input
              type="file"
              accept="image/*"
              onClick={(e) => {
                e.currentTarget.value = '';
              }}
              onChange={(e) => setLocalTextureImageFile(prim.id, e.target.files?.[0] ?? null)}
            />
            <span title={localTextureImageFile?.name ?? undefined}>
              {localTextureImageFile?.name ?? 'Choose image override'}
            </span>
          </label>
          {(prim.localTextureText || prim.localTextureImagePath || localTextureImageFile) && (
            <button
              type="button"
              className="local-texture-clear-btn"
              onClick={clearLocalTextureOverride}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title">
          Scales (A, B, C)
          <span
            className="help-badge"
            title="Half-axes (not diameter). The slider uses a logarithmic, scene-aware range so small fitted parts remain editable; use the number box for exact values."
          >
            ?
          </span>
        </div>
        {(['A', 'B', 'C'] as const).map((label, i) => (
          <SliderRow
            key={label}
            label={label}
            value={prim.scales[i]}
            onChange={(v) => updateScales(i, v)}
            min={SCALE_MIN}
            max={scaleSliderMax}
            step={SCALE_STEP}
            inputDecimals={6}
            rangeMode="log"
          />
        ))}
      </div>

      <div className="section">
        <div className="section-title">
          Shape (e₁, e₂)
          <span className="help-badge" title="Presets: ball [1,1]; ellipsoid scales [0.5,0.5,1] + [1,1]; cylinder [0.05,1] + euler 90°X; cube [0.05,0.05] + 90°X; astroid [4,4].">?</span>
        </div>
        <SliderRow
          label="e₁"
          value={prim.shapes[0]}
          onChange={(v) => updateShapes(0, v)}
          min={0.05} max={4} step={0.001}
          tooltip="e₁: use ~1 for ball; ~0.05 with e₂=1 and 90°X for upright cylinder"
          inputDecimals={6}
          rangeMode="log"
        />
        <SliderRow
          label="e₂"
          value={prim.shapes[1]}
          onChange={(v) => updateShapes(1, v)}
          min={0.05} max={4} step={0.001}
          tooltip="e₂: use ~1 for ball/cylinder cross-section; ~0.05 with e₁=0.05 and 90°X for cube"
          inputDecimals={6}
          rangeMode="log"
        />
        <div className="shape-presets">
          <button
            className="preset-chip"
            type="button"
            onClick={() =>
              updatePrimitive(prim.id, {
                shapes: [1, 1],
                eulerDeg: [0, 0, 0],
                rotation: eulerToMatrix([0, 0, 0]),
              })
            }
          >
            Ball
          </button>
          <button
            className="preset-chip"
            type="button"
            onClick={() =>
              updatePrimitive(prim.id, {
                scales: [0.5, 0.5, 1],
                shapes: [1, 1],
                eulerDeg: [0, 0, 0],
                rotation: eulerToMatrix([0, 0, 0]),
              })
            }
          >
            Ellipsoid
          </button>
          <button
            className="preset-chip"
            type="button"
            onClick={() =>
              updatePrimitive(prim.id, {
                shapes: [0.05, 1],
                eulerDeg: [90, 0, 0],
                rotation: eulerToMatrix([90, 0, 0]),
              })
            }
          >
            Cylinder
          </button>
          <button
            className="preset-chip"
            type="button"
            onClick={() =>
              updatePrimitive(prim.id, {
                shapes: [0.05, 0.05],
                eulerDeg: [90, 0, 0],
                rotation: eulerToMatrix([90, 0, 0]),
              })
            }
          >
            Cube
          </button>
          <button
            className="preset-chip"
            type="button"
            onClick={() =>
              updatePrimitive(prim.id, {
                shapes: [4, 4],
                eulerDeg: [0, 0, 0],
                rotation: eulerToMatrix([0, 0, 0]),
              })
            }
          >
            Astroid
          </button>
        </div>
      </div>

      <div className="section">
        <div className="section-title">Translation (x, y, z)</div>
        {(['x', 'y', 'z'] as const).map((label, i) => (
          <SliderRow
            key={label}
            label={label}
            value={prim.translation[i]}
            onChange={(v) => updateTranslation(i, v)}
            min={-5} max={5} step={0.01}
          />
        ))}
      </div>

      <div className="section">
        <div className="section-title">
          Rotation (Euler ZYX, degrees)
          <span className="help-badge" title="Euler angles in ZYX intrinsic order. Internally stored as 3×3 rotation matrix.">?</span>
        </div>
        {(['Rx', 'Ry', 'Rz'] as const).map((label, i) => (
          <SliderRow
            key={label}
            label={label}
            value={prim.eulerDeg[i]}
            onChange={(v) => updateEuler(i, v)}
            min={-180} max={180} step={0.5}
          />
        ))}
        {!ortho && (
          <div className="warning-badge">
            ⚠ Rotation matrix is not orthogonal
          </div>
        )}
        {Math.abs(determinant - 1) > 0.01 && Math.abs(determinant + 1) > 0.01 && (
          <div className="warning-badge">
            ⚠ det(R) = {determinant.toFixed(3)} (expected ±1)
          </div>
        )}
        {determinant < 0 && Math.abs(determinant + 1) < 0.01 && (
          <div className="info-badge">
            ℹ Improper rotation (det = -1): u-parameter will be reversed in mesh
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">Preview</div>
        <SliderRow
          label="Resolution"
          value={previewResolution}
          onChange={setPreviewResolution}
          min={16} max={128} step={4}
          tooltip="Mesh quality for preview only (does not affect export)"
        />
        <div className="checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={showNormalized}
              onChange={(e) => setShowNormalized(e.target.checked)}
            />
            <span>Show pipeline-normalized view</span>
          </label>
        </div>
        <div className="checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={showControlPreview}
              onChange={(e) => setShowControlPreview(e.target.checked)}
            />
            <span>Show control preview</span>
          </label>
        </div>
      </div>
    </div>
  );
}
