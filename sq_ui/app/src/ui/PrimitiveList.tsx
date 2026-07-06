import { useCallback, useState, useRef } from 'react';
import { useStore, PRESETS } from '../state/store';
import {
  LOW_CONTROL_BBOX_MARGIN_MAX,
  LOW_CONTROL_BBOX_MARGIN_MIN,
  LOW_CONTROL_BBOX_MARGIN_STEP,
} from '../state/spaceflowConfig';
import {
  CopyIcon,
  DiamondIcon,
  EyeIcon,
  EyeOffIcon,
  GripIcon,
  PlusIcon,
  TrashIcon,
} from './icons';

const PRIM_COLORS = [
  '#4fc3f7', '#81c784', '#ffb74d', '#e57373',
  '#ba68c8', '#4dd0e1', '#aed581', '#ffd54f',
  '#f06292', '#7986cb', '#a1887f', '#90a4ae',
];

export default function PrimitiveList() {
  const primitives = useStore(s => s.primitives);
  const selectedId = useStore(s => s.selectedId);
  const addPrimitive = useStore(s => s.addPrimitive);
  const removePrimitive = useStore(s => s.removePrimitive);
  const duplicatePrimitive = useStore(s => s.duplicatePrimitive);
  const selectPrimitive = useStore(s => s.selectPrimitive);
  const updatePrimitive = useStore(s => s.updatePrimitive);
  const reorderPrimitives = useStore(s => s.reorderPrimitives);
  const lowControlBBoxMargin = useStore(s => s.lowControlBBoxMargin);
  const setLowControlBBoxMargin = useStore(s => s.setLowControlBBoxMargin);
  const lowControlBBoxMarginPercent = Math.round(lowControlBBoxMargin * 100);

  const [showPresets, setShowPresets] = useState(false);
  const dragItem = useRef<number | null>(null);
  const dragOverItem = useRef<number | null>(null);

  const handleDragStart = (idx: number) => { dragItem.current = idx; };
  const handleDragEnter = (idx: number) => { dragOverItem.current = idx; };
  const handleDragEnd = () => {
    if (dragItem.current !== null && dragOverItem.current !== null && dragItem.current !== dragOverItem.current) {
      reorderPrimitives(dragItem.current, dragOverItem.current);
    }
    dragItem.current = null;
    dragOverItem.current = null;
  };

  const handleAdd = useCallback((presetKey?: string) => {
    if (presetKey && PRESETS[presetKey]) {
      addPrimitive(PRESETS[presetKey]());
    } else {
      addPrimitive();
    }
    setShowPresets(false);
  }, [addPrimitive]);

  return (
    <div className="panel left-panel">
      <div className="panel-header">
        <span className="panel-title">Scene</span>
        <span className="prim-count">{primitives.length} primitive{primitives.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="prim-list">
        {primitives.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon"><DiamondIcon size={38} strokeWidth={1.6} /></div>
            <div className="empty-text">No primitives yet</div>
            <div className="empty-sub">Add your first superquadric</div>
          </div>
        )}

        {primitives.map((p, i) => (
          <div
            key={p.id}
            className={`prim-row ${p.id === selectedId ? 'selected' : ''}`}
            onClick={() => selectPrimitive(p.id)}
            draggable
            onDragStart={() => handleDragStart(i)}
            onDragEnter={() => handleDragEnter(i)}
            onDragEnd={handleDragEnd}
            onDragOver={(e) => e.preventDefault()}
          >
            <span className="drag-handle" title="Drag to reorder"><GripIcon size={15} /></span>
            <span
              className="prim-color-dot"
              style={{ background: PRIM_COLORS[i % PRIM_COLORS.length] }}
            />
            <span className="prim-name">{p.name}</span>
            <span className={`control-badge ${p.controlLevel}`}>
              {p.controlLevel === 'high' ? 'High' : 'Low'}
            </span>
            <div className="prim-actions">
              <button
                type="button"
                className="icon-btn"
                title={p.visible ? 'Hide' : 'Show'}
                aria-label={p.visible ? `Hide ${p.name}` : `Show ${p.name}`}
                onClick={(e) => { e.stopPropagation(); updatePrimitive(p.id, { visible: !p.visible }); }}
              >
                {p.visible ? <EyeIcon size={14} /> : <EyeOffIcon size={14} />}
              </button>
              <button
                type="button"
                className="icon-btn"
                title="Duplicate"
                aria-label={`Duplicate ${p.name}`}
                onClick={(e) => { e.stopPropagation(); duplicatePrimitive(p.id); }}
              >
                <CopyIcon size={14} />
              </button>
              <button
                type="button"
                className="icon-btn danger"
                title="Delete"
                aria-label={`Delete ${p.name}`}
                onClick={(e) => { e.stopPropagation(); removePrimitive(p.id); }}
              >
                <TrashIcon size={14} />
              </button>
            </div>
          </div>
        ))}

        {primitives.length > 0 && (
          <div className="scene-list-control">
            <div className="scene-list-control-label">
              <span>Low bbox margin</span>
              <span>{lowControlBBoxMarginPercent}%</span>
            </div>
            <input
              className="slider"
              type="range"
              min={LOW_CONTROL_BBOX_MARGIN_MIN}
              max={LOW_CONTROL_BBOX_MARGIN_MAX}
              step={LOW_CONTROL_BBOX_MARGIN_STEP}
              value={lowControlBBoxMargin}
              onChange={(e) => setLowControlBBoxMargin(Number.parseFloat(e.target.value))}
              title="0% is the tight bounding box touching the low-control superquadrics."
            />
          </div>
        )}
      </div>

      <div className="add-section">
        <button type="button" className="btn-primary" onClick={() => setShowPresets(!showPresets)}>
          <PlusIcon size={15} />
          Add Primitive
        </button>
        {showPresets && (
          <div className="preset-menu">
            <button className="preset-item" onClick={() => handleAdd()}>Blank (ball)</button>
            {Object.keys(PRESETS).map(k => (
              <button key={k} className="preset-item" onClick={() => handleAdd(k)}>
                {k}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
