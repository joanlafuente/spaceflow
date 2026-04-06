import { create } from 'zustand';
import { eulerToMatrix, matrixToEuler } from './rotation';

export interface Primitive {
  id: string;
  name: string;
  visible: boolean;
  scales: [number, number, number];
  shapes: [number, number];
  translation: [number, number, number];
  rotation: number[][];
  eulerDeg: [number, number, number]; // cached Euler ZYX in degrees
}

interface HistoryEntry {
  primitives: Primitive[];
  selectedId: string | null;
}

export interface AppState {
  primitives: Primitive[];
  selectedId: string | null;
  previewResolution: number;
  showNormalized: boolean;

  undoStack: HistoryEntry[];
  redoStack: HistoryEntry[];

  addPrimitive: (preset?: Partial<Primitive>) => void;
  removePrimitive: (id: string) => void;
  duplicatePrimitive: (id: string) => void;
  selectPrimitive: (id: string | null) => void;
  updatePrimitive: (id: string, updates: Partial<Primitive>) => void;
  /** Same as updatePrimitive but does not record undo; use during drags, then rely on pushUndoSnapshot. */
  updatePrimitiveLive: (id: string, updates: Partial<Primitive>) => void;
  /** Push current primitives/selection as one undo step (e.g. once at drag start). */
  pushUndoSnapshot: () => void;
  reorderPrimitives: (fromIndex: number, toIndex: number) => void;
  setPreviewResolution: (res: number) => void;
  setShowNormalized: (v: boolean) => void;
  undo: () => void;
  redo: () => void;
  loadPreset: (primitives: Primitive[]) => void;
}

let idCounter = 0;
function nextId(): string {
  return `prim_${++idCounter}_${Date.now()}`;
}

function identity3(): number[][] {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

function defaultPrimitive(overrides?: Partial<Primitive>): Primitive {
  return {
    id: nextId(),
    name: `Primitive ${idCounter}`,
    visible: true,
    scales: [1, 1, 1],
    shapes: [1, 1],
    translation: [0, 0, 0],
    rotation: identity3(),
    eulerDeg: [0, 0, 0],
    ...overrides,
  };
}

function clonePrimitive(p: Primitive): Primitive {
  return {
    ...p,
    id: nextId(),
    name: `${p.name} (copy)`,
    scales: [...p.scales],
    shapes: [...p.shapes],
    translation: [...p.translation],
    rotation: p.rotation.map(r => [...r]),
    eulerDeg: [...p.eulerDeg],
  };
}

function snapshot(state: { primitives: Primitive[]; selectedId: string | null }): HistoryEntry {
  return {
    primitives: state.primitives.map(p => ({
      ...p,
      scales: [...p.scales],
      shapes: [...p.shapes],
      translation: [...p.translation],
      rotation: p.rotation.map(r => [...r]),
      eulerDeg: [...p.eulerDeg],
    })),
    selectedId: state.selectedId,
  };
}

function withSyncedRotation(updates: Partial<Primitive>): Partial<Primitive> {
  const finalUpdates = { ...updates };
  if (updates.eulerDeg && !updates.rotation) {
    finalUpdates.rotation = eulerToMatrix(updates.eulerDeg as [number, number, number]);
  } else if (updates.rotation && !updates.eulerDeg) {
    finalUpdates.eulerDeg = matrixToEuler(updates.rotation as number[][]);
  }
  return finalUpdates;
}

/** Canonical superquadric recipes for this editor (shapes + eulerDeg + rotation must stay in sync). */
function sqPreset(
  scales: [number, number, number],
  shapes: [number, number],
  eulerDeg: [number, number, number]
): Partial<Primitive> {
  return {
    scales,
    shapes,
    eulerDeg,
    rotation: eulerToMatrix(eulerDeg),
  };
}

export const PRESETS: Record<string, () => Partial<Primitive>> = {
  Ball: () => sqPreset([1, 1, 1], [1, 1], [0, 0, 0]),
  Ellipsoid: () => sqPreset([0.5, 0.5, 1], [1, 1], [0, 0, 0]),
  Cylinder: () => sqPreset([1, 1, 1], [0.05, 1], [90, 0, 0]),
  Cube: () => sqPreset([1, 1, 1], [0.05, 0.05], [90, 0, 0]),
  'Astroid (star)': () => sqPreset([1, 1, 1], [4, 4], [0, 0, 0]),
};

export const useStore = create<AppState>((set, get) => ({
  primitives: [],
  selectedId: null,
  previewResolution: 48,
  showNormalized: true,
  undoStack: [],
  redoStack: [],

  addPrimitive: (preset) => {
    const state = get();
    const entry = snapshot(state);
    const p = defaultPrimitive(preset);
    set({
      primitives: [...state.primitives, p],
      selectedId: p.id,
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  removePrimitive: (id) => {
    const state = get();
    const entry = snapshot(state);
    const newPrims = state.primitives.filter(p => p.id !== id);
    set({
      primitives: newPrims,
      selectedId: state.selectedId === id ? null : state.selectedId,
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  duplicatePrimitive: (id) => {
    const state = get();
    const entry = snapshot(state);
    const idx = state.primitives.findIndex(p => p.id === id);
    if (idx === -1) return;
    const copy = clonePrimitive(state.primitives[idx]);
    const newPrims = [...state.primitives];
    newPrims.splice(idx + 1, 0, copy);
    set({
      primitives: newPrims,
      selectedId: copy.id,
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  selectPrimitive: (id) => set({ selectedId: id }),

  updatePrimitive: (id, updates) => {
    const state = get();
    const entry = snapshot(state);
    const finalUpdates = withSyncedRotation(updates);

    set({
      primitives: state.primitives.map(p =>
        p.id === id ? { ...p, ...finalUpdates } : p
      ),
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  updatePrimitiveLive: (id, updates) => {
    const state = get();
    const finalUpdates = withSyncedRotation(updates);
    set({
      primitives: state.primitives.map(p =>
        p.id === id ? { ...p, ...finalUpdates } : p
      ),
    });
  },

  pushUndoSnapshot: () => {
    const state = get();
    const entry = snapshot(state);
    set({
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  reorderPrimitives: (fromIndex, toIndex) => {
    const state = get();
    const entry = snapshot(state);
    const newPrims = [...state.primitives];
    const [moved] = newPrims.splice(fromIndex, 1);
    newPrims.splice(toIndex, 0, moved);
    set({
      primitives: newPrims,
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },

  setPreviewResolution: (res) => set({ previewResolution: res }),
  setShowNormalized: (v) => set({ showNormalized: v }),

  undo: () => {
    const state = get();
    if (state.undoStack.length === 0) return;
    const prev = state.undoStack[state.undoStack.length - 1];
    set({
      primitives: prev.primitives,
      selectedId: prev.selectedId,
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [...state.redoStack, snapshot(state)],
    });
  },

  redo: () => {
    const state = get();
    if (state.redoStack.length === 0) return;
    const next = state.redoStack[state.redoStack.length - 1];
    set({
      primitives: next.primitives,
      selectedId: next.selectedId,
      undoStack: [...state.undoStack, snapshot(state)],
      redoStack: state.redoStack.slice(0, -1),
    });
  },

  loadPreset: (primitives) => {
    const state = get();
    const entry = snapshot(state);
    set({
      primitives,
      selectedId: primitives.length > 0 ? primitives[0].id : null,
      undoStack: [...state.undoStack, entry],
      redoStack: [],
    });
  },
}));
