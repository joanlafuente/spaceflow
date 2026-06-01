import { useCallback, useState, useRef, useEffect } from 'react';
import { useStore } from '../state/store';
import type { Primitive } from '../state/store';
import { exportNpz } from '../mesh/npzExport';
import type { PrimitiveExport } from '../mesh/npzExport';
import { importNpzToPrimitives, maybeRescalePrimitivesForEditor } from '../mesh/npzImport';
import { primitiveToExport } from '../mesh/spaceflowExport';
import { isOrthogonal } from '../state/rotation';
import { eulerToMatrix, matrixToEuler } from '../state/rotation';
import { editFromText } from '../state/generate';
import { createFromTextViaSuperdec } from '../state/createPipeline';
import { generateWithSuperdec } from '../state/superdec';
import { generateWithSuperflex } from '../state/superflex';
import {
  fetchSpaceflowHistory,
  getSpaceflowRunStatus,
  openSpaceflowAsset,
  resolveSpaceflowUrl,
  saveSpaceflowAsset,
  startSpaceflowRun,
  type SpaceflowHistoryEntry,
  type SpaceflowOutputFile,
  type SpaceflowRunStatus,
} from '../state/spaceflow';
import { npzEditorUrl } from '../state/npzUrl';
import {
  captureSuperquadricRenderBlob,
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

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

function outputFileLabel(file: SpaceflowOutputFile) {
  switch (file.relative_path) {
    case 'out_sim.glb':
      return 'Refined geometry';
    case 'out_app.glb':
      return 'Appearance-refined geometry';
    case 'out_gaussian_sim.mp4':
      return 'Refined Gaussian video';
    case 'out_gaussian_app.mp4':
      return 'Appearance Gaussian video';
    case 'sample.glb':
      return 'Structure mesh';
    case 'struct_mesh_zup.glb':
      return 'Structure mesh Z-up';
    case 'struct_mesh.glb':
      return 'Structure mesh Y-up';
    case 'spatial_control_mesh.ply':
      return 'All SQ control mesh';
    case 'high_control_spatial_control_mesh.ply':
      return 'High-control mesh';
    case 'low_control_superquadric_mask.ply':
      return 'Low-control mask';
    case 'voxels/struct_voxels.ply':
      return 'Structure voxels';
    case 'struct_renders/000.png':
      return 'Structure preview';
    case 'denoising_evolution.mp4':
      return 'Denoising video';
    default:
      return file.relative_path;
  }
}

const SPACEFLOW_INSPECTION_MESH_PRIORITY = [
  'out_sim.glb',
  'out_app.glb',
  'struct_mesh_zup.glb',
  'sample.glb',
  'struct_mesh.glb',
  'app_mesh_zup.glb',
  'app_mesh.glb',
];

function pickSpaceflowInspectionMesh(files: SpaceflowOutputFile[]) {
  const byPath = new Map(files.map(file => [file.relative_path.toLowerCase(), file]));
  for (const relativePath of SPACEFLOW_INSPECTION_MESH_PRIORITY) {
    const file = byPath.get(relativePath);
    if (file) return file;
  }
  return (
    files.find(file => file.kind === 'mesh' && /\.(glb|gltf)$/i.test(file.relative_path))
  );
}

let nextPresetId = 0;

export default function TopBar() {
  const primitives = useStore(s => s.primitives);
  const selectedId = useStore(s => s.selectedId);
  const selectPrimitive = useStore(s => s.selectPrimitive);
  const loadPreset = useStore(s => s.loadPreset);
  const setMeshInspection = useStore(s => s.setMeshInspection);
  const rotateAllWorld = useStore(s => s.rotateAllWorld);
  const undo = useStore(s => s.undo);
  const redo = useStore(s => s.redo);
  const undoStack = useStore(s => s.undoStack);
  const redoStack = useStore(s => s.redoStack);
  const lowControlBBoxMargin = useStore(s => s.lowControlBBoxMargin);
  const [toast, setToast] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [showRotateAll, setShowRotateAll] = useState(false);
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
  const [showSuperflex, setShowSuperflex] = useState(false);
  const [superflexGenerating, setSuperflexGenerating] = useState(false);
  const [superflexFile, setSuperflexFile] = useState<File | null>(null);
  const [superflexName, setSuperflexName] = useState('');
  const [superflexZUp, setSuperflexZUp] = useState(false);
  const [superflexNormalize, setSuperflexNormalize] = useState(true);
  const [superflexLmOptimization, setSuperflexLmOptimization] = useState(false);
  const [superflexMaxPrimitives, setSuperflexMaxPrimitives] = useState('16');
  const [superflexExistThreshold, setSuperflexExistThreshold] = useState('0.5');
  const [showSpaceflow, setShowSpaceflow] = useState(false);
  const [spaceflowSaving, setSpaceflowSaving] = useState(false);
  const [spaceflowRunning, setSpaceflowRunning] = useState(false);
  const [spaceflowHistory, setSpaceflowHistory] = useState<SpaceflowHistoryEntry[]>([]);
  const [spaceflowHistoryLoading, setSpaceflowHistoryLoading] = useState(false);
  const [showSpaceflowHistoryPanel, setShowSpaceflowHistoryPanel] = useState(false);
  const [spaceflowRun, setSpaceflowRun] = useState<SpaceflowRunStatus | null>(null);
  const [spaceflowLogTail, setSpaceflowLogTail] = useState('');
  const [spaceflowTextPrompt, setSpaceflowTextPrompt] = useState('A chair');
  const [spaceflowAppearanceMode, setSpaceflowAppearanceMode] = useState<'text' | 'image'>('text');
  const [spaceflowAppearanceText, setSpaceflowAppearanceText] = useState('');
  const [spaceflowAppearanceImagePath, setSpaceflowAppearanceImagePath] = useState('');
  const [spaceflowAppearanceImageFile, setSpaceflowAppearanceImageFile] = useState<File | null>(null);
  const [spaceflowLowTau, setSpaceflowLowTau] = useState('3.0');
  const [spaceflowHighTau, setSpaceflowHighTau] = useState('10.0');
  const [spaceflowPolyakTau, setSpaceflowPolyakTau] = useState('0.18');
  const [spaceflowOutputName, setSpaceflowOutputName] = useState('');
  const [spaceflowConvertYupToZup, setSpaceflowConvertYupToZup] = useState(true);
  const [spaceflowDryRun, setSpaceflowDryRun] = useState(false);
  const [projectName, setProjectName] = useState('superquadrics');
  const [genMode, setGenMode] = useState<'create' | 'edit'>('create');
  const [editFocusNames, setEditFocusNames] = useState<string[]>([]);
  const [includeViewportInEdit, setIncludeViewportInEdit] = useState(true);
  const [viewportPreviewUrl, setViewportPreviewUrl] = useState<string | null>(null);
  const [viewportPreviewModal, setViewportPreviewModal] = useState(false);
  const [viewportModalUrl, setViewportModalUrl] = useState<string | null>(null);
  const genInputRef = useRef<HTMLInputElement>(null);
  const superdecNameRef = useRef<HTMLInputElement>(null);
  const superflexNameRef = useRef<HTMLInputElement>(null);
  const spaceflowPromptRef = useRef<HTMLInputElement>(null);
  const inspectedSpaceflowRunRef = useRef<string | null>(null);

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
    if (showSuperflex) {
      setTimeout(() => superflexNameRef.current?.focus(), 50);
    }
  }, [showSuperflex]);

  useEffect(() => {
    if (showSpaceflow) {
      setTimeout(() => spaceflowPromptRef.current?.focus(), 50);
    }
  }, [showSpaceflow]);

  useEffect(() => {
    if (!viewportPreviewModal) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setViewportPreviewModal(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [viewportPreviewModal]);

  useEffect(() => {
    if (!spaceflowRun?.run_id || spaceflowRun.status !== 'running') return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getSpaceflowRunStatus(spaceflowRun.run_id);
        if (cancelled) return;
        setSpaceflowRun(status.run);
        setSpaceflowLogTail(status.logTail);
        if (status.run.status !== 'running') {
          setSpaceflowRunning(false);
          const inspectedFile =
            status.run.status === 'succeeded'
              ? inspectSpaceflowRunMesh(status.run, true)
              : null;
          showToast(
            inspectedFile
              ? `SpaceFlow succeeded: loaded ${outputFileLabel(inspectedFile)}`
              : `SpaceFlow ${status.run.status}: ${status.run.output_dir ?? spaceflowRun.run_id}`,
            8000,
          );
        }
      } catch (err) {
        if (!cancelled) setSpaceflowLogTail(`Status check failed: ${err instanceof Error ? err.message : err}`);
      }
    };
    const id = window.setInterval(() => void poll(), 5000);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [spaceflowRun?.run_id, spaceflowRun?.status]);

  const openViewportPreviewModal = useCallback(() => {
    const full = captureViewportDataUrl();
    setViewportModalUrl(full ?? viewportPreviewUrl);
    setViewportPreviewModal(true);
  }, [viewportPreviewUrl]);

  const allOrtho = primitives.every(p => isOrthogonal(p.rotation));
  const hasWarnings = !allOrtho;
  const spaceflowOutputFiles = spaceflowRun?.output_files ?? [];
  const spaceflowInspectionMesh = pickSpaceflowInspectionMesh(spaceflowOutputFiles);
  const spaceflowPreviewImage =
    spaceflowOutputFiles.find(file => file.relative_path === 'struct_renders/000.png') ??
    spaceflowOutputFiles.find(file => file.kind === 'image');
  const spaceflowVisibleOutputs = spaceflowOutputFiles
    .filter(file => file.kind !== 'log')
    .slice(0, 14);
  const showToast = (msg: string, durationMs = 3000) => {
    setToast(msg);
    setTimeout(() => setToast(null), durationMs);
  };

  function inspectSpaceflowRunMesh(run: SpaceflowRunStatus, automatic = false) {
    const meshFile = pickSpaceflowInspectionMesh(run.output_files ?? []);
    if (!meshFile) return null;
    if (automatic && inspectedSpaceflowRunRef.current === run.run_id) return meshFile;
    inspectedSpaceflowRunRef.current = run.run_id;
    setMeshInspection({
      url: resolveSpaceflowUrl(meshFile.url),
      name: outputFileLabel(meshFile),
      runId: run.run_id,
      path: meshFile.path,
      relativePath: meshFile.relative_path,
    });
    setShowSpaceflow(false);
    return meshFile;
  }

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

  const resetSuperflex = useCallback(() => {
    setShowSuperflex(false);
    setSuperflexFile(null);
    setSuperflexName('');
    setSuperflexZUp(false);
    setSuperflexNormalize(true);
    setSuperflexLmOptimization(false);
    setSuperflexMaxPrimitives('16');
    setSuperflexExistThreshold('0.5');
  }, []);

  const refreshSpaceflowHistory = useCallback(async () => {
    setSpaceflowHistoryLoading(true);
    try {
      const entries = await fetchSpaceflowHistory(30);
      setSpaceflowHistory(entries);
    } catch (err) {
      showToast(`History failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setSpaceflowHistoryLoading(false);
    }
  }, []);

  const handleSaveSpaceflowInputs = useCallback(async () => {
    if (spaceflowSaving || primitives.length === 0) return;
    setSpaceflowSaving(true);
    try {
      const lowTau = Number.parseFloat(spaceflowLowTau) || 3;
      const highTau = Number.parseFloat(spaceflowHighTau) || 10;
      const { entry, bundle } = await saveSpaceflowAsset({
        projectName,
        primitives,
        lowTau,
        highTau,
        lowControlBBoxMargin,
      });
      showToast(
        `Saved SpaceFlow inputs (${bundle.counts.high} high, ${bundle.counts.low} low)\n${entry.asset_dir}`,
        8000,
      );
      if (showSpaceflowHistoryPanel) await refreshSpaceflowHistory();
    } catch (err) {
      showToast(`SpaceFlow save failed: ${err instanceof Error ? err.message : err}`, 10000);
    } finally {
      setSpaceflowSaving(false);
    }
  }, [
    primitives,
    projectName,
    refreshSpaceflowHistory,
    showSpaceflowHistoryPanel,
    spaceflowHighTau,
    spaceflowLowTau,
    spaceflowSaving,
    lowControlBBoxMargin,
  ]);

  const handleOpenSpaceflowHistory = useCallback(async (entry: SpaceflowHistoryEntry) => {
    try {
      const prims = await openSpaceflowAsset(entry);
      loadPreset(prims);
      setProjectName(entry.project_name || projectName);
      showToast(`Loaded ${prims.length} primitives from ${entry.project_name}`);
      setShowSpaceflow(false);
    } catch (err) {
      showToast(`Open saved asset failed: ${err instanceof Error ? err.message : err}`, 8000);
    }
  }, [loadPreset, projectName]);

  const handleStartSpaceflowRun = useCallback(async () => {
    if (spaceflowRunning || primitives.length === 0) return;
    const lowTau = Number.parseFloat(spaceflowLowTau);
    const highTau = Number.parseFloat(spaceflowHighTau);
    const polyakTau = Number.parseFloat(spaceflowPolyakTau);
    if (!Number.isFinite(lowTau) || !Number.isFinite(highTau) || highTau <= lowTau) {
      showToast('High tau must be greater than low tau.', 5000);
      return;
    }
    if (!spaceflowTextPrompt.trim()) {
      showToast('Enter a SpaceFlow text prompt.', 5000);
      return;
    }
    setSpaceflowRunning(true);
    setSpaceflowLogTail('');
    try {
      const outputName =
        spaceflowOutputName.trim() ||
        projectName.replace(/[^a-zA-Z0-9_-]/g, '_') ||
        'spaceflow_run';
      const { run, bundle } = await startSpaceflowRun({
        projectName,
        primitives,
        appearanceImageFile: spaceflowAppearanceImageFile,
        runConfig: {
          textPrompt: spaceflowTextPrompt.trim(),
          appearanceMode: spaceflowAppearanceMode,
          appearanceText: spaceflowAppearanceText.trim() || spaceflowTextPrompt.trim(),
          appearanceImagePath: spaceflowAppearanceImagePath.trim(),
          lowTau,
          highTau,
          polyakTau: Number.isFinite(polyakTau) ? polyakTau : 0.18,
          outputName,
          convertYupToZup: spaceflowConvertYupToZup,
          lowControlBBoxMargin,
          dryRun: spaceflowDryRun,
        },
      });
      setSpaceflowRun(run);
      if (run.status === 'succeeded') {
        inspectSpaceflowRunMesh(run, true);
      }
      showToast(
        `${spaceflowDryRun ? 'Prepared' : 'Started'} SpaceFlow (${bundle.counts.high} high, ${bundle.counts.low} low)\n${run.output_dir ?? run.run_id}`,
        8000,
      );
      if (showSpaceflowHistoryPanel) await refreshSpaceflowHistory();
      if (run.status !== 'running') setSpaceflowRunning(false);
    } catch (err) {
      showToast(`SpaceFlow run failed: ${err instanceof Error ? err.message : err}`, 10000);
      setSpaceflowRunning(false);
    }
  }, [
    primitives,
    projectName,
    refreshSpaceflowHistory,
    showSpaceflowHistoryPanel,
    spaceflowAppearanceImageFile,
    spaceflowAppearanceImagePath,
    spaceflowAppearanceMode,
    spaceflowAppearanceText,
    spaceflowConvertYupToZup,
    spaceflowDryRun,
    spaceflowHighTau,
    spaceflowLowTau,
    spaceflowOutputName,
    spaceflowPolyakTau,
    spaceflowRunning,
    spaceflowTextPrompt,
    lowControlBBoxMargin,
  ]);

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
        const result = await createFromTextViaSuperdec(prompt, {
          name: prompt.replace(/^a\s+/i, '').trim() || 'superquadrics',
        });
        const prims = result.primitives;
        loadPreset(prims);
        setProjectName(prompt.replace(/^a\s+/i, '').trim() || 'superquadrics');
        showToast(
          `Generated ${result.primitiveCount} primitives from ${result.pointCount} TRELLIS points`
        );
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

  const handleSuperflexGenerate = useCallback(async () => {
    if (!superflexFile || superflexGenerating) return;
    setSuperflexGenerating(true);
    const baseName =
      superflexName.trim() ||
      superflexFile.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_') ||
      'superflex';
    try {
      const maxPrimitives = Math.max(0, Number.parseInt(superflexMaxPrimitives || '0', 10) || 0);
      const existThreshold = Math.min(
        1,
        Math.max(0, Number.parseFloat(superflexExistThreshold || '0.5') || 0.5),
      );
      const result = await generateWithSuperflex({
        file: superflexFile,
        name: baseName,
        zUp: superflexZUp,
        normalize: superflexNormalize,
        lmOptimization: superflexLmOptimization,
        maxPrimitives,
        existThreshold,
      });
      loadPreset(result.primitives);
      setProjectName(baseName);
      showToast(`Generated ${result.primitiveCount} primitives with SuperFlex`);
      resetSuperflex();
    } catch (err) {
      showToast(`SuperFlex failed: ${err instanceof Error ? err.message : err}`, 10000);
    } finally {
      setSuperflexGenerating(false);
    }
  }, [
    loadPreset,
    resetSuperflex,
    setProjectName,
    superflexExistThreshold,
    superflexFile,
    superflexGenerating,
    superflexLmOptimization,
    superflexMaxPrimitives,
    superflexName,
    superflexNormalize,
    superflexZUp,
  ]);

  const handleDownloadNpz = useCallback(async () => {
    try {
      const exports: PrimitiveExport[] = primitives.map(primitiveToExport);
      const blob = await exportNpz(exports);
      const filename = `${projectName.replace(/[^a-zA-Z0-9_-]/g, '_') || 'superquadrics'}.npz`;
      downloadBlob(blob, filename);
      showToast(`Downloaded ${filename}`);
    } catch (err) {
      showToast(`Export failed: ${err}`);
    }
    setShowExport(false);
  }, [primitives, projectName]);

  const handleDownloadRendering = useCallback(async () => {
    if (primitives.length === 0) return;
    try {
      const blob = await captureSuperquadricRenderBlob();
      if (!blob) {
        showToast('Could not render superquadrics. Switch back to the SQ viewport and try again.', 5000);
        return;
      }
      const filename = `${projectName.replace(/[^a-zA-Z0-9_-]/g, '_') || 'superquadrics'}_render.png`;
      downloadBlob(blob, filename);
      showToast(`Downloaded ${filename}`);
    } catch (err) {
      showToast(`Render export failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setShowExport(false);
    }
  }, [primitives.length, projectName]);

  const handleCopyJson = useCallback(() => {
    const data = primitives.map(p => ({
      name: p.name,
      controlLevel: p.controlLevel,
      scales: p.scales,
      shapes: p.shapes,
      translation: p.translation,
      eulerDeg: p.eulerDeg,
      ...(p.tapering !== undefined ? { tapering: p.tapering } : {}),
      ...(p.bending !== undefined ? { bending: p.bending } : {}),
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
          controlLevel?: 'high' | 'low';
          scales: [number, number, number];
          shapes: [number, number];
          translation: [number, number, number];
          eulerDeg?: [number, number, number];
          rotation?: number[][];
          tapering?: [number, number];
          bending?: [number, number, number, number, number, number];
        }>;
        const prims: Primitive[] = data.map((d) => {
          const euler: [number, number, number] = d.eulerDeg ?? (d.rotation ? matrixToEuler(d.rotation) : [0, 0, 0]);
          const rotation = d.rotation ?? eulerToMatrix(euler);
          return {
            id: `imported_${++nextPresetId}_${Date.now()}`,
            name: d.name ?? `Imported ${nextPresetId}`,
            visible: true,
            controlLevel: d.controlLevel ?? 'high',
            scales: d.scales,
            shapes: d.shapes,
            translation: d.translation,
            rotation,
            eulerDeg: euler,
            ...(d.tapering !== undefined ? { tapering: d.tapering } : {}),
            ...(d.bending !== undefined ? { bending: d.bending } : {}),
          };
        });
        loadPreset(maybeRescalePrimitivesForEditor(prims));
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

  const handleOpenNpzPath = useCallback(() => {
    const path = window.prompt('Path to superflex.npz');
    const source = path?.trim();
    if (!source) return;
    if (!source.toLowerCase().split(/[?#]/, 1)[0]?.endsWith('.npz')) {
      showToast('Please enter a .npz path.', 5000);
      return;
    }
    setShowExport(false);
    window.location.assign(npzEditorUrl(source));
  }, []);

  const handleLoadTemplate = useCallback((template: string) => {
    const templates: Record<string, () => Primitive[]> = {
      'Single Ellipsoid': () => {
        const euler: [number, number, number] = [0, 0, 0];
        return [{
          id: `t_${++nextPresetId}_${Date.now()}`, name: 'Ellipsoid', visible: true, controlLevel: 'high',
          scales: [0.5, 0.5, 1], shapes: [1, 1], translation: [0, 0, 0],
          rotation: eulerToMatrix(euler), eulerDeg: euler,
        }];
      },
      'Table (5 parts)': () => {
        const leg = (x: number, z: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`, name: `Leg ${idx}`, visible: true, controlLevel: 'high',
          scales: [0.06, 0.4, 0.06], shapes: [0.4, 0.4], translation: [x, -0.44, z],
          rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0],
        });
        return [
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Top', visible: true, controlLevel: 'high',
            scales: [0.8, 0.04, 0.5], shapes: [0.3, 0.3], translation: [0, 0, 0],
            rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0] },
          leg(-0.65, -0.4, 1), leg(0.65, -0.4, 2),
          leg(-0.65, 0.4, 3), leg(0.65, 0.4, 4),
        ];
      },
      'Chair (6 parts)': () => {
        const leg = (x: number, z: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`, name: `Leg ${idx}`, visible: true, controlLevel: 'high',
          scales: [0.05, 0.35, 0.05], shapes: [0.4, 0.4], translation: [x, -0.39, z],
          rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0],
        });
        return [
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Seat', visible: true, controlLevel: 'high',
            scales: [0.5, 0.04, 0.45], shapes: [0.3, 0.3], translation: [0, 0, 0],
            rotation: [[1,0,0],[0,1,0],[0,0,1]], eulerDeg: [0, 0, 0] },
          { id: `t_${++nextPresetId}_${Date.now()}`, name: 'Backrest', visible: true, controlLevel: 'high',
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
        <div className={`export-dropdown rotate-all-group${showRotateAll ? ' is-open' : ''}`}>
          <button
            type="button"
            className={`toolbar-btn toolbar-btn-menu${showRotateAll ? ' is-open' : ''}`}
            onClick={() => setShowRotateAll(v => !v)}
            disabled={primitives.length === 0}
            title="Rotate all primitives 90° about a world axis (wrong up-axis / upside-down)"
          >
            <svg
              className="toolbar-btn-menu-icon"
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M23 4v6h-6" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
            <span className="toolbar-btn-menu-label">90°</span>
            <span className="toolbar-btn-menu-caret" aria-hidden>
              ▾
            </span>
          </button>
          {showRotateAll && (
            <div className="dropdown-menu dropdown-menu-toolbar" role="menu">
              <button
                type="button"
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  rotateAllWorld([90, 0, 0]);
                  setShowRotateAll(false);
                  showToast('Rotated all parts +90° about world X');
                }}
              >
                +90° about world X
              </button>
              <button
                type="button"
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  rotateAllWorld([0, 90, 0]);
                  setShowRotateAll(false);
                  showToast('Rotated all parts +90° about world Y');
                }}
              >
                +90° about world Y
              </button>
              <button
                type="button"
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  rotateAllWorld([0, 0, 90]);
                  setShowRotateAll(false);
                  showToast('Rotated all parts +90° about world Z');
                }}
              >
                +90° about world Z
              </button>
            </div>
          )}
        </div>
        <span className="separator" />
        <div className={`generate-group ${showGenerate ? 'is-open' : ''}`}>
          {!showGenerate ? (
            <button
              className="btn-generate"
              onClick={() => {
                resetSuperdec();
                resetSuperflex();
                setShowSpaceflow(false);
                setShowGenerate(true);
                setTimeout(() => genInputRef.current?.focus(), 50);
              }}
              disabled={generating || superdecGenerating || superflexGenerating}
              title="Create via TRELLIS + SuperDec, or edit via Ollama"
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
              onClick={() => {
                setShowGenerate(false);
                setShowSpaceflow(false);
                resetSuperflex();
                setShowSuperdec(true);
              }}
              disabled={superdecGenerating || superflexGenerating}
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
                      accept=".ply,.pcd,.xyz,.xyzn,.xyzrgb,.pts,.obj,.stl,.glb,.gltf"
                      onChange={(e) => setSuperdecFile(e.target.files?.[0] ?? null)}
                      disabled={superdecGenerating}
                    />
                    <span>{superdecFile ? superdecFile.name : 'Choose point cloud or mesh file'}</span>
                  </label>
                  <p className="edit-focus-hint">
                    Supports `.ply` directly and also common formats such as `.pcd`, `.xyz`, `.pts`, `.obj`, `.stl`, `.glb`, and `.gltf` via the service-side loader.
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
        <div className={`generate-group ${showSuperflex ? 'is-open' : ''}`}>
          {!showSuperflex ? (
            <button
              className="btn-generate btn-superflex"
              onClick={() => {
                setShowGenerate(false);
                setShowSpaceflow(false);
                resetSuperdec();
                setShowSuperflex(true);
              }}
              disabled={superdecGenerating || superflexGenerating}
              title="SuperFlex: superquadrics with tapering and bending (separate service)"
            >
              {superflexGenerating ? '...' : 'SuperFlex'}
            </button>
          ) : (
            <>
              <div
                className="generate-popover-backdrop"
                onClick={() => {
                  if (!superflexGenerating) resetSuperflex();
                }}
                aria-hidden
              />
              <div className="generate-open-bar">
                <div className="gen-mode-row" role="group" aria-label="SuperFlex mode">
                  <span className="superdec-open-label">Taper + bend</span>
                </div>
                <button
                  className="toolbar-btn"
                  onClick={resetSuperflex}
                  disabled={superflexGenerating}
                  title="Close (Esc)"
                  type="button"
                >
                  ✕
                </button>
              </div>
              <div
                className="generate-popover superdec-popover"
                role="dialog"
                aria-label="SuperFlex from point cloud or mesh"
                onClick={(e) => e.stopPropagation()}
              >
                <label className="generate-popover-label" htmlFor="sq-superflex-name-input">
                  Scene name
                </label>
                <input
                  id="sq-superflex-name-input"
                  ref={superflexNameRef}
                  type="text"
                  className="generate-input generate-input-popover"
                  placeholder="e.g. chair_scan"
                  value={superflexName}
                  onChange={(e) => setSuperflexName(e.target.value)}
                  disabled={superflexGenerating}
                />

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Input point cloud or mesh</span>
                  </div>
                  <label className="superdec-file-picker">
                    <input
                      type="file"
                      accept=".ply,.pcd,.xyz,.xyzn,.xyzrgb,.pts,.obj,.stl,.glb,.gltf"
                      onChange={(e) => setSuperflexFile(e.target.files?.[0] ?? null)}
                      disabled={superflexGenerating}
                    />
                    <span>{superflexFile ? superflexFile.name : 'Choose file'}</span>
                  </label>
                  <p className="edit-focus-hint">
                    Same formats as SuperDec. Use a SuperFlex-trained checkpoint so tapering and bending heads are
                    populated (generic SuperDec weights still run but may predict zeros for bend/taper).
                  </p>
                </div>

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Inference options</span>
                  </div>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superflexZUp}
                      onChange={(e) => setSuperflexZUp(e.target.checked)}
                      disabled={superflexGenerating}
                    />
                    <span>Treat input as Z-up and convert it into the editor&apos;s Y-up frame</span>
                  </label>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superflexNormalize}
                      onChange={(e) => setSuperflexNormalize(e.target.checked)}
                      disabled={superflexGenerating}
                    />
                    <span>Normalize point cloud before inference</span>
                  </label>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={superflexLmOptimization}
                      onChange={(e) => setSuperflexLmOptimization(e.target.checked)}
                      disabled={superflexGenerating}
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
                        value={superflexMaxPrimitives}
                        onChange={(e) => setSuperflexMaxPrimitives(e.target.value)}
                        disabled={superflexGenerating}
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
                        value={superflexExistThreshold}
                        onChange={(e) => setSuperflexExistThreshold(e.target.value)}
                        disabled={superflexGenerating}
                      />
                    </label>
                  </div>
                </div>

                <div className="generate-popover-footer">
                  <button
                    className="btn-generate-go"
                    type="button"
                    onClick={handleSuperflexGenerate}
                    disabled={superflexGenerating || !superflexFile}
                  >
                    {superflexGenerating ? (
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
        <div className={`generate-group ${showSpaceflow ? 'is-open' : ''}`}>
          {!showSpaceflow ? (
            <button
              className="btn-generate btn-spaceflow"
              onClick={() => {
                setShowGenerate(false);
                resetSuperdec();
                resetSuperflex();
                setShowSpaceflow(true);
              }}
              disabled={spaceflowSaving || spaceflowRunning}
              title="Save SpaceFlow inputs and launch two-level tau runs"
            >
              {spaceflowRunning ? 'Running...' : 'SpaceFlow'}
            </button>
          ) : (
            <>
              <div
                className="generate-popover-backdrop"
                onClick={() => {
                  if (!spaceflowSaving && !spaceflowRunning) setShowSpaceflow(false);
                }}
                aria-hidden
              />
              <div className="generate-open-bar">
                <div className="gen-mode-row" role="group" aria-label="SpaceFlow">
                  <span className="superdec-open-label">SpaceFlow refinement</span>
                </div>
                <button
                  className="toolbar-btn"
                  onClick={() => setShowSpaceflow(false)}
                  disabled={spaceflowSaving || spaceflowRunning}
                  title="Close"
                  type="button"
                >
                  ✕
                </button>
              </div>
              <div
                className="generate-popover spaceflow-popover"
                role="dialog"
                aria-label="Run SpaceFlow"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Inputs</span>
                    <div className="spaceflow-head-actions">
                      <button
                        type="button"
                        className="edit-focus-add-sel"
                        onClick={() => {
                          const next = !showSpaceflowHistoryPanel;
                          setShowSpaceflowHistoryPanel(next);
                          if (next) void refreshSpaceflowHistory();
                        }}
                        disabled={spaceflowHistoryLoading}
                        title="Show saved SpaceFlow input bundles"
                      >
                        {showSpaceflowHistoryPanel ? 'Hide saved' : 'Saved inputs'}
                      </button>
                      <button
                        type="button"
                        className="edit-focus-add-sel"
                        onClick={handleSaveSpaceflowInputs}
                        disabled={spaceflowSaving || primitives.length === 0}
                        title="Save all.npz, high_control.npz, and low_control_bbox.npz"
                      >
                        {spaceflowSaving ? 'Saving...' : 'Save inputs'}
                      </button>
                    </div>
                  </div>
                  <p className="spaceflow-summary">
                    {primitives.filter(p => p.controlLevel === 'high').length} high · {primitives.filter(p => p.controlLevel === 'low').length} low
                  </p>
                  <p className="edit-focus-hint">
                    Current launch path writes the structure-stage outputs; high and low groups control the local tau sampler.
                  </p>
                </div>

                {showSpaceflowHistoryPanel && (
                  <div className="edit-focus-block spaceflow-history-block">
                    <div className="edit-focus-head">
                      <span className="edit-focus-title">Saved inputs</span>
                      <button
                        type="button"
                        className="edit-focus-add-sel"
                        onClick={() => void refreshSpaceflowHistory()}
                        disabled={spaceflowHistoryLoading}
                      >
                        {spaceflowHistoryLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>
                    <div className="spaceflow-history-list">
                      {spaceflowHistory.length === 0 && (
                        <p className="edit-focus-hint">No saved SpaceFlow input bundles yet.</p>
                      )}
                      {spaceflowHistory.map(entry => (
                        <button
                          key={entry.id}
                          type="button"
                          className="spaceflow-history-item"
                          onClick={() => void handleOpenSpaceflowHistory(entry)}
                          title={`${entry.paths.all}\n${entry.paths.high_control}\n${entry.paths.low_control_bbox}`}
                        >
                          <span className="spaceflow-history-title">{entry.project_name}</span>
                          <span className="spaceflow-history-meta">
                            {entry.counts?.high ?? '?'} high · {entry.counts?.low ?? '?'} low · {entry.saved_at}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <label className="generate-popover-label" htmlFor="sq-spaceflow-prompt-input">
                  Shape prompt
                </label>
                <input
                  id="sq-spaceflow-prompt-input"
                  ref={spaceflowPromptRef}
                  type="text"
                  className="generate-input generate-input-popover"
                  value={spaceflowTextPrompt}
                  onChange={(e) => setSpaceflowTextPrompt(e.target.value)}
                  disabled={spaceflowRunning}
                  placeholder="e.g. A chair"
                />

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Appearance guidance</span>
                    <div className="gen-mode-row">
                      <button
                        type="button"
                        className={`gen-mode-btn ${spaceflowAppearanceMode === 'text' ? 'active' : ''}`}
                        onClick={() => setSpaceflowAppearanceMode('text')}
                        disabled={spaceflowRunning}
                      >
                        Text
                      </button>
                      <button
                        type="button"
                        className={`gen-mode-btn ${spaceflowAppearanceMode === 'image' ? 'active' : ''}`}
                        onClick={() => setSpaceflowAppearanceMode('image')}
                        disabled={spaceflowRunning}
                      >
                        Image
                      </button>
                    </div>
                  </div>
                  {spaceflowAppearanceMode === 'text' ? (
                    <input
                      type="text"
                      className="generate-input generate-input-popover"
                      value={spaceflowAppearanceText}
                      onChange={(e) => setSpaceflowAppearanceText(e.target.value)}
                      disabled={spaceflowRunning}
                      placeholder="Defaults to the shape prompt"
                    />
                  ) : (
                    <div className="spaceflow-image-inputs">
                      <input
                        type="text"
                        className="generate-input generate-input-popover"
                        value={spaceflowAppearanceImagePath}
                        onChange={(e) => setSpaceflowAppearanceImagePath(e.target.value)}
                        disabled={spaceflowRunning}
                        placeholder="Cluster path to image, or choose file below"
                      />
                      <label className="superdec-file-picker">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => setSpaceflowAppearanceImageFile(e.target.files?.[0] ?? null)}
                          disabled={spaceflowRunning}
                        />
                        <span>{spaceflowAppearanceImageFile ? spaceflowAppearanceImageFile.name : 'Choose appearance image'}</span>
                      </label>
                    </div>
                  )}
                </div>

                <div className="edit-focus-block">
                  <div className="edit-focus-head">
                    <span className="edit-focus-title">Run options</span>
                  </div>
                  <div className="superdec-number-grid">
                    <label className="superdec-number-field">
                      <span>Low tau</span>
                      <input className="num-input" type="number" step="0.5" value={spaceflowLowTau} onChange={(e) => setSpaceflowLowTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="superdec-number-field">
                      <span>High tau</span>
                      <input className="num-input" type="number" step="0.5" value={spaceflowHighTau} onChange={(e) => setSpaceflowHighTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="superdec-number-field">
                      <span>Polyak tau</span>
                      <input className="num-input" type="number" step="0.01" value={spaceflowPolyakTau} onChange={(e) => setSpaceflowPolyakTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="superdec-number-field">
                      <span>Output name</span>
                      <input className="num-input" type="text" value={spaceflowOutputName} onChange={(e) => setSpaceflowOutputName(e.target.value)} disabled={spaceflowRunning} placeholder={projectName} />
                    </label>
                  </div>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={spaceflowConvertYupToZup}
                      onChange={(e) => setSpaceflowConvertYupToZup(e.target.checked)}
                      disabled={spaceflowRunning}
                    />
                    <span>Convert generated mesh from Y-up to Z-up</span>
                  </label>
                  <label className="edit-viewport-include">
                    <input
                      type="checkbox"
                      checked={spaceflowDryRun}
                      onChange={(e) => setSpaceflowDryRun(e.target.checked)}
                      disabled={spaceflowRunning}
                    />
                    <span>Dry run: save inputs and build command without GPU work</span>
                  </label>
                </div>

                <div className="generate-popover-footer">
                  <button
                    className="btn-generate-go"
                    type="button"
                    onClick={handleStartSpaceflowRun}
                    disabled={spaceflowRunning || primitives.length === 0}
                  >
                    {spaceflowRunning ? 'Running' : spaceflowDryRun ? 'Dry Run' : 'Run'}
                  </button>
                  {spaceflowRun?.run_id && (
                    <span className={`spaceflow-run-state ${spaceflowRun.status}`}>
                      {spaceflowRun.status}
                    </span>
                  )}
                </div>

                {spaceflowRun?.output_dir && (
                  <div className="spaceflow-output-path">
                    <span>Output directory</span>
                    <code className="spaceflow-path-block" title={spaceflowRun.output_dir}>
                      {spaceflowRun.output_dir}
                    </code>
                  </div>
                )}

                {spaceflowVisibleOutputs.length > 0 && (
                  <div className="edit-focus-block spaceflow-results-block">
                    <div className="edit-focus-head">
                      <span className="edit-focus-title">Generated files</span>
                      <div className="spaceflow-head-actions">
                        {spaceflowRun?.pipeline_stage === 'structure_only' ? (
                          <span className="spaceflow-stage-pill">structure stage</span>
                        ) : spaceflowRun?.pipeline_stage === 'full_pipeline' ? (
                          <span className="spaceflow-stage-pill">full pipeline</span>
                        ) : null}
                        {spaceflowRun && spaceflowInspectionMesh && (
                          <button
                            type="button"
                            className="edit-focus-add-sel"
                            onClick={() => {
                              const inspectedFile = inspectSpaceflowRunMesh(spaceflowRun);
                              showToast(
                                inspectedFile
                                  ? `Loaded ${outputFileLabel(inspectedFile)}`
                                  : 'No GLB mesh found in this run.',
                                5000,
                              );
                            }}
                          >
                            Inspect mesh
                          </button>
                        )}
                      </div>
                    </div>
                    {spaceflowPreviewImage && (
                      <a
                        className="spaceflow-output-preview"
                        href={resolveSpaceflowUrl(spaceflowPreviewImage.url)}
                        target="_blank"
                        rel="noreferrer"
                        title={spaceflowPreviewImage.path}
                      >
                        <img src={resolveSpaceflowUrl(spaceflowPreviewImage.url)} alt="SpaceFlow preview" />
                      </a>
                    )}
                    <div className="spaceflow-output-list">
                      {spaceflowVisibleOutputs.map(file => (
                        <a
                          key={file.relative_path}
                          className={`spaceflow-output-item ${file.kind}`}
                          href={resolveSpaceflowUrl(file.url)}
                          target="_blank"
                          rel="noreferrer"
                          title={file.path}
                        >
                          <span>{outputFileLabel(file)}</span>
                          <small>{file.kind} {formatFileSize(file.size)}</small>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {spaceflowLogTail && (
                  <div className="edit-focus-block spaceflow-log-block">
                    <div className="edit-focus-head">
                      <span className="edit-focus-title">Run log</span>
                    </div>
                    <pre className="spaceflow-log-tail">{spaceflowLogTail}</pre>
                  </div>
                )}
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
                onClick={handleDownloadRendering}
                disabled={primitives.length === 0}
              >
                Download rendering (.png)
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
              <button type="button" className="dropdown-item" onClick={handleOpenNpzPath}>
                Open .npz path...
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
