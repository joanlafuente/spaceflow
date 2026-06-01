/* eslint-disable react-hooks/immutability, react-hooks/refs */
import { useRef, useMemo, useCallback, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport, Grid, Html } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';
import { useStore } from '../state/store';
import type { MeshInspectionSource, Primitive } from '../state/store';
import { createSuperquadricMesh, normalizeMergedVertices } from '../mesh/superquadric';
import { buildLowControlBoundingBoxPrimitive } from '../mesh/spaceflowExport';
import { setViewportCapture, setViewportRenderExport } from '../state/viewportCapture';

function superflexDeformForPrimitive(p: Primitive) {
  if (p.tapering === undefined && p.bending === undefined) return undefined;
  return {
    tapering: (p.tapering ?? [0, 0]) as [number, number],
    bending: (p.bending ?? [0, 0, 0, 0, 0, 0]) as [
      number,
      number,
      number,
      number,
      number,
      number,
    ],
  };
}

const RENDER_EXPORT_BG = '#ffffff';
const RENDER_EXPORT_HIGH_COLOR = '#f59e0b';
const RENDER_EXPORT_LOW_COLOR = '#f8fafc';
const RENDER_EXPORT_FRAME_FILL = 0.82;

function cloneExportCamera(camera: THREE.Camera, width: number, height: number): THREE.Camera {
  const exportCamera = camera.clone();
  if ((exportCamera as THREE.PerspectiveCamera).isPerspectiveCamera) {
    const c = exportCamera as THREE.PerspectiveCamera;
    c.aspect = width / height;
    c.updateProjectionMatrix();
  } else if ((exportCamera as THREE.OrthographicCamera).isOrthographicCamera) {
    (exportCamera as THREE.OrthographicCamera).updateProjectionMatrix();
  }
  exportCamera.updateMatrixWorld(true);
  return exportCamera;
}

function boxCorners(box: THREE.Box3): THREE.Vector3[] {
  return [
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
    new THREE.Vector3(box.max.x, box.max.y, box.max.z),
  ];
}

function fitExportCameraToMeshes(
  camera: THREE.Camera,
  width: number,
  height: number,
  meshes: THREE.Mesh[],
): THREE.Camera {
  const exportCamera = cloneExportCamera(camera, width, height);
  const bounds = new THREE.Box3();
  meshes.forEach(mesh => {
    mesh.updateWorldMatrix(true, false);
    bounds.expandByObject(mesh);
  });
  if (bounds.isEmpty()) return exportCamera;

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  bounds.getCenter(center);
  bounds.getSize(size);

  const viewDir = new THREE.Vector3();
  exportCamera.getWorldDirection(viewDir).normalize();
  const inverseCameraRotation = exportCamera.quaternion.clone().invert();
  const localCorners = boxCorners(bounds).map(corner =>
    corner.sub(center).applyQuaternion(inverseCameraRotation)
  );

  let minZ = Infinity;
  let maxZ = -Infinity;
  let maxX = 0;
  let maxY = 0;
  for (const corner of localCorners) {
    maxX = Math.max(maxX, Math.abs(corner.x));
    maxY = Math.max(maxY, Math.abs(corner.y));
    minZ = Math.min(minZ, corner.z);
    maxZ = Math.max(maxZ, corner.z);
  }

  const frameFill = Math.min(Math.max(RENDER_EXPORT_FRAME_FILL, 0.1), 0.98);
  const padding = Math.max(size.length() * 0.03, 0.01);

  if ((exportCamera as THREE.PerspectiveCamera).isPerspectiveCamera) {
    const c = exportCamera as THREE.PerspectiveCamera;
    const tanY = Math.tan(THREE.MathUtils.degToRad(c.fov) / 2);
    const tanX = tanY * c.aspect;
    let distance = Math.max(0.01, maxZ + padding);
    for (const corner of localCorners) {
      distance = Math.max(
        distance,
        corner.z + Math.abs(corner.x) / (tanX * frameFill),
        corner.z + Math.abs(corner.y) / (tanY * frameFill),
      );
    }
    distance += padding;
    c.position.copy(center).addScaledVector(viewDir, -distance);
    c.near = Math.max(0.001, distance - maxZ - padding * 2);
    c.far = Math.max(c.near + 1, distance - minZ + padding * 4);
    c.updateProjectionMatrix();
  } else if ((exportCamera as THREE.OrthographicCamera).isOrthographicCamera) {
    const c = exportCamera as THREE.OrthographicCamera;
    const aspect = width / height;
    let halfWidth = Math.max(maxX / frameFill, 0.01);
    let halfHeight = Math.max(maxY / frameFill, 0.01);
    if (halfWidth / halfHeight < aspect) {
      halfWidth = halfHeight * aspect;
    } else {
      halfHeight = halfWidth / aspect;
    }
    const depth = Math.max(maxZ - minZ, 1);
    c.left = -halfWidth;
    c.right = halfWidth;
    c.top = halfHeight;
    c.bottom = -halfHeight;
    c.zoom = 1;
    c.position.copy(center).addScaledVector(viewDir, -(depth + padding * 8));
    c.near = 0.001;
    c.far = Math.max(depth + padding * 16, 10);
    c.updateProjectionMatrix();
  } else {
    exportCamera.position.copy(center).addScaledVector(viewDir, -Math.max(size.length(), 1));
  }

  exportCamera.updateMatrixWorld(true);
  return exportCamera;
}

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise(resolve => canvas.toBlob(blob => resolve(blob), 'image/png'));
}

function cloneExportableSuperquadrics(sourceScene: THREE.Scene, targetScene: THREE.Scene): THREE.Mesh[] {
  const meshes: THREE.Mesh[] = [];
  sourceScene.traverse(object => {
    const sourceMesh = object as THREE.Mesh<THREE.BufferGeometry, THREE.Material | THREE.Material[]>;
    if (!sourceMesh.isMesh || sourceMesh.userData.sqExportable !== true) return;

    const controlLevel = sourceMesh.userData.sqControlLevel === 'low' ? 'low' : 'high';
    const material = new THREE.MeshStandardMaterial({
      color: controlLevel === 'low' ? RENDER_EXPORT_LOW_COLOR : RENDER_EXPORT_HIGH_COLOR,
      roughness: 0.45,
      metalness: 0.06,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(sourceMesh.geometry.clone(), material);
    sourceMesh.updateWorldMatrix(true, false);
    mesh.applyMatrix4(sourceMesh.matrixWorld);
    meshes.push(mesh);
    targetScene.add(mesh);
  });
  return meshes;
}

/** Registers WebGL canvas readback for AI Edit screenshots and clean SQ render exports. */
function ViewportCaptureRegister() {
  const gl = useThree(s => s.gl);
  const scene = useThree(s => s.scene);
  const camera = useThree(s => s.camera);
  useEffect(() => {
    setViewportCapture(() => {
      try {
        return gl.domElement.toDataURL('image/png');
      } catch {
        return null;
      }
    });
    setViewportRenderExport(async () => {
      const width = Math.max(1, gl.domElement.width);
      const height = Math.max(1, gl.domElement.height);
      const exportScene = new THREE.Scene();
      exportScene.background = new THREE.Color(RENDER_EXPORT_BG);
      exportScene.add(new THREE.AmbientLight(0xffffff, 0.7));
      const key = new THREE.DirectionalLight(0xffffff, 1.15);
      key.position.set(5, 8, 5);
      exportScene.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.35);
      fill.position.set(-3, -4, -2);
      exportScene.add(fill);

      const meshes = cloneExportableSuperquadrics(scene, exportScene);
      if (meshes.length === 0) return null;

      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
      });
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setPixelRatio(1);
      renderer.setSize(width, height, false);

      const exportCamera = fitExportCameraToMeshes(camera, width, height, meshes);
      renderer.render(exportScene, exportCamera);
      const blob = await canvasToPngBlob(renderer.domElement);

      meshes.forEach(mesh => {
        mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach(material => material.dispose());
        } else {
          mesh.material.dispose();
        }
      });
      renderer.dispose();
      return blob;
    });
    return () => {
      setViewportCapture(null);
      setViewportRenderExport(null);
    };
  }, [camera, gl, scene]);
  return null;
}

const PRIM_COLORS = [
  '#4fc3f7', '#81c784', '#ffb74d', '#e57373',
  '#ba68c8', '#4dd0e1', '#aed581', '#ffd54f',
  '#f06292', '#7986cb', '#a1887f', '#90a4ae',
];

type DragState = {
  plane: THREE.Plane;
  startHit: THREE.Vector3;
  startTranslation: [number, number, number];
  frozenNormScale: number;
  pointerId: number;
  undoPushed: boolean;
};

type LoadedGeneratedMesh = {
  object: THREE.Object3D;
  center: [number, number, number];
  fitScale: number;
};

type GeneratedMeshState = {
  sourceUrl: string;
  mesh: LoadedGeneratedMesh | null;
  error: string | null;
};

function prepareGeneratedMesh(gltf: GLTF): LoadedGeneratedMesh {
  const object = gltf.scene.clone(true);
  object.traverse(child => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (!mesh.material) {
      mesh.material = new THREE.MeshStandardMaterial({
        color: '#cbd5e1',
        roughness: 0.55,
        metalness: 0.05,
        side: THREE.DoubleSide,
      });
      return;
    }
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    materials.forEach(material => {
      material.side = THREE.DoubleSide;
      material.needsUpdate = true;
    });
  });

  const box = new THREE.Box3().setFromObject(object);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);
  const fitScale = Number.isFinite(maxDim) && maxDim > 1e-6 ? 2.4 / maxDim : 1;

  return {
    object,
    center: [center.x, center.y, center.z],
    fitScale,
  };
}

function GeneratedMesh({ source }: { source: MeshInspectionSource }) {
  const [state, setState] = useState<GeneratedMeshState>({
    sourceUrl: source.url,
    mesh: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    const loader = new GLTFLoader();
    loader.load(
      source.url,
      (gltf) => {
        if (cancelled) return;
        setState({
          sourceUrl: source.url,
          mesh: prepareGeneratedMesh(gltf),
          error: null,
        });
      },
      undefined,
      (err) => {
        if (cancelled) return;
        setState({
          sourceUrl: source.url,
          mesh: null,
          error: err instanceof Error ? err.message : 'Could not load generated mesh',
        });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [source.url]);

  const mesh = state.sourceUrl === source.url ? state.mesh : null;
  const error = state.sourceUrl === source.url ? state.error : null;

  if (error) {
    return (
      <Html center className="viewport-mesh-status">
        Mesh load failed
      </Html>
    );
  }

  if (!mesh) {
    return (
      <Html center className="viewport-mesh-status">
        Loading mesh
      </Html>
    );
  }

  const [cx, cy, cz] = mesh.center;
  const s = mesh.fitScale;
  return (
    <group scale={s} position={[-cx * s, -cy * s, -cz * s]}>
      <primitive object={mesh.object} />
    </group>
  );
}

function ResetCameraOnModeChange({ modeKey }: { modeKey: string }) {
  const camera = useThree(s => s.camera);
  const controls = useThree(s => s.controls as OrbitControlsImpl | null);

  useEffect(() => {
    camera.position.set(2.5, -2.2, 2.2);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [camera, controls, modeKey]);

  return null;
}

function SuperquadricMesh({
  primitive,
  index,
  selected,
  resolution,
  normCenter,
  normScale,
  showNormalized,
  showControlPreview,
}: {
  primitive: Primitive;
  index: number;
  selected: boolean;
  resolution: number;
  normCenter: [number, number, number];
  normScale: number;
  showNormalized: boolean;
  showControlPreview: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const outlineRef = useRef<THREE.LineSegments>(null);
  const dragRef = useRef<DragState | null>(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const ndcRef = useRef(new THREE.Vector2());
  const hitRef = useRef(new THREE.Vector3());

  const selectPrimitive = useStore(s => s.selectPrimitive);
  const updatePrimitiveLive = useStore(s => s.updatePrimitiveLive);
  const pushUndoSnapshot = useStore(s => s.pushUndoSnapshot);

  const camera = useThree(s => s.camera);
  const gl = useThree(s => s.gl);
  const controls = useThree(s => s.controls as OrbitControlsImpl | null);

  const { geometry, edgesGeometry } = useMemo(() => {
    const { vertices, indices } = createSuperquadricMesh(
      primitive.scales[0], primitive.scales[1], primitive.scales[2],
      primitive.shapes[0], primitive.shapes[1],
      primitive.rotation,
      primitive.translation,
      resolution,
      superflexDeformForPrimitive(primitive),
    );

    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(vertices.length);

    if (showNormalized) {
      for (let i = 0; i < vertices.length; i += 3) {
        positions[i] = (vertices[i] - normCenter[0]) * normScale;
        positions[i + 1] = (vertices[i + 1] - normCenter[1]) * normScale;
        positions[i + 2] = (vertices[i + 2] - normCenter[2]) * normScale;
      }
    } else {
      for (let i = 0; i < vertices.length; i++) {
        positions[i] = vertices[i];
      }
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();

    const edges = new THREE.EdgesGeometry(geo, 30);
    return { geometry: geo, edgesGeometry: edges };
  }, [primitive, resolution, normCenter, normScale, showNormalized]);

  const color = showControlPreview
    ? (primitive.controlLevel === 'high' ? '#2dd4bf' : '#f59e0b')
    : PRIM_COLORS[index % PRIM_COLORS.length];

  const windowDragCleanupRef = useRef<(() => void) | null>(null);

  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  const endDrag = useCallback(() => {
    windowDragCleanupRef.current?.();
    windowDragCleanupRef.current = null;
    dragRef.current = null;
    if (controls) controls.enabled = true;
  }, [controls]);

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    selectPrimitive(primitive.id);
    if (e.button !== 0) return;

    windowDragCleanupRef.current?.();
    windowDragCleanupRef.current = null;

    const cam = cameraRef.current;
    const planeNormal = new THREE.Vector3();
    cam.getWorldDirection(planeNormal);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(planeNormal, e.point);

    const pointerId = e.pointerId;
    const primId = primitive.id;
    const startTranslation = [...primitive.translation] as [number, number, number];
    const frozenNormScale = Math.max(normScale, 1e-6);

    dragRef.current = {
      plane,
      startHit: e.point.clone(),
      startTranslation,
      frozenNormScale,
      pointerId,
      undoPushed: false,
    };

    if (controls) controls.enabled = false;

    const applyMove = (clientX: number, clientY: number, buttons: number) => {
      const drag = dragRef.current;
      if (!drag || (buttons & 1) === 0) return;

      const rect = gl.domElement.getBoundingClientRect();
      ndcRef.current.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      ndcRef.current.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycasterRef.current.setFromCamera(ndcRef.current, cameraRef.current);

      const hit = hitRef.current;
      if (!raycasterRef.current.ray.intersectPlane(drag.plane, hit)) return;

      if (!drag.undoPushed) {
        pushUndoSnapshot();
        drag.undoPushed = true;
      }

      const s = drag.frozenNormScale;
      const dx = (hit.x - drag.startHit.x) / s;
      const dy = (hit.y - drag.startHit.y) / s;
      const dz = (hit.z - drag.startHit.z) / s;

      updatePrimitiveLive(primId, {
        translation: [
          drag.startTranslation[0] + dx,
          drag.startTranslation[1] + dy,
          drag.startTranslation[2] + dz,
        ],
      });
    };

    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      applyMove(ev.clientX, ev.clientY, ev.buttons);
    };

    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      endDrag();
    };

    windowDragCleanupRef.current = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  }, [
    controls,
    endDrag,
    gl.domElement,
    normScale,
    primitive.id,
    primitive.translation,
    pushUndoSnapshot,
    selectPrimitive,
    updatePrimitiveLive,
  ]);

  useEffect(() => {
    return () => {
      windowDragCleanupRef.current?.();
      windowDragCleanupRef.current = null;
      dragRef.current = null;
      if (controls) controls.enabled = true;
    };
  }, [controls]);

  useFrame(() => {
    if (outlineRef.current) {
      outlineRef.current.visible = selected;
    }
  });

  if (!primitive.visible) return null;

  return (
    <group>
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerDown={handlePointerDown}
        userData={{ sqExportable: true, sqControlLevel: primitive.controlLevel }}
      >
        <meshStandardMaterial
          color={color}
          roughness={0.45}
          metalness={0.1}
          side={THREE.DoubleSide}
          transparent={!selected}
          opacity={selected ? 1 : 0.85}
        />
      </mesh>
      <lineSegments ref={outlineRef} geometry={edgesGeometry} visible={selected}>
        <lineBasicMaterial color="#ffffff" linewidth={1} />
      </lineSegments>
    </group>
  );
}

function LowControlBBoxPreview({
  primitives,
  normCenter,
  normScale,
  showNormalized,
  marginFraction,
}: {
  primitives: Primitive[];
  normCenter: [number, number, number];
  normScale: number;
  showNormalized: boolean;
  marginFraction: number;
}) {
  const geometry = useMemo(() => {
    try {
      const { bbox } = buildLowControlBoundingBoxPrimitive(primitives, marginFraction);
      const corners: [number, number, number][] = [
        [bbox.min[0], bbox.min[1], bbox.min[2]],
        [bbox.max[0], bbox.min[1], bbox.min[2]],
        [bbox.max[0], bbox.max[1], bbox.min[2]],
        [bbox.min[0], bbox.max[1], bbox.min[2]],
        [bbox.min[0], bbox.min[1], bbox.max[2]],
        [bbox.max[0], bbox.min[1], bbox.max[2]],
        [bbox.max[0], bbox.max[1], bbox.max[2]],
        [bbox.min[0], bbox.max[1], bbox.max[2]],
      ];
      const edges = [
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
      ];
      const positions = new Float32Array(edges.length * 3);
      edges.forEach((cornerIndex, i) => {
        const corner = corners[cornerIndex]!;
        positions[i * 3] = showNormalized ? (corner[0] - normCenter[0]) * normScale : corner[0];
        positions[i * 3 + 1] = showNormalized ? (corner[1] - normCenter[1]) * normScale : corner[1];
        positions[i * 3 + 2] = showNormalized ? (corner[2] - normCenter[2]) * normScale : corner[2];
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      return geo;
    } catch {
      return null;
    }
  }, [primitives, normCenter, normScale, showNormalized, marginFraction]);

  if (!geometry) return null;
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#fbbf24" transparent opacity={0.95} />
    </lineSegments>
  );
}

function Scene() {
  const primitives = useStore(s => s.primitives);
  const meshInspection = useStore(s => s.meshInspection);
  const selectedId = useStore(s => s.selectedId);
  const resolution = useStore(s => s.previewResolution);
  const showNormalized = useStore(s => s.showNormalized);
  const showControlPreview = useStore(s => s.showControlPreview);
  const lowControlBBoxMargin = useStore(s => s.lowControlBBoxMargin);
  const selectPrimitive = useStore(s => s.selectPrimitive);

  // Pipeline normalization is useful for matching training/export conventions,
  // but recomputing the center on every drag makes the whole scene "swim".
  // Freeze the normalization transform for the duration of normalized mode.
  const frozenNormRef = useRef<{ center: [number, number, number]; scale: number } | null>(null);

  const computedNorm = useMemo(() => {
    if (!showNormalized || primitives.length === 0) return null;
    const allVerts = primitives
      .filter(p => p.visible)
      .map(p => {
        const { vertices } = createSuperquadricMesh(
          p.scales[0], p.scales[1], p.scales[2],
          p.shapes[0], p.shapes[1],
          p.rotation, p.translation, resolution,
          superflexDeformForPrimitive(p),
        );
        return vertices;
      });
    if (allVerts.length === 0) return null;
    return normalizeMergedVertices(allVerts);
  }, [primitives, resolution, showNormalized]);

  useEffect(() => {
    if (!showNormalized) {
      frozenNormRef.current = null;
      return;
    }
    // Only capture the normalization transform once when entering normalized mode.
    // This avoids "world drift" while dragging/animating primitives.
    if (computedNorm && frozenNormRef.current === null) {
      frozenNormRef.current = computedNorm;
    }
  }, [computedNorm, showNormalized]);

  const normCenter = frozenNormRef.current?.center ?? ([0, 0, 0] as [number, number, number]);
  const normScale = frozenNormRef.current?.scale ?? 1;

  // Match `gui/` conventions: Z-up world.
  const camera = useThree(s => s.camera);
  useEffect(() => {
    camera.up.set(0, 0, 1);
    camera.updateProjectionMatrix();
  }, [camera]);

  const handleMiss = useCallback(() => {
    selectPrimitive(null);
  }, [selectPrimitive]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1} />
      <directionalLight position={[-3, -4, -2]} intensity={0.3} />

      <Grid
        args={[20, 20]}
        // drei/Grid is XZ (Y-up). Rotate to XY for Z-up.
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#333a48"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#4a5568"
        fadeDistance={15}
        infiniteGrid
      />

      {/* Axes */}
      <axesHelper args={[1.5]} />

      <ResetCameraOnModeChange modeKey={meshInspection?.url ?? 'sqs'} />

      {meshInspection ? (
        <GeneratedMesh source={meshInspection} />
      ) : (
        <group onPointerMissed={handleMiss}>
          {primitives.map((p, i) => (
            <SuperquadricMesh
              key={p.id}
              primitive={p}
              index={i}
              selected={p.id === selectedId}
              resolution={resolution}
              normCenter={normCenter}
              normScale={normScale}
              showNormalized={showNormalized}
              showControlPreview={showControlPreview}
            />
          ))}
          {showControlPreview && (
            <LowControlBBoxPreview
              primitives={primitives}
              normCenter={normCenter}
              normScale={normScale}
              showNormalized={showNormalized}
              marginFraction={lowControlBBoxMargin}
            />
          )}
        </group>
      )}

      <OrbitControls makeDefault />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport labelColor="#fff" axisHeadScale={0.8} />
      </GizmoHelper>
    </>
  );
}

export default function Viewport() {
  const meshInspection = useStore(s => s.meshInspection);
  const setMeshInspection = useStore(s => s.setMeshInspection);

  return (
    <div className="viewport-shell">
      <Canvas
        // Z-up camera position (like the viser GUI).
        camera={{ position: [2.5, -2.2, 2.2], fov: 50, near: 0.01, far: 100 }}
        style={{ background: '#0e1014' }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
      >
        <ViewportCaptureRegister />
        <Scene />
      </Canvas>
      {meshInspection && (
        <div className="viewport-inspection-bar">
          <div className="viewport-inspection-title" title={meshInspection.path ?? meshInspection.url}>
            <span>Inspecting</span>
            <strong>{meshInspection.name}</strong>
          </div>
          <button
            type="button"
            className="viewport-back-btn"
            onClick={() => setMeshInspection(null)}
          >
            Back to SQs
          </button>
        </div>
      )}
    </div>
  );
}
