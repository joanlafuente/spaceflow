import { useRef, useMemo, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { useStore } from '../state/store';
import type { Primitive } from '../state/store';
import { createSuperquadricMesh, normalizeMergedVertices } from '../mesh/superquadric';
import { setViewportCapture } from '../state/viewportCapture';

/** Registers WebGL canvas readback for AI Edit viewport screenshots. */
function ViewportCaptureRegister() {
  const gl = useThree(s => s.gl);
  useEffect(() => {
    setViewportCapture(() => {
      try {
        return gl.domElement.toDataURL('image/png');
      } catch {
        return null;
      }
    });
    return () => setViewportCapture(null);
  }, [gl]);
  return null;
}

const PRIM_COLORS = [
  '#4fc3f7', '#81c784', '#ffb74d', '#e57373',
  '#ba68c8', '#4dd0e1', '#aed581', '#ffd54f',
  '#f06292', '#7986cb', '#a1887f', '#90a4ae',
];

function SuperquadricMesh({
  primitive,
  index,
  selected,
  resolution,
  normCenter,
  normScale,
  showNormalized,
}: {
  primitive: Primitive;
  index: number;
  selected: boolean;
  resolution: number;
  normCenter: [number, number, number];
  normScale: number;
  showNormalized: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const outlineRef = useRef<THREE.LineSegments>(null);
  const selectPrimitive = useStore(s => s.selectPrimitive);

  const { geometry, edgesGeometry } = useMemo(() => {
    const { vertices, indices } = createSuperquadricMesh(
      primitive.scales[0], primitive.scales[1], primitive.scales[2],
      primitive.shapes[0], primitive.shapes[1],
      primitive.rotation,
      primitive.translation,
      resolution,
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

  const color = PRIM_COLORS[index % PRIM_COLORS.length];

  const handleClick = useCallback((e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    selectPrimitive(primitive.id);
  }, [primitive.id, selectPrimitive]);

  useFrame(() => {
    if (outlineRef.current) {
      outlineRef.current.visible = selected;
    }
  });

  if (!primitive.visible) return null;

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} onClick={handleClick}>
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

function Scene() {
  const primitives = useStore(s => s.primitives);
  const selectedId = useStore(s => s.selectedId);
  const resolution = useStore(s => s.previewResolution);
  const showNormalized = useStore(s => s.showNormalized);
  const selectPrimitive = useStore(s => s.selectPrimitive);

  const { normCenter, normScale } = useMemo(() => {
    if (!showNormalized || primitives.length === 0) {
      return { normCenter: [0, 0, 0] as [number, number, number], normScale: 1 };
    }
    const allVerts = primitives
      .filter(p => p.visible)
      .map(p => {
        const { vertices } = createSuperquadricMesh(
          p.scales[0], p.scales[1], p.scales[2],
          p.shapes[0], p.shapes[1],
          p.rotation, p.translation, resolution,
        );
        return vertices;
      });
    if (allVerts.length === 0) {
      return { normCenter: [0, 0, 0] as [number, number, number], normScale: 1 };
    }
    const { center, scale } = normalizeMergedVertices(allVerts);
    return { normCenter: center, normScale: scale };
  }, [primitives, resolution, showNormalized]);

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
        position={[0, -0.5, 0]}
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
          />
        ))}
      </group>

      <OrbitControls makeDefault />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport labelColor="#fff" axisHeadScale={0.8} />
      </GizmoHelper>
    </>
  );
}

export default function Viewport() {
  return (
    <div style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
      <Canvas
        camera={{ position: [2.5, 2, 2.5], fov: 50, near: 0.01, far: 100 }}
        style={{ background: '#0e1014' }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
      >
        <ViewportCaptureRegister />
        <Scene />
      </Canvas>
    </div>
  );
}
