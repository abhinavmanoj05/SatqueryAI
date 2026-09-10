import { useRef, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'

// ---------------------------------------------------------------------------
// Individual geometry helpers
// ---------------------------------------------------------------------------

/** Cylinder mesh oriented along the X axis */
function HorizontalCylinder({
  position,
  radius = 0.22,
  length = 0.9,
  color = '#C8D8E8',
  segments = 14,
}: {
  position: [number, number, number]
  radius?: number
  length?: number
  color?: string
  segments?: number
}) {
  return (
    <mesh position={position} rotation={[0, 0, Math.PI / 2]}>
      <cylinderGeometry args={[radius, radius, length, segments]} />
      <meshPhysicalMaterial
        color={color}
        metalness={0.45}
        roughness={0.55}
        envMapIntensity={0.8}
      />
    </mesh>
  )
}

/** Thin box — used for truss arms and solar panel frames */
function Box({
  position,
  rotation,
  size,
  color = '#B8C8D8',
}: {
  position: [number, number, number]
  rotation?: [number, number, number]
  size: [number, number, number]
  color?: string
}) {
  return (
    <mesh position={position} rotation={rotation ?? [0, 0, 0]}>
      <boxGeometry args={size} />
      <meshPhysicalMaterial color={color} metalness={0.35} roughness={0.6} />
    </mesh>
  )
}

/** Flat solar panel quad with a dark blue PV colour */
function SolarPanel({
  position,
  rotation,
  width = 1.3,
  depth = 0.72,
}: {
  position: [number, number, number]
  rotation: [number, number, number]
  width?: number
  depth?: number
}) {
  return (
    <mesh position={position} rotation={rotation}>
      {/* Panel face */}
      <boxGeometry args={[width, 0.03, depth]} />
      <meshPhysicalMaterial
        color="#152C5A"
        metalness={0.15}
        roughness={0.35}
        reflectivity={0.5}
      />
    </mesh>
  )
}

/** Small docking node sphere */
function Node({
  position,
}: {
  position: [number, number, number]
}) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.14, 12, 12]} />
      <meshPhysicalMaterial color="#D8E6F0" metalness={0.6} roughness={0.4} />
    </mesh>
  )
}

// ---------------------------------------------------------------------------
// Full space station assembly
// ---------------------------------------------------------------------------
function SpaceStation({ mouseX, mouseY }: { mouseX: number; mouseY: number }) {
  const groupRef = useRef<THREE.Group>(null!)

  useFrame(() => {
    if (!groupRef.current) return
    // Slow continuous rotation around Y
    groupRef.current.rotation.y += 0.0015
    // Subtle mouse-driven tilt
    groupRef.current.rotation.x = THREE.MathUtils.lerp(
      groupRef.current.rotation.x,
      mouseY * 0.25,
      0.04,
    )
    groupRef.current.rotation.z = THREE.MathUtils.lerp(
      groupRef.current.rotation.z,
      -mouseX * 0.12,
      0.04,
    )
  })

  // Shared materials via useMemo to avoid re-creating per render
  const moduleMat = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: '#C8D8E8',
        metalness: 0.45,
        roughness: 0.55,
      }),
    [],
  )

  return (
    <group ref={groupRef}>
      {/* ── CENTRAL HUB ─────────────────────────────────────────── */}
      <mesh rotation={[0, 0, Math.PI / 2]} material={moduleMat}>
        <cylinderGeometry args={[0.38, 0.38, 1.3, 18]} />
      </mesh>

      {/* ── LEFT WING TRUSS ARM ─────────────────────────────────── */}
      <Box position={[-2.1, 0.48, 0]} size={[2.8, 0.05, 0.05]} color="#A0B5C8" />
      <Box position={[2.1, 0.48, 0]} size={[2.8, 0.05, 0.05]} color="#A0B5C8" />

      {/* ── RADIAL TUNNEL CONNECTORS (left / right) ─────────────── */}
      <HorizontalCylinder position={[-0.95, 0, 0]} radius={0.18} length={0.75} />
      <HorizontalCylinder position={[0.95, 0, 0]} radius={0.18} length={0.75} />

      {/* ── END MODULES (larger cylinders at each wing) ─────────── */}
      <HorizontalCylinder position={[-1.7, 0, 0]} radius={0.26} length={0.85} color="#BED0E0" />
      <HorizontalCylinder position={[1.7, 0, 0]} radius={0.26} length={0.85} color="#BED0E0" />

      {/* ── RESEARCH MODULE (front, perpendicular to hub) ───────── */}
      <mesh position={[0, 0, 0.85]} material={moduleMat}>
        <cylinderGeometry args={[0.22, 0.22, 0.8, 14]} />
      </mesh>

      {/* ── DOCKING NODES ───────────────────────────────────────── */}
      <Node position={[0, 0.52, 0]} />
      <Node position={[0, -0.52, 0]} />
      <Node position={[-2.15, 0, 0]} />
      <Node position={[2.15, 0, 0]} />

      {/* ── SOLAR PANELS — 4 arrays (2 per truss side) ─────────── */}
      {/* Top-left outer */}
      <SolarPanel position={[-2.6, 0.82, 0]} rotation={[0, 0, 0]} width={1.35} depth={0.7} />
      {/* Top-left inner */}
      <SolarPanel position={[-1.3, 0.82, 0]} rotation={[0, 0, 0]} width={1.1} depth={0.65} />
      {/* Top-right inner */}
      <SolarPanel position={[1.3, 0.82, 0]} rotation={[0, 0, 0]} width={1.1} depth={0.65} />
      {/* Top-right outer */}
      <SolarPanel position={[2.6, 0.82, 0]} rotation={[0, 0, 0]} width={1.35} depth={0.7} />

      {/* ── THERMAL RADIATORS (below truss) ─────────────────────── */}
      <Box position={[-1.2, -0.55, 0.3]} size={[1.0, 0.03, 0.5]} color="#C0D0DC" />
      <Box position={[1.2, -0.55, 0.3]} size={[1.0, 0.03, 0.5]} color="#C0D0DC" />

      {/* ── COMMUNICATIONS DISH ─────────────────────────────────── */}
      <mesh position={[0.3, 0.7, 0.25]} rotation={[Math.PI / 5, 0, 0]}>
        <cylinderGeometry args={[0.22, 0.0, 0.12, 16, 1, true]} />
        <meshPhysicalMaterial
          color="#E0EAF4"
          metalness={0.5}
          roughness={0.4}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  )
}

// ---------------------------------------------------------------------------
// Scene — lights + station
// ---------------------------------------------------------------------------
function Scene({ mouseX, mouseY }: { mouseX: number; mouseY: number }) {
  const { gl } = useThree()

  // Light background hint via renderer clear
  gl.setClearColor(0x00000000, 0) // transparent

  return (
    <>
      {/* Key light — from upper right */}
      <directionalLight
        position={[5, 8, 4]}
        intensity={2.4}
        color="#FFFFFF"
        castShadow={false}
      />
      {/* Fill light — left bounce */}
      <directionalLight position={[-4, 2, 2]} intensity={0.7} color="#C8D8F0" />
      {/* Ambient */}
      <ambientLight intensity={0.55} color="#F0F4FF" />
      {/* Subtle back rim light */}
      <directionalLight position={[0, -3, -5]} intensity={0.3} color="#D0DCEC" />

      <PerspectiveCamera makeDefault position={[0, 1.6, 7]} fov={38} />
      <SpaceStation mouseX={mouseX} mouseY={mouseY} />
    </>
  )
}

// ---------------------------------------------------------------------------
// Public export — wraps the scene in a <Canvas>
// ---------------------------------------------------------------------------
export default function BASModel({ mouseX = 0, mouseY = 0 }: { mouseX?: number; mouseY?: number }) {
  return (
    <Canvas
      className="bas-canvas"
      gl={{ antialias: true, alpha: true }}
      dpr={[1, 2]}
      aria-hidden="true"
    >
      <Scene mouseX={mouseX} mouseY={mouseY} />
    </Canvas>
  )
}
