"use client";

import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef, useState } from "react";
import type { Mesh } from "three";
import type { Vec3 } from "@/lib/api";

export interface Track {
  positions: Vec3[][]; // frames x bodies x 3
  masses: number[];
  style: "observed" | "predicted" | "actual";
}

const COLORS = ["#e0a84a", "#7fa7c9"];

function Body({ frames, index, mass, color, playhead }: { frames: Vec3[][]; index: number; mass: number; color: string; playhead: React.MutableRefObject<number> }) {
  const ref = useRef<Mesh>(null);
  useFrame(() => {
    if (!ref.current || frames.length === 0) return;
    const frame = frames[Math.min(frames.length - 1, Math.floor(playhead.current))];
    const p = frame[index];
    ref.current.position.set(p[0], p[2], p[1]);
  });
  const radius = 0.05 + 0.04 * Math.cbrt(mass);
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial color={color} roughness={0.5} metalness={0.1} />
    </mesh>
  );
}

function Playhead({ length, playhead, speed }: { length: number; playhead: React.MutableRefObject<number>; speed: number }) {
  useFrame((_, delta) => {
    if (length === 0) return;
    playhead.current += delta * speed;
    if (playhead.current > length + 20) playhead.current = 0;
  });
  return null;
}

export function Scene({ tracks, speed = 40 }: { tracks: Track[]; speed?: number }) {
  const playhead = useRef(0);
  const [key] = useState(0);
  const longest = Math.max(0, ...tracks.map((t) => t.positions.length));
  const lines = useMemo(
    () =>
      tracks.flatMap((track, t) =>
        track.masses.map((_, body) => ({
          key: `${t}-${body}`,
          points: track.positions.map((frame) => [frame[body][0], frame[body][2], frame[body][1]] as [number, number, number]),
          color: COLORS[body % COLORS.length],
          dashed: track.style === "predicted",
          opacity: track.style === "observed" ? 0.9 : track.style === "actual" ? 1 : 0.8,
        })),
      ),
    [tracks],
  );
  const bodyTrack = tracks.find((t) => t.style !== "predicted") ?? tracks[0];
  return (
    <Canvas key={key} camera={{ position: [3.2, 2.6, 3.6], fov: 45 }} dpr={[1, 2]} style={{ background: "#0b0d10" }}>
      <ambientLight intensity={0.35} />
      <directionalLight position={[4, 6, 3]} intensity={0.9} />
      <gridHelper args={[8, 16, "#2a3037", "#171b20"]} />
      <axesHelper args={[0.6]} />
      {lines.map((line) =>
        line.points.length > 1 ? (
          <Line key={line.key} points={line.points} color={line.color} lineWidth={line.dashed ? 1.2 : 1.6} dashed={line.dashed} dashSize={0.06} gapSize={0.05} transparent opacity={line.opacity} />
        ) : null,
      )}
      {bodyTrack &&
        bodyTrack.masses.map((mass, index) => (
          <Body key={index} frames={bodyTrack.positions} index={index} mass={mass} color={COLORS[index % COLORS.length]} playhead={playhead} />
        ))}
      <Playhead length={longest} playhead={playhead} speed={speed} />
      <OrbitControls enableDamping makeDefault />
    </Canvas>
  );
}
