import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface AtomProps {
    position: [number, number, number];
    color: string;
    size?: number;
}

const Atom = ({ position, color, size = 0.4 }: AtomProps) => {
    return (
        <Sphere args={[size, 32, 32]} position={position}>
            <meshStandardMaterial
                color={color}
                roughness={0.2}
                metalness={0.8}
                emissive={color}
                emissiveIntensity={0.2}
            />
        </Sphere>
    );
};

interface BondProps {
    start: [number, number, number];
    end: [number, number, number];
}

const Bond = ({ start, end }: BondProps) => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3().subVectors(endVec, startVec);
    const length = direction.length();
    const position = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);

    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());

    return (
        <mesh position={position} quaternion={quaternion}>
            <cylinderGeometry args={[0.08, 0.08, length, 8]} />
            <meshStandardMaterial color="#666" transparent opacity={0.6} />
        </mesh>
    );
};

interface RotatingMoleculeProps {
    structure: any;
    activeFeature: string | null;
}

const RotatingMolecule = ({ structure, activeFeature }: RotatingMoleculeProps) => {
    const groupRef = useRef<THREE.Group>(null);

    useFrame((state) => {
        if (groupRef.current) {
            groupRef.current.rotation.y += 0.002;
            groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
        }
    });

    // Default to Benzene if no structure provided (e.g. loading state or error)
    const displayStructure = structure || {
        atoms: [
            { idx: 0, pos: [1.4, 0, 0], color: '#333' },
            { idx: 1, pos: [0.7, 1.2, 0], color: '#333' },
            { idx: 2, pos: [-0.7, 1.2, 0], color: '#333' },
            { idx: 3, pos: [-1.4, 0, 0], color: '#333' },
            { idx: 4, pos: [-0.7, -1.2, 0], color: '#333' },
            { idx: 5, pos: [0.7, -1.2, 0], color: '#333' },
        ],
        bonds: [
            { start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 3 },
            { start: 3, end: 4 }, { start: 4, end: 5 }, { start: 5, end: 0 }
        ]
    };

    // Calculate coloring based on weights
    // We use useMemo here because re-calculating colors for 50+ atoms every frame (if we didn't) would kill performance.
    const coloredAtoms = useMemo(() => {
        if (!displayStructure.atoms) return [];

        let maxWeight = 0;
        if (activeFeature && displayStructure.weights && displayStructure.weights[activeFeature]) {
            // Find max weight for normalization so the heatmap always looks "full"
            // even if the raw saliency scores are tiny.
            const weights = displayStructure.weights[activeFeature];
            maxWeight = Math.max(...weights, 0.000001);
        }

        return displayStructure.atoms.map((atom: any, i: number) => {
            let finalColor = atom.color || "#FFFFFF";

            // Apply the heatmap overlay if a feature is selected
            if (activeFeature && displayStructure.weights && displayStructure.weights[activeFeature]) {
                let weight = displayStructure.weights[activeFeature][i] || 0;

                // Scale locally: 0.0 -> Base Color, 1.0 (Max) -> Hot Pink
                const normalizedWeight = weight / maxWeight;

                const base = new THREE.Color(finalColor);
                const highlight = new THREE.Color("#ff00ff"); // Hot Pink for visibility

                // Linear interpolation looks best for this
                base.lerp(highlight, normalizedWeight);
                finalColor = "#" + base.getHexString();
            }
            return { ...atom, renderColor: finalColor };
        });
    }, [displayStructure, activeFeature]);

    // Calculate centroid to center the molecule
    const centroid = new THREE.Vector3();
    if (coloredAtoms.length > 0) {
        coloredAtoms.forEach((atom: any) => {
            centroid.add(new THREE.Vector3(...atom.pos));
        });
        centroid.divideScalar(coloredAtoms.length);
    }

    return (
        <group ref={groupRef} position={[-centroid.x, -centroid.y, -centroid.z]}>
            {coloredAtoms.map((atom: any, i: number) => (
                <Atom key={i} position={atom.pos} color={atom.renderColor} />
            ))}
            {displayStructure.bonds?.map((bond: any, i: number) => {
                const startAtom = coloredAtoms.find((a: any) => a.idx === bond.start);
                const endAtom = coloredAtoms.find((a: any) => a.idx === bond.end);

                if (startAtom && endAtom) {
                    return <Bond key={i} start={startAtom.pos} end={endAtom.pos} />;
                }

                // Fallback for simple index format
                if (coloredAtoms[bond.start] && coloredAtoms[bond.end]) {
                    return <Bond key={i} start={coloredAtoms[bond.start].pos} end={coloredAtoms[bond.end].pos} />;
                }
                return null;
            })}
        </group>
    );
};

export const MoleculeViewer = ({ structure, activeFeature = null }: { structure: any, activeFeature?: string | null }) => {
    return (
        <div className="w-full h-full relative">
            <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
                <ambientLight intensity={0.6} />
                <pointLight position={[10, 10, 10]} intensity={1.5} color="#3b82f6" />
                <pointLight position={[-10, -10, -10]} intensity={0.5} color="#a855f7" />

                <RotatingMolecule structure={structure} activeFeature={activeFeature} />

                <OrbitControls enableZoom={true} autoRotate autoRotateSpeed={0.5} />
            </Canvas>
            <div className="absolute bottom-4 right-4 text-xs text-gray-500 font-mono pointer-events-none">
                RENDER: R3F / THREE.JS
            </div>
        </div>
    );
};
