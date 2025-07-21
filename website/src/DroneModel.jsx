import React from 'react';
import PropTypes from 'prop-types';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader';
import { DDSLoader } from 'three-stdlib';
import { useControls, folder } from 'leva';

THREE.DefaultLoadingManager.addHandler(/\.dds$/i, new DDSLoader());

function Model({ scale }) {
  const materials = useLoader(MTLLoader, './models/Aerial_Explorer/model.mtl');
  const obj = useLoader(OBJLoader, './models/Aerial_Explorer/model.obj', (loader) => {
    materials.preload();
    loader.setMaterials(materials);
  });
  return <primitive object={obj} scale={scale || 1}/>;
}

Model.propTypes = {
  scale: PropTypes.number
};

export default function DroneModel() {
  const { scale, opacity } = useControls('Drone', () => ({
    'Drone Settings': folder({
      scale: { value: getScalingFactor(1.997, 0.251), min: 0.005, max: 1, step: 0.005 },
      opacity: { value: 0.5, min: 0, max: 1, step: 0.01 }
    }, { collapsed: true })
  }));

  // Ensure scale is a valid number
  const safeScale = Number.isFinite(scale) ? scale : 0.001;
  const sphereRadius = Math.max(0.001, safeScale * 2);

  return (
    <group>
      <Model scale={safeScale} />
      <mesh>
        <sphereGeometry args={[sphereRadius, 32, 32]} />
        <meshStandardMaterial color="blue" transparent opacity={opacity} />
      </mesh>
    </group>
  );
}

function getScalingFactor(originalSize, targetSize) {
  if (originalSize === 0) {
    throw new Error("Original size cannot be zero.");
  }
  return targetSize / originalSize;
}