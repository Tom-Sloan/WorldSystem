import React, {
  forwardRef,
  Suspense,
  useEffect,
  useRef
} from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import Box from './Box';
import Route from './Route';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { DDSLoader } from 'three/examples/jsm/loaders/DDSLoader';
import { folder, useControls } from 'leva';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader';
import Drone from './Drone';
import PropTypes from 'prop-types';

THREE.DefaultLoadingManager.addHandler(/\.dds$/i, new DDSLoader());

// We convert this component into a forwardRef so that <App> can do:
//   const roomRef = useRef(null);
//   <RoomModel ref={roomRef} />
// Then any parent or sibling can rotate the model by accessing roomRef.current.rotation, etc.
const RoomModel = forwardRef(function RoomModel(props, forwardedRef) {
  const { opacity, file_name } = useControls({
    'Room Settings': folder({
      opacity: { value: 0.7, min: 0, max: 1, step: 0.01 },
      file_name: { options: ['Room', 'InteriorTest', 'room_new', 'interior', 'lab'] }
    }, { collapsed: true })
  });

  const boxHelperRef = useRef();

  // We define a child component that actually loads the OBJ/MTL, etc.
  // Then we pass 'forwardedRef' down so the <mesh> uses it.
  const Scene = React.memo(({ opacity, file_name, forwardedRef }) => {
    if (import.meta.env.PROD) {
      console.log("Loading Room");
    }
    const objUrl = `${import.meta.env.VITE_API_URL}/obj?file_name=${file_name}.obj`;
    const mtlUrl = `${import.meta.env.VITE_API_URL}/obj?file_name=${file_name}.mtl`;

    // Try to load MTL first, but make it optional
    let materials = null;
    try {
      materials = useLoader(MTLLoader, mtlUrl, undefined, (error) => {
        console.log('MTL file not found, continuing without materials:', error);
        return null;
      });
    } catch (error) {
      console.log('MTL file not found, continuing without materials:', error);
    }

    const obj = useLoader(OBJLoader, objUrl, loader => {
      if (materials) {
        materials.preload();
        loader.setMaterials(materials);
      }
    });

    useEffect(() => {
      obj.traverse((child) => {
        if (child.isMesh) {
          child.material.transparent = true;
          child.material.opacity = opacity;
          child.material.side = THREE.DoubleSide;
        }
      });
    }, [obj, opacity]);

    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    const min = box.min;
    const max = box.max;
    const center = box.getCenter(new THREE.Vector3());

    const dimensions = {
      size,
      min,
      max
    };
    console.log("Loaded Room");
    console.log("Dimensions: ", dimensions);
    console.log('Center: ', center);

    // Scale so that either x or z dimension is about 10 units
    const scaleFactor = 10 / Math.min(size.x, size.z);

    // Create BoxHelper for the bounding box
    const boxHelper = new THREE.BoxHelper(obj, 0xff0000);
    boxHelperRef.current = boxHelper;

    return (
      <mesh ref={forwardedRef} scale={[scaleFactor, scaleFactor, scaleFactor]}>
        <primitive object={obj} />
        <primitive object={boxHelper} />
        <Route room_scale={scaleFactor} dimensions={dimensions} />
        <Drone room_scale={scaleFactor} dimensions={dimensions} />
      </mesh>
    );
  });
  Scene.displayName = 'Scene';
  Scene.propTypes = {
    opacity: PropTypes.number.isRequired,
    file_name: PropTypes.string.isRequired,
    forwardedRef: PropTypes.object
  };

  useFrame((state, delta) => {
    if (boxHelperRef.current) {
      // Pulsate the color of the box helper
      const time = state.clock.getElapsedTime();
      const r = (Math.sin(time * 2) + 1) / 2;
      const g = (Math.sin(time * 0.7) + 1) / 2;
      const b = (Math.sin(time * 1.3) + 1) / 2;
      boxHelperRef.current.material.color.setRGB(r, g, b);
    }
  });

  return (
    <Suspense fallback={<Box />}>
      <Scene opacity={opacity} file_name={file_name} forwardedRef={forwardedRef} />
    </Suspense>
  );
});

export default RoomModel;
