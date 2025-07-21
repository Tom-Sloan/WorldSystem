import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

/**
 * ViewCube widget:
 *   - Placed at bottom‐right with responsive width/height (15vw).
 *   - A separate <Canvas> that draws a solid‐colored cube with text labels.
 *   - When you drag it, it rotates the *model* via modelRef (doesn't affect OrbitControls).
 *   - When you click a face label, it snaps the *model* to that orientation.
 *   - No background color for the corner widget (transparent).
 */
export default function ViewCube({ modelRef }) {
  // Create a ref to store the main scene's camera controls
  const mainControlsRef = useRef(null);

  return (
    <div
      style={{
        position: "fixed",
        bottom: "2vh",
        left: "2vw",
        width: "120px",
        height: "120px",
        border: "1px solid #aaa",
        background: "#fff",
        borderRadius: 4,
        overflow: "hidden",
        zIndex: 1000
      }}
    >
      <Canvas
        style={{ background: "none" }}
        gl={{ alpha: true }}
        orthographic
        camera={{
          zoom: 15,
          position: [50, 50, 50],
          up: [0, 1, 0],
          near: -100,
          far: 100,
        }}
      >
        <MiniCubeScene modelRef={modelRef} mainControlsRef={mainControlsRef} />
      </Canvas>
    </div>
  );
}

/**
 * MiniCubeScene:
 *  - We store local angles (azimuth, polar).
 *  - Drags update those angles, which we then apply to modelRef.current.rotation.
 */
function MiniCubeScene({ modelRef, mainControlsRef }) {
  const cubeControlsRef = useRef(null);

  // Create textures for each face
  const createFaceTexture = (text) => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');
    
    // Fill background
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 512, 512);
    
    // Draw text
    context.fillStyle = 'black';
    context.font = 'bold 80px Arial';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 256, 256);
    
    return new THREE.CanvasTexture(canvas);
  };

  const faceNames = ['RIGHT', 'LEFT', 'TOP', 'BOTTOM', 'FRONT', 'BACK'];

  // Create materials array with click handlers
  const materials = [
    new THREE.MeshBasicMaterial({ map: createFaceTexture('RIGHT') }), // Right
    new THREE.MeshBasicMaterial({ map: createFaceTexture('LEFT') }),  // Left
    new THREE.MeshBasicMaterial({ map: createFaceTexture('TOP') }),   // Top
    new THREE.MeshBasicMaterial({ map: createFaceTexture('BOTTOM') }), // Bottom
    new THREE.MeshBasicMaterial({ map: createFaceTexture('FRONT') }), // Front
    new THREE.MeshBasicMaterial({ map: createFaceTexture('BACK') })   // Back
  ];

  // Update ViewCube rotation to match the main scene camera
  useFrame(({ camera: mainCamera }) => {
    if (!modelRef?.current) return;

    // Get the main scene camera's position
    const cameraPosition = new THREE.Vector3();
    mainCamera.getWorldPosition(cameraPosition);

    // Calculate spherical coordinates from camera position
    const spherical = new THREE.Spherical();
    spherical.setFromVector3(cameraPosition);

    // Update ViewCube controls to match main camera
    if (cubeControlsRef.current) {
      cubeControlsRef.current.setAzimuthalAngle(spherical.theta);
      cubeControlsRef.current.setPolarAngle(spherical.phi);
    }

    // Update model rotation
    if (modelRef.current) {
      modelRef.current.rotation.copy(cubeControlsRef.current.object.rotation);
    }
  });

  const handleFaceClick = (event, face) => {
    event.stopPropagation();
    console.log(`Clicked face: ${face}`);

    if (!cubeControlsRef.current) return;

    // Set rotation based on clicked face - adjusted for direct face views
    switch (face) {
      case 'TOP':
        cubeControlsRef.current.setAzimuthalAngle(0);
        cubeControlsRef.current.setPolarAngle(0.001);
        break;
      case 'BOTTOM':
        cubeControlsRef.current.setAzimuthalAngle(0);
        cubeControlsRef.current.setPolarAngle(Math.PI - 0.001);
        break;
      case 'FRONT':
        cubeControlsRef.current.setAzimuthalAngle(0);
        cubeControlsRef.current.setPolarAngle(Math.PI / 2);
        break;
      case 'BACK':
        cubeControlsRef.current.setAzimuthalAngle(Math.PI);
        cubeControlsRef.current.setPolarAngle(Math.PI / 2);
        break;
      case 'LEFT':
        cubeControlsRef.current.setAzimuthalAngle(-Math.PI / 2);
        cubeControlsRef.current.setPolarAngle(Math.PI / 2);
        break;
      case 'RIGHT':
        cubeControlsRef.current.setAzimuthalAngle(Math.PI / 2);
        cubeControlsRef.current.setPolarAngle(Math.PI / 2);
        break;
      default:
        break;
    }
  };

  return (
    <>
      <OrbitControls 
        ref={cubeControlsRef} 
        enablePan={false} 
        makeDefault 
      />
      <ambientLight intensity={1} />
      <mesh 
        scale={4}
        onPointerMove={(e) => {
          e.stopPropagation();
          if (e.faceIndex !== undefined) {
            const faceIndex = Math.floor(e.faceIndex / 2);
            e.object.cursor = 'pointer';
          }
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          if (e.faceIndex !== undefined) {
            const faceIndex = Math.floor(e.faceIndex / 2);
            handleFaceClick(e, faceNames[faceIndex]);
          }
        }}
      >
        <boxGeometry />
        <meshStandardMaterial attach="material" map={null} />
        {materials.map((material, index) => (
          <primitive 
            key={index} 
            object={material} 
            attach={`material-${index}`}
          />
        ))}
      </mesh>
    </>
  );
}
