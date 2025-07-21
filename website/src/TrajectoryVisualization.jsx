import React, { useRef, useEffect, useContext, useState } from 'react';
import { WebSocketContext } from './contexts/WebSocketContext';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Ground from './Ground';
import * as THREE from 'three';
import { useControls } from 'leva';

// This component renders a wireframe camera (a pyramid/frustum) using line segments.
// It accepts both a position and a quaternion for orientation.
function CameraWireframe({ position, quaternion, scale = 1, color = 'yellow' }) {
  // Define vertices for the frustum:
  //   - Four segments from the apex (0,0,0) to the corners of the far plane at z = -1.
  //   - Four segments to connect the far plane corners.
  const vertices = [
    // Lines from the camera center (apex) to the far plane corners:
    0, 0, 0,   -0.5, -0.5, -1, // Bottom-left
    0, 0, 0,    0.5, -0.5, -1, // Bottom-right
    0, 0, 0,    0.5,  0.5, -1, // Top-right
    0, 0, 0,   -0.5,  0.5, -1, // Top-left

    // Far plane rectangle edges:
   -0.5, -0.5, -1,   0.5, -0.5, -1, // Bottom edge
    0.5, -0.5, -1,   0.5,  0.5, -1, // Right edge
    0.5,  0.5, -1,  -0.5,  0.5, -1, // Top edge
   -0.5,  0.5, -1,  -0.5, -0.5, -1  // Left edge
  ];

  // Apply scaling to the vertices so you can adjust the size of the frustum.
  const scaledVertices = vertices.map((v) => v * scale);

  return (
    // Passing both position and quaternion lets React Three Fiber apply these to the object.
    <lineSegments position={position} quaternion={quaternion}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={new Float32Array(scaledVertices)}
          count={scaledVertices.length / 3}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} />
    </lineSegments>
  );
}

function TrajectoryVisualization() {
  const { subscribe, unsubscribe } = useContext(WebSocketContext);
  const { scaleFactor } = useControls({
    scaleFactor: { value: 8, min: 1, max: 15 }
  });
  
  const lineRef = useRef();
  const [poses, setPoses] = useState([]);
  // Instead of storing only the position, we store both position and orientation.
  const [currentTransform, setCurrentTransform] = useState(null);

  useEffect(() => {
    // Handle incoming WebSocket messages.
    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'trajectory_update' && message.pose) {
          console.log('Trajectory update:', message.pose);
          setPoses((prevPoses) => [...prevPoses, message.pose]);
        } else if (message.type === 'restart') {
          console.log('Restart received, clearing trajectory');
          setPoses([]);
          setCurrentTransform(null);
          if (lineRef.current) {
            lineRef.current.geometry.setFromPoints([]);
          }
        }
      } catch (error) {
        console.error('Error parsing message in TrajectoryVisualization:', error);
      }
    };

    subscribe(handleMessage);
    return () => {
      unsubscribe(handleMessage);
    };
  }, [subscribe, unsubscribe]);

  useEffect(() => {
    if (!poses || poses.length === 0) return;

    // Create an array of positions for the trajectory line.
    const positions = poses
      .map((pose) => {
        const matrixData = pose.pose || pose.pose_matrix;
        if (!matrixData) return null;
        return new THREE.Vector3(
          matrixData[0][3] * scaleFactor,
          matrixData[1][3] * scaleFactor,
          matrixData[2][3] * scaleFactor
        );
      })
      .filter((p) => p !== null);

    // Update the line geometry with the trajectory positions.
    if (lineRef.current) {
      const geometry = lineRef.current.geometry;
      geometry.setFromPoints(positions);
      geometry.attributes.position.needsUpdate = true;
    }

    // Compute the current transform (position & orientation) from the last pose.
    const lastPose = poses[poses.length - 1];
    const matrixData = lastPose.pose || lastPose.pose_matrix;
    if (matrixData) {
      // Create a new Matrix4. Our pose data is assumed to be in row-major order,
      // so we need to convert it into column-major order for THREE.Matrix4.
      const m = new THREE.Matrix4();
      const colMajor = [];
      for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
          let value = matrixData[row][col];
          // Apply scaling only to the translation components.
          if (col === 3 && row < 3) {
            value *= scaleFactor;
          }
          colMajor.push(value);
        }
      }
      m.fromArray(colMajor);

      // Decompose the matrix into a position and a quaternion.
      const pos = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      const dummyScale = new THREE.Vector3();
      m.decompose(pos, quat, dummyScale);
      setCurrentTransform({ position: pos, quaternion: quat });
    }
  }, [poses, scaleFactor]);

  return (
    <Canvas
      shadows
      camera={{ fov: 45, near: 0.1, far: 1000, position: [10, 20, 20] }}
      style={{ backgroundColor: 'grey' }}
    >
      <OrbitControls makeDefault />
      <directionalLight position={[1, 2, 3]} intensity={4.5} />
      <ambientLight intensity={1.5} />
      <axesHelper args={[5]} />
      <Ground />
      <line ref={lineRef}>
        <bufferGeometry />
        <lineBasicMaterial color="hotpink" linewidth={3} />
      </line>
      {/* Render the camera wireframe at the current pose with orientation */}
      {currentTransform && (
        <CameraWireframe 
          position={currentTransform.position} 
          quaternion={currentTransform.quaternion} 
          scale={0.5} 
          color="yellow" 
        />
      )}
    </Canvas>
  );
}

export default TrajectoryVisualization;
