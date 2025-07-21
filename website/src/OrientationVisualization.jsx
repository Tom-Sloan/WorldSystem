import React, { useRef, useEffect, useContext, useState } from 'react';
import { WebSocketContext } from './contexts/WebSocketContext';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader';

// DroneModel component (unchanged)
const DroneModel = React.forwardRef(({ position, scale }, ref) => {
  const [model, setModel] = useState(null);
  const internalRef = useRef();

  useEffect(() => {
    const mtlLoader = new MTLLoader();
    const objLoader = new OBJLoader();

    mtlLoader.load(
      '/models/Aerial_Explorer/model.mtl',
      (materials) => {
        materials.preload();
        objLoader.setMaterials(materials);
        objLoader.load(
          '/models/Aerial_Explorer/model.obj',
          (object) => {
            setModel(object);
          },
          undefined,
          (error) => {
            console.error('Error loading OBJ file:', error);
          }
        );
      },
      undefined,
      (error) => {
        console.error('Error loading MTL file:', error);
      }
    );

    return () => {
      if (model) {
        model.traverse((child) => {
          if (child.material) {
            child.material.dispose();
          }
          if (child.geometry) {
            child.geometry.dispose();
          }
        });
      }
    };
  }, []);

  // Combine internal ref with forwarded ref
  useEffect(() => {
    if (ref) {
      ref.current = internalRef.current;
    }
  }, [ref, internalRef.current]);

  if (!model) return null;

  return (
    <mesh ref={internalRef}>
      <primitive 
        object={model} 
        position={position}
        scale={scale}
        rotation={[0, 0, 0]}
      />
    </mesh>
  );
});
DroneModel.displayName = 'DroneModel';

function OrientationVisualization() {
  const { subscribe, unsubscribe } = useContext(WebSocketContext);
  const trajectoryRef = useRef();
  const imuRef = useRef();
  const [trajectoryPose, setTrajectoryPose] = useState(null);
  const [imuData, setImuData] = useState(null);

  // Refs to keep track of our IMU fusion state
  const lastImuTimestampRef = useRef(null);
  const imuOrientationRef = useRef(new THREE.Quaternion());

  // Subscribe to WebSocket messages
  useEffect(() => {
    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        switch (message.type) {
          case 'trajectory_update':
            if (message.pose) {
              setTrajectoryPose(message.pose);
            }
            break;
          case 'imu_data':
            // console.log('IMU data received:', message.imu_data);
            if (message.imu_data) {
              setImuData(message.imu_data);
            }
            break;
          case 'restart':
            console.log('Restart received, resetting orientation visualizations');
            setTrajectoryPose(null);
            setImuData(null);
            lastImuTimestampRef.current = null;
            imuOrientationRef.current.identity();
            if (trajectoryRef.current) {
              trajectoryRef.current.quaternion.set(0, 0, 0, 1);
            }
            if (imuRef.current) {
              imuRef.current.quaternion.set(0, 0, 0, 1);
            }
            break;
          default:
            break;
        }
      } catch (error) {
        console.error('Error processing message:', error);
      }
    };

    subscribe(handleMessage);
    return () => unsubscribe(handleMessage);
  }, [subscribe, unsubscribe]);

  // Update the trajectory model from trajectory_pose (unchanged)
  useEffect(() => {
    if (!trajectoryPose || !trajectoryRef.current) return;

    // Use either "pose" or "pose_matrix" (flattened) to update orientation
    const poseMatrix = trajectoryPose.pose || trajectoryPose.pose_matrix;
    if (!poseMatrix) return;

    const matrix = new THREE.Matrix4().fromArray(poseMatrix.flat());
    const quaternion = new THREE.Quaternion();
    matrix.decompose(new THREE.Vector3(), quaternion, new THREE.Vector3());
    trajectoryRef.current.quaternion.copy(quaternion);
  }, [trajectoryPose]);

  // Update the IMU visualization using the new imu_data format
  useEffect(() => {
    if (!imuData || !imuRef.current) return;

    // The expected structure:
    // {
    //   timestamp: 1701143016948,
    //   accelerometer: { x: -0.51235914, y: -0.088585466, z: 9.57561 },
    //   gyroscope: { x: -0.002130529, y: 0, z: 0.0010652645 }
    // }
    const { timestamp, accelerometer, gyroscope } = imuData;
    if (!timestamp || !accelerometer || !gyroscope) {
      console.warn("Incomplete IMU data:", imuData);
      return;
    }

    // Initialize our state on the first reading
    if (lastImuTimestampRef.current === null) {
      lastImuTimestampRef.current = timestamp;
      // Compute initial roll and pitch from the accelerometer (assuming only gravity)
      const { x: ax, y: ay, z: az } = accelerometer;
      const rollAcc = Math.atan2(ay, az);
      const pitchAcc = Math.atan2(-ax, Math.sqrt(ay * ay + az * az));
      const initialEuler = new THREE.Euler(rollAcc, pitchAcc, 0, 'XYZ');
      const initialQuat = new THREE.Quaternion().setFromEuler(initialEuler);
      imuOrientationRef.current.copy(initialQuat);
      imuRef.current.quaternion.copy(initialQuat);
      return;
    }

    // Compute the time delta (in seconds) between the current and last update
    // Compute the time delta in seconds and clamp it to a maximum value.
    let dt = (timestamp - lastImuTimestampRef.current) / 1e6;
    const maxDt = 0.05; // maximum allowed dt in seconds (adjust as needed)
    dt = Math.min(dt, maxDt);
    lastImuTimestampRef.current = timestamp;


    // --- Gyroscope integration ---
    // We assume gyroscope readings are in rad/s.
    const { x: gx, y: gy, z: gz } = gyroscope;
    const omega = new THREE.Vector3(gx, gy, gz);
    const omegaMag = omega.length();
    let deltaQuat = new THREE.Quaternion();
    if (omegaMag * dt > 0) {
      const axis = omega.clone().normalize();
      deltaQuat.setFromAxisAngle(axis, omegaMag * dt);
    } else {
      deltaQuat.identity();
    }

    // Update our orientation by integrating the gyroscope (local rotation)
    let newOrientation = imuOrientationRef.current.clone().multiply(deltaQuat);

    // --- Accelerometer estimation ---
    // Under the assumption that acceleration is mostly due to gravity,
    // we can estimate the roll and pitch.
    const { x: ax, y: ay, z: az } = accelerometer;
    const rollAcc = Math.atan2(ay, az);
    const pitchAcc = Math.atan2(-ax, Math.sqrt(ay * ay + az * az));

    // Extract yaw from the integrated (gyroscope) orientation
    const eulerGyro = new THREE.Euler().setFromQuaternion(newOrientation, 'XYZ');
    const yawGyro = eulerGyro.z;

    // Construct a quaternion from the accelerometer’s roll/pitch and the gyro’s yaw
    const fusedEuler = new THREE.Euler(rollAcc, pitchAcc, yawGyro, 'XYZ');
    const fusedQuat = new THREE.Quaternion().setFromEuler(fusedEuler);

    // --- Complementary filter ---
    // Blend the gyroscope integration with the accelerometer correction.
    // The "alpha" value determines the weight given to the gyroscope (high alpha) vs. accelerometer.
    const alpha = 0.98;
    newOrientation.slerp(fusedQuat, 1 - alpha);

    // Save and apply the updated orientation
    imuOrientationRef.current.copy(newOrientation);
    imuRef.current.quaternion.copy(newOrientation);
  }, [imuData]);

  return (
    <Canvas
      shadows
      camera={{ fov: 45, near: 0.1, far: 1000, position: [0, 20, 20] }}
      style={{ backgroundColor: 'grey' }}
    >
      {/* Trajectory Visualization */}
      <group position={[-7, 0, 0]}>
        <DroneModel ref={trajectoryRef} />
        <axesHelper args={[5]} />
        <Text
          position={[0, -3, 0]}
          fontSize={0.5}
          color="white"
          anchorX="center"
          anchorY="top"
        >
          Trajectory Pose
        </Text>
      </group>

      {/* IMU Visualization */}
      <group position={[7, 0, 0]}>
        <DroneModel ref={imuRef} />
        <axesHelper args={[5]} />
        <Text
          position={[0, -3, 0]}
          fontSize={0.5}
          color="white"
          anchorX="center"
          anchorY="top"
        >
          IMU Data
        </Text>
      </group>

      {/* Common scene elements */}
      <OrbitControls makeDefault />
      <directionalLight position={[1, 2, 3]} intensity={4.5} />
      <ambientLight intensity={1.5} />
    </Canvas>
  );
}

export default OrientationVisualization;
