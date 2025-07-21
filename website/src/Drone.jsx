import React, { useEffect, useState, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useControls, folder } from "leva";
import { Suspense } from "react";
import DroneModel from "./DroneModel";
import Box from "./Box";
import PropTypes from 'prop-types';

const Drone = ({ room_scale, dimensions }) => {
  const droneRef = useRef();
  const [dronePosition, setDronePosition] = useState({
    old: new THREE.Vector3(0, 0, 0),
    current: new THREE.Vector3(0, 0, 0),
  });
  const [pythonWorldDimensions, setPythonWorldDimensions] = useState({x:115, y:103, z:178});
  
  // Define prop types for the dimensions object
  Drone.propTypes = {
    room_scale: PropTypes.number.isRequired,
    dimensions: PropTypes.shape({
      min: PropTypes.shape({
        x: PropTypes.number.isRequired,
        y: PropTypes.number.isRequired,
        z: PropTypes.number.isRequired
      }).isRequired,
      max: PropTypes.shape({
        x: PropTypes.number.isRequired,
        y: PropTypes.number.isRequired,
        z: PropTypes.number.isRequired
      }).isRequired
    }).isRequired
  };

  const [{ mode, X_curr, Y_curr, Z_curr }, set, get] = useControls("Drone", () => ({
    'Drone Position': folder({
      mode: {
        value: 'stopped',
        options: { Torus: 'torus', "Current Location": 'current_location', Stopped: 'stopped' }
      },
      X_curr: { value: 0, min: dimensions.min.x * room_scale, max: dimensions.max.x * room_scale, step: 0.1 },
      Y_curr: { value: 0, min: dimensions.min.y * room_scale, max: dimensions.max.y * room_scale, step: 0.1 },
      Z_curr: { value: 0, min: dimensions.min.z * room_scale, max: dimensions.max.z * room_scale, step: 0.1 }
    }, { collapsed: true })
  }));

  useEffect(() => {
    const fetchCoordinates = async () => {
      try {
        console.log("Drone Position: ", dronePosition)  
        console.log(pythonWorldDimensions)
        let response;
        if (mode === 'torus') {
          response = await fetch(`${import.meta.env.VITE_API_URL}/torus`);
        } else if (mode === 'current_location') {
          response = await fetch(`${import.meta.env.VITE_API_URL}/current_location`);
        } else {
          return; // Do nothing if mode is 'stopped'
        }
        let data = await response.json();
        console.log("Data: ", data)
        if (mode === 'current_location') {
          if (data.dimensions.x !== pythonWorldDimensions.x || data.dimensions.y !== pythonWorldDimensions.y || data.dimensions.z !== pythonWorldDimensions.z) {
            console.log("Setting python world dimensions", data.dimensions)
            setPythonWorldDimensions(new THREE.Vector3(data.dimensions[0], data.dimensions[1], data.dimensions[2]));
          }
          data = transformPoints_python_to_room([data.point], data.dimensions)[0];
          console.log("Transformed data", data)
        }

        if (data) {
          console.log("Setting drone position")
          setDronePosition((prevValue) => {
            return {
              old: prevValue.current,
              current: new THREE.Vector3(data.x, data.y, data.z),
            };
          });
        }
      } catch (error) {
        console.error('Error fetching coordinates:', error);
      }
    };

    if (mode !== 'stopped') {
      const intervalId = setInterval(fetchCoordinates, 50);
      return () => clearInterval(intervalId);
    }
  }, [mode]);

  useEffect(() => {
    const sendCoordinates = async () => {
      try {
        if (!pythonWorldDimensions) return; // Add safety check
        
        console.log("getting coordinates: ", get("X_curr"), get("Y_curr"), get("Z_curr"), pythonWorldDimensions);
        const curr_points_scale_to_python = transformPoints_room_to_python(
          [{x: get("X_curr"), y: get('Y_curr'), z: get('Z_curr')}], 
          [pythonWorldDimensions.x, pythonWorldDimensions.y, pythonWorldDimensions.z]
        )[0];
        
        console.log("Sending coordinates", curr_points_scale_to_python);
        // const response = await fetch(`${import.meta.env.VITE_API_URL}/set_current_location`, {
        //   method: 'POST',
        //   headers: {
        //     'Content-Type': 'application/json',
        //   },
        //   body: JSON.stringify({ 
        //     x: Math.round(curr_points_scale_to_python.x),
        //     y: Math.round(curr_points_scale_to_python.y),
        //     z: Math.round(curr_points_scale_to_python.z)
        //   }),
        // });
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
      } catch (error) {
        console.error('Error sending coordinates:', error);
      }
    };

    sendCoordinates();
  }, [X_curr, Y_curr, Z_curr, pythonWorldDimensions]); // Add pythonWorldDimensions to dependency array

  function transformPoints_python_to_room(points, pythonGridDimensions) {
    let pythonGrid = {
      min: new THREE.Vector3(0, 0, 0),
      max: new THREE.Vector3(...pythonGridDimensions),
    };
    let scaled_dimensions = {
      min: new THREE.Vector3(dimensions.min.x * room_scale, dimensions.min.y * room_scale, dimensions.min.z * room_scale),
      max: new THREE.Vector3(dimensions.max.x * room_scale, dimensions.max.y * room_scale, dimensions.max.z * room_scale),
    };
    return points.map(point => mapPoints(point, pythonGrid, scaled_dimensions));
  }

  function transformPoints_room_to_python(points, pythonGridDimensions) {
    let pythonGrid = {
      min: new THREE.Vector3(0, 0, 0),
      max: new THREE.Vector3(...pythonGridDimensions),
    };
    let scaled_dimensions = {
      min: new THREE.Vector3(dimensions.min.x * room_scale, dimensions.min.y * room_scale, dimensions.min.z * room_scale),
      max: new THREE.Vector3(dimensions.max.x * room_scale, dimensions.max.y * room_scale, dimensions.max.z * room_scale),
    };
    return points.map(point => mapPoints(point, scaled_dimensions, pythonGrid));
  }

  function mapRange(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
  }

  function mapPoints(point, grid1, grid2) {
    const value = new THREE.Vector3(
      mapRange(point.x, grid1.min.x, grid1.max.x, grid2.min.x, grid2.max.x),
      mapRange(point.y, grid1.min.y, grid1.max.y, grid2.min.y, grid2.max.y),
      mapRange(point.z, grid1.min.z, grid1.max.z, grid2.min.z, grid2.max.z)
    );
    return value;
  }

  useFrame(() => {
    if (droneRef.current) {
      droneRef.current.rotation.y += 0.01;
      if (dronePosition.current) {
        droneRef.current.position.set(
          dronePosition.current.x / room_scale, 
          dronePosition.current.y / room_scale, 
          dronePosition.current.z / room_scale
        );
      }
    }
  });

  useEffect(() => console.log("Drone position: ", dronePosition), [dronePosition]);

  return (
    <group ref={droneRef}>
      <Suspense fallback={<Box />}>
        <DroneModel />
      </Suspense>
    </group>
  );
};

export default Drone;
