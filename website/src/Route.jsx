/* eslint-disable react/prop-types */
import React, { useEffect, useState } from "react";
import * as THREE from "three";
import { Line, Sphere } from "@react-three/drei";
import { useControls, folder, button } from "leva";

const Route = ({room_scale, dimensions}) => {

  const [pointsSets, setPointsSets] = useState([]);
  const [rawPoints, setRawPoints] = useState([]);
  const [pythonWorldDimensions, setPythonWorldDimensions] = useState({x:115, y:103, z:178});
  const color_array = ['blue', 'green', "red", 'orange']

  // Leva controls
  const [{ routeType, X_dest, Y_dest, Z_dest }, set, get] = useControls(() => ({
    'Route Settings': folder({
      routeType: { options: { Random: 'route_random', Projected: 'route_random' } },
      fetchRouteProjection: button(() => fetchRouteProjectionData()),
      X_dest: { value: 0, min: dimensions.min.x * room_scale, max: dimensions.max.x * room_scale, step: 0.1 },
      Y_dest: { value: 0, min: dimensions.min.y * room_scale, max: dimensions.max.y * room_scale, step: 0.1 },
      Z_dest: { value: 0, min: dimensions.min.z * room_scale, max: dimensions.max.z * room_scale, step: 0.1 }
    }, { collapsed: true })
  }));

  // Function to fetch route data
  const fetchRouteData = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/${routeType}`);
      const data = await response.json();
      const routePoints = data.map(point => new THREE.Vector3(point.x, point.y, point.z));
      setPointsSets([routePoints]);
    } catch (error) {
      console.error('Error fetching route data:', error);
    }
  };

  // Function to fetch route projection data
  const fetchRouteProjectionData = async () => {
    try {
      // Transform room scale to python scale
      console.log("Fetching route projection data", get("X_dest"), get("Y_dest"), get("Z_dest"))
      const dest_points_scale_to_python = transformPoints_room_to_python([{x: get("X_dest"), y: get('Y_dest'), z: get('Z_dest')}], [pythonWorldDimensions.x, pythonWorldDimensions.y, pythonWorldDimensions.z])[0];
      const response = await fetch(`${import.meta.env.VITE_API_URL}/route_projection?destination_x=${Math.floor(dest_points_scale_to_python.x)}&destination_y=${Math.floor(dest_points_scale_to_python.y)}&destination_z=${Math.floor(dest_points_scale_to_python.z)}`);
      const data = await response.json();
      data.path = data.path.map(point => new THREE.Vector3(...point));
      data.path_small = data.path_small.map(point => new THREE.Vector3(...point));
      console.log("setting raw points")
      console.log(data)
      setRawPoints(data);
      console.log('Route projection data:', data);
    } catch (error) {
      console.error('Error fetching route projection data:', error);
    }
  };

  // Function to fetch destination data
  const fetchDestinationData = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/destination`);
      let data = await response.json();
      console.log('Destination data:', data);
      if(data && data.point && data.dimensions){
        console.log("Transforming destination data")
        let data_transformed = transformPoints_python_to_room([data.point], data.dimensions)[0];
        set({ "X_dest": data_transformed.x, "Y_dest": data_transformed.y, "Z_dest": data_transformed.z });
      }
      // setPythonWorldDimensions(data.dimensions);   
    } catch (error) {
      console.error('Error fetching destination data:', error);
    }
  };

  useEffect(() => {
    console.log("Python world dimensions updated")
    console.log(pythonWorldDimensions)
  },[pythonWorldDimensions]);

  function mapRange(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
  }

  function mapPoints(point, grid1, grid2){
    const value = new THREE.Vector3(
      mapRange(point.x, grid1.min.x, grid1.max.x, grid2.min.x, grid2.max.x),
      mapRange(point.y, grid1.min.y, grid1.max.y, grid2.min.y, grid2.max.y),
      mapRange(point.z, grid1.min.z, grid1.max.z, grid2.min.z, grid2.max.z)
    );
    return value;
  }

  function transformPoints_python_to_room(points, pythonGridDimensions){
      let pythonGrid = {
        min: new THREE.Vector3(0, 0, 0),
        max: new THREE.Vector3(...pythonGridDimensions),
      }
      let scaled_dimensions = {
        min: new THREE.Vector3(dimensions.min.x * room_scale, dimensions.min.y * room_scale, dimensions.min.z * room_scale),
        max: new THREE.Vector3(dimensions.max.x * room_scale, dimensions.max.y * room_scale, dimensions.max.z * room_scale),
      }
      return points.map(point => mapPoints(point, pythonGrid, scaled_dimensions));
  }

  function transformPoints_room_to_python(points, pythonGridDimensions){
    console.log("python grid")
    console.log(pythonGridDimensions)
    let pythonGrid = {
      min: new THREE.Vector3(0, 0, 0),
      max: new THREE.Vector3(...pythonGridDimensions),
    }
    let scaled_dimensions = {
      min: new THREE.Vector3(dimensions.min.x * room_scale, dimensions.min.y * room_scale, dimensions.min.z * room_scale),
      max: new THREE.Vector3(dimensions.max.x * room_scale, dimensions.max.y * room_scale, dimensions.max.z * room_scale),
    }

    console.log("Transforming points")
    console.log(points)
    console.log(pythonGrid)
    console.log(scaled_dimensions)
    return points.map(point => mapPoints(point, scaled_dimensions, pythonGrid));
  }

  useEffect(() => {
    // fetchRouteData();
    fetchDestinationData();
  }, [routeType]);

  useEffect(() => {
    if (rawPoints && rawPoints.path && rawPoints.path_small) {
        const transformedPoints = []

        transformedPoints.push(transformPoints_python_to_room(rawPoints.path, rawPoints.dimensions));
        transformedPoints.push(transformPoints_python_to_room(rawPoints.path_small, rawPoints.dimensions));

        console.log("Transformed points")
        console.log(transformedPoints)
        setPointsSets(transformedPoints);
        setPythonWorldDimensions(new THREE.Vector3(...rawPoints.dimensions));
    }
  }, [rawPoints]);

  return (
    <mesh scale={1/room_scale}>
      {pointsSets.map((points, index) => {
        console.log("Lines: ")
        console.log(points)
        return(
        <Line
          key={index}
          points={points} // Array of Vector3 points
          color={color_array[index < color_array.length ? index : 0]} // Line color
          lineWidth={2} // Line width
        />
      )})}
      { (
        <Sphere args={[0.2, 32, 32]} position={[X_dest, Y_dest, Z_dest]}>
          <meshStandardMaterial color="red" />
        </Sphere>
      )}
    </mesh>
  );
};

export default Route;
