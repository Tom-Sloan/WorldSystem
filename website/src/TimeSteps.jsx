import React, { useEffect, useState, useContext } from 'react';
import { WebSocketContext } from './contexts/WebSocketContext';

function TimeSteps() {
  const { subscribe, unsubscribe } = useContext(WebSocketContext);
  const [imuTimestamps, setImuTimestamps] = useState([]);
  const [imageTimestamps, setImageTimestamps] = useState([]);

  // Maximum number of entries to keep in the list
  const MAX_ENTRIES = 20;

  useEffect(() => {
    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        // Process IMU messages
        if (message.type === 'imu_data' && message.imu_data && message.imu_data.timestamp) {
          const newTs = message.imu_data.timestamp;
          console.log("imu timestamp:", newTs);
          setImuTimestamps((prev) => {
            // If the list is empty or the new timestamp is in order, update the list.
            if (prev.length === 0 || newTs >= prev[prev.length - 1]) {
              const newList = [...prev, newTs];
              return newList.slice(-MAX_ENTRIES);
            } else {
              // Otherwise, ignore the out-of-order timestamp.
              console.warn("Ignoring out-of-order imu timestamp:", newTs);
              return prev;
            }
          });
        }
        // Process image (processed frame) messages that include a timestamp
        else if (message.type === 'processed_frame' && message.timestamp) {
          console.log("image timestamp:", message.timestamp);
          setImageTimestamps((prev) => {
            const newList = [...prev, message.timestamp];
            return newList.slice(-MAX_ENTRIES);
          });
        }
      } catch (err) {
        console.error('Error processing TimeSteps message:', err);
      }
    };

    subscribe(handleMessage);
    return () => {
      unsubscribe(handleMessage);
    };
  }, [subscribe, unsubscribe]);

  // Determine the number of rows (we combine the two lists by index)
  const maxRows = Math.max(imuTimestamps.length, imageTimestamps.length);

  const renderRows = () => {
    const rows = [];
    for (let i = 0; i < maxRows; i++) {
      // Get the ith timestamp from each list (if available)
      const imu = imuTimestamps[i] || '';
      const image = imageTimestamps[i] || '';
      // Compute delta only if both timestamps are available.
      const delta = (imu && image) ? image - imu : '';
      rows.push(
        <tr key={i}>
          <td style={{ padding: '0.5rem', border: '1px solid #fff' }}>{imu}</td>
          <td style={{ padding: '0.5rem', border: '1px solid #fff' }}>{image}</td>
          <td style={{ padding: '0.5rem', border: '1px solid #fff' }}>{delta}</td>
        </tr>
      );
    }
    return rows;
  };

  return (
    <div
      style={{
        padding: '1rem',
        color: 'white',
        backgroundColor: '#333', // Dark background for contrast
        minHeight: '100vh',      // Ensures the container occupies full viewport height
      }}
    >
      <h2>Time Steps</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ padding: '0.5rem', border: '1px solid #fff', textAlign: 'left' }}>
              IMU Timestamp (ms)
            </th>
            <th style={{ padding: '0.5rem', border: '1px solid #fff', textAlign: 'left' }}>
              Image Timestamp (ms)
            </th>
            <th style={{ padding: '0.5rem', border: '1px solid #fff', textAlign: 'left' }}>
              Delta (ms)
            </th>
          </tr>
        </thead>
        <tbody>{renderRows()}</tbody>
      </table>
    </div>
  );
}

export default TimeSteps;
