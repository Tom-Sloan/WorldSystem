import React, { useEffect, useContext, useState, useRef, useCallback } from 'react';
import PropTypes from 'prop-types';
import { WebSocketContext } from './contexts/WebSocketContext';
import { useControls, folder, button } from 'leva';
import FrameProcessorControl from './FrameProcessorControl';
import ControlVisualizer from './components/ControlVisualizer';

// NTP time synchronization
const NTP_SYNC_INTERVAL = 60000; // Sync every minute
const NTP_SERVER_URL = "https://worldtimeapi.org/api/ip"; // Public time API

function VideoStream({ style = {}, isXRMode = false }) {
  const { subscribe, unsubscribe, sendMessage } = useContext(WebSocketContext);
  const [videoSrc, setVideoSrc] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const ntpOffsetRef = useRef(0); // Time offset in ms between local clock and NTP time
  const lastNtpSyncRef = useRef(0);
  const videoContainerRef = useRef(null); // Reference to track video container width
  const [containerWidth, setContainerWidth] = useState(0);
  const [cameraActive, setCameraActive] = useState(false); // Default camera state is OFF
  const [flightMode, setFlightMode] = useState('land'); // Default flight mode is LAND
  
  const [stats, setStats] = useState({
    frameSize: '',
    processingTime: 0,
    networkLatency: 0,
    totalLatency: 0,
    resolution: 'Unknown'
  });
  
  // Add arrays to track the last 10 samples for calculating averages
  const [processingTimeHistory, setProcessingTimeHistory] = useState([]);
  const [networkLatencyHistory, setNetworkLatencyHistory] = useState([]);
  const [totalLatencyHistory, setTotalLatencyHistory] = useState([]);
  const MAX_HISTORY_LENGTH = 10;

  // Track active keys for smooth joystick-like control
  const activeKeys = useRef({
    // Movement (WASD)
    'w': false,
    'W': false,
    'a': false,
    'A': false,
    's': false,
    'S': false,
    'd': false,
    'D': false,
    // Rotation/Altitude (Arrow keys)
    'ArrowLeft': false,
    'ArrowRight': false,
    'ArrowUp': false,
    'ArrowDown': false,
    // Flight mode toggle
    't': false,
    'T': false,
    // Camera toggle
    'p': false,
    'P': false
  });

  // Helper function to calculate average of an array
  const calculateAverage = (arr) => {
    if (arr.length === 0) return 0;
    const sum = arr.reduce((acc, val) => acc + parseFloat(val), 0);
    return (sum / arr.length).toFixed(1);
  };

  // NTP synchronization
  const syncNtpTime = async () => {
    try {
      const startTime = Date.now();
      const response = await fetch(NTP_SERVER_URL);
      const endTime = Date.now();
      const data = await response.json();
      
      // Parse UTC time from API
      const serverTime = new Date(data.utc_datetime).getTime();
      // Adjust for network latency (approximately half of round trip time)
      const networkDelay = (endTime - startTime) / 2;
      // Server time + network delay = adjusted server time
      const adjustedServerTime = serverTime + networkDelay;
      // Calculate offset: server time - local time
      const offset = adjustedServerTime - endTime;
      
      ntpOffsetRef.current = offset;
      lastNtpSyncRef.current = Date.now();
      console.log(`[NTP] Synchronized time, offset: ${offset}ms`);
    } catch (error) {
      console.error('[NTP] Synchronization failed:', error);
    }
  };

  // Get current time in nanoseconds, synchronized with NTP
  const getNtpTimeNs = () => {
    // Check if we need to resync
    if (Date.now() - lastNtpSyncRef.current > NTP_SYNC_INTERVAL) {
      syncNtpTime(); // Async function, will update offset later
    }
    // Current time adjusted by NTP offset, rounded to nearest integer
    const timeMs = Math.round(Date.now() + ntpOffsetRef.current);
    return BigInt(timeMs) * BigInt(1000000); // Convert to ns
  };

  // Initial NTP sync
  useEffect(() => {
    syncNtpTime();
  }, []);

  // Track video container width to ensure stats bar matches video width
  useEffect(() => {
    if (videoContainerRef.current) {
      const updateWidth = () => {
        const width = videoContainerRef.current.offsetWidth;
        setContainerWidth(width);
      };
      
      // Initial width calculation
      updateWidth();
      
      // Update on resize
      const resizeObserver = new ResizeObserver(updateWidth);
      resizeObserver.observe(videoContainerRef.current);
      
      return () => {
        if (videoContainerRef.current) {
          resizeObserver.unobserve(videoContainerRef.current);
        }
      };
    }
  }, [videoSrc]);

  // Handle keyboard controls for drone movement
  const handleKeyDown = useCallback((event) => {
    const key = event.key;
    if (key in activeKeys.current) {
      event.preventDefault();
      activeKeys.current[key] = true;
      
      // Handle camera toggle (P key)
      if (key === 'p' || key === 'P') {
        const newCameraState = !cameraActive;
        setCameraActive(newCameraState);
        sendMessage({
          type: 'camera',
          action: newCameraState ? 'on' : 'off',
          timestamp: getNtpTimeNs().toString()
        });
        return;
      }
      
      // Handle takeoff/land toggle (T key)
      if (key === 't' || key === 'T') {
        // If current mode is land, first ensure camera is on before takeoff
        if (flightMode === 'land') {
          // If camera is not on, turn it on first
          if (!cameraActive) {
            setCameraActive(true);
            sendMessage({
              type: 'camera',
              action: 'on',
              timestamp: getNtpTimeNs().toString()
            });
          }
          
          // Then takeoff
          setFlightMode('takeoff');
          sendMessage({
            type: 'flightmode',
            action: 'takeoff',
            timestamp: getNtpTimeNs().toString()
          });
        } else {
          // If current mode is takeoff, then land
          setFlightMode('land');
          sendMessage({
            type: 'flightmode',
            action: 'land',
            timestamp: getNtpTimeNs().toString()
          });
        }
        return;
      }
      
      // Process movement keys (WASD)
      const movementX = (activeKeys.current['d'] || activeKeys.current['D'] ? 1 : 0) - 
                        (activeKeys.current['a'] || activeKeys.current['A'] ? 1 : 0);
      const movementY = (activeKeys.current['w'] || activeKeys.current['W'] ? 1 : 0) - 
                        (activeKeys.current['s'] || activeKeys.current['S'] ? 1 : 0);
      
      if (['w', 'W', 'a', 'A', 's', 'S', 'd', 'D'].includes(key)) {
        sendMessage({
          type: 'movement',
          x: movementX,
          y: movementY,
          timestamp: getNtpTimeNs().toString()
        });
      }
      
      // Process rotation/altitude keys (Arrow keys)
      const yaw = (activeKeys.current['ArrowRight'] ? 1 : 0) - 
                  (activeKeys.current['ArrowLeft'] ? 1 : 0);
      const z = (activeKeys.current['ArrowUp'] ? 1 : 0) - 
                (activeKeys.current['ArrowDown'] ? 1 : 0);
      
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(key)) {
        sendMessage({
          type: 'rotation',
          yaw: yaw,
          z: z,
          timestamp: getNtpTimeNs().toString()
        });
      }
    }
  }, [sendMessage, cameraActive, flightMode]);

  // Handle key up to reset values
  const handleKeyUp = useCallback((event) => {
    const key = event.key;
    if (key in activeKeys.current) {
      event.preventDefault();
      activeKeys.current[key] = false;
      
      // Skip one-time actions
      if (['t', 'T', 'p', 'P'].includes(key)) {
        return;
      }
      
      // Process movement keys (WASD)
      const movementX = (activeKeys.current['d'] || activeKeys.current['D'] ? 1 : 0) - 
                        (activeKeys.current['a'] || activeKeys.current['A'] ? 1 : 0);
      const movementY = (activeKeys.current['w'] || activeKeys.current['W'] ? 1 : 0) - 
                        (activeKeys.current['s'] || activeKeys.current['S'] ? 1 : 0);
      
      if (['w', 'W', 'a', 'A', 's', 'S', 'd', 'D'].includes(key)) {
        sendMessage({
          type: 'movement',
          x: movementX,
          y: movementY,
          timestamp: getNtpTimeNs().toString()
        });
      }
      
      // Process rotation/altitude keys (Arrow keys)
      const yaw = (activeKeys.current['ArrowRight'] ? 1 : 0) - 
                  (activeKeys.current['ArrowLeft'] ? 1 : 0);
      const z = (activeKeys.current['ArrowUp'] ? 1 : 0) - 
                (activeKeys.current['ArrowDown'] ? 1 : 0);
      
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(key)) {
        sendMessage({
          type: 'rotation',
          yaw: yaw,
          z: z,
          timestamp: getNtpTimeNs().toString()
        });
      }
    }
  }, [sendMessage]);

  // Force a re-render when keys are pressed to update the visualizer
  const [, forceUpdate] = useState({});
  
  // Add keyboard event listeners
  useEffect(() => {
    // Enhanced key handler that also forces a re-render
    const enhancedKeyDown = (event) => {
      handleKeyDown(event);
      // Force component to re-render when keys change
      forceUpdate({});
    };
    
    const enhancedKeyUp = (event) => {
      handleKeyUp(event);
      // Force component to re-render when keys change
      forceUpdate({});
    };
    
    window.addEventListener('keydown', enhancedKeyDown, { capture: true });
    window.addEventListener('keyup', enhancedKeyUp, { capture: true });
    
    return () => {
      window.removeEventListener('keydown', enhancedKeyDown, { capture: true });
      window.removeEventListener('keyup', enhancedKeyUp, { capture: true });
    };
  }, [handleKeyDown, handleKeyUp]);

  // Improved WebSocket handling with reconnect logic
  useEffect(() => {
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    let reconnectTimeout;

    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'processed_frame') {
          // Get NTP-synchronized time when frame is received
          const clientReceivedNs = getNtpTimeNs();
          setVideoSrc('data:image/jpeg;base64,' + message.frame_data);
          setConnectionStatus('connected');
          
          // Calculate latencies using NTP-synchronized timestamps
          const frameTimestamp = BigInt(message.timestamp_ns);        // Original frame capture
          const serverNtpTime = BigInt(message.ntp_time);             // Server NTP time
          
          // Calculate server processing time (if available, otherwise use estimate)
          const processingTimeMs = message.processing_time_ms 
            ? Number(message.processing_time_ms) 
            : 10.0; // Default estimate
          
          // Calculate network latency (server → client)
          const networkLatencyMs = Number(clientReceivedNs - serverNtpTime) / 1e6;
          
          // Calculate total latency
          const totalLatencyMs = Number(clientReceivedNs - frameTimestamp) / 1e6;
          
          // Update histories with new values and maintain max length
          setProcessingTimeHistory(prev => {
            const updated = [...prev, processingTimeMs.toFixed(1)];
            return updated.length > MAX_HISTORY_LENGTH ? updated.slice(-MAX_HISTORY_LENGTH) : updated;
          });
          
          setNetworkLatencyHistory(prev => {
            const updated = [...prev, networkLatencyMs.toFixed(1)];
            return updated.length > MAX_HISTORY_LENGTH ? updated.slice(-MAX_HISTORY_LENGTH) : updated;
          });
          
          setTotalLatencyHistory(prev => {
            const updated = [...prev, totalLatencyMs.toFixed(1)];
            return updated.length > MAX_HISTORY_LENGTH ? updated.slice(-MAX_HISTORY_LENGTH) : updated;
          });
          
          // Update stats
          setStats({
            frameSize: `${message.width}x${message.height}`,
            resolution: message.resolution || "Unknown",
            processingTime: processingTimeMs.toFixed(1),
            networkLatency: networkLatencyMs.toFixed(1),
            totalLatency: totalLatencyMs.toFixed(1),
            ntpOffset: (ntpOffsetRef.current / 1000).toFixed(3), // Show NTP offset in seconds
            // Add average values
            avgProcessingTime: calculateAverage(processingTimeHistory),
            avgNetworkLatency: calculateAverage(networkLatencyHistory),
            avgTotalLatency: calculateAverage(totalLatencyHistory)
          });
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    const connect = () => {
      subscribe(handleMessage, {
        onOpen: () => {
          reconnectAttempts = 0;
          setConnectionStatus('connected');
        },
        onClose: () => {
          if (reconnectAttempts < maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            reconnectTimeout = setTimeout(() => {
              reconnectAttempts++;
              connect();
            }, delay);
            setConnectionStatus(`reconnecting (${reconnectAttempts}/${maxReconnectAttempts})`);
          }
        },
        onError: (error) => {
          console.error('WebSocket error:', error);
          setConnectionStatus('connection failed');
        }
      });
    };

    connect();

    return () => {
      unsubscribe(handleMessage);
      clearTimeout(reconnectTimeout);
    };
  }, [subscribe, unsubscribe]);

  const [controls, set] = useControls(() => ({
    'Image Controls': folder(
      {
        brightness: { value: 1, min: 0, max: 2, step: 0.1 },
        contrast: { value: 1, min: 0, max: 2, step: 0.1 },
        saturation: { value: 1, min: 0, max: 2, step: 0.1 },
        sendMessage: button(() => sendMessage({
          type: 'machineLearning',
          data: false,
          timestamp: getNtpTimeNs().toString()
        })),
      },
      { collapsed: true }
    ),
    'Drone Controls': folder(
      {
        Movement: {
          value: [0, 0],
          max: [1, 1],
          min: [-1, -1],
          joystick: 'invertY',
          step: 0.1,
        },
        Rotation: {
          value: [0, 0],
          max: [1, 1],
          min: [-1, -1],
          joystick: 'invertY',
          step: 0.1,
        },
        TakeOff: button(() => {
          // If camera is not on, turn it on first
          if (!cameraActive) {
            setCameraActive(true);
            sendMessage({ 
              type: 'camera', 
              action: 'on',
              timestamp: getNtpTimeNs().toString()
            });
          }
          
          setFlightMode('takeoff');
          sendMessage({ 
            type: 'flightmode', 
            action: 'takeoff',
            timestamp: getNtpTimeNs().toString()
          });
          console.log('Take Off');
        }),
        Land: button(() => {
          setFlightMode('land');
          sendMessage({ 
            type: 'flightmode', 
            action: 'land',
            timestamp: getNtpTimeNs().toString()
          });
          console.log('Land');
        }),
        ToggleCamera: button(() => {
          const newCameraState = !cameraActive;
          setCameraActive(newCameraState);
          sendMessage({ 
            type: 'camera', 
            action: newCameraState ? 'on' : 'off',
            timestamp: getNtpTimeNs().toString()
          });
          console.log('Camera Toggle:', newCameraState ? 'On' : 'Off');
        }),
      },
      { collapsed: true }
    )
  }));

  // Define tooltips for metrics
  const tooltips = {
    processing: "Average server-side processing time (ms) over the last 10 frames",
    network: "Average network transmission latency (ms) over the last 10 frames",
    total: "Average end-to-end latency from frame capture to display (ms) over the last 10 frames",
    ntp: "Network Time Protocol offset in seconds - difference between local time and server time. Helps synchronize time across devices."
  };

  // Add connection status overlay
  return (
    <div style={{ position: 'relative', ...style }}>
      {connectionStatus !== 'connected' && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          zIndex: 1000
        }}>
          {connectionStatus.toUpperCase()}
        </div>
      )}
      <div 
        className="video-container" 
        style={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
        ref={videoContainerRef}
      >
        <img
          src={videoSrc}
          alt=""
          style={{
            width: '100%',
            height: 'auto',
            maxHeight: 'calc(70vh - 200px)', // Leave room for controls
            objectFit: 'contain',
            filter: `
              brightness(${controls.brightness})
              contrast(${controls.contrast})
              saturate(${controls.saturation})
            `,
          }}
        />
        <div style={{
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '8px',
          fontSize: '14px',
          fontFamily: 'monospace',
          display: 'flex',
          justifyContent: 'space-between',
          width: '100%',
          boxSizing: 'border-box',
          maxWidth: '100%' // Ensure it doesn't exceed parent width
        }}>
          <span>Size: {stats.frameSize}</span>
          <span title={tooltips.processing} style={{ cursor: 'help' }}>
            Processing: {stats.processingTime}ms
          </span>
          <span title={tooltips.network} style={{ cursor: 'help' }}>
            Network: {stats.networkLatency}ms
          </span>
          <span title={tooltips.ntp} style={{ cursor: 'help' }}>
            NTP: {stats.ntpOffset}s
          </span>
        </div>
      </div>
      <FrameProcessorControl />
      <div className="controls-container" style={{
        background: 'rgba(0, 0, 0, 0.7)',
        borderRadius: '12px',
        padding: '12px',
        marginTop: '12px',
        height: 'auto', // Adjust height automatically
        width: containerWidth > 0 ? containerWidth : '100%', // Match video width
        maxWidth: '100%', // Prevent overflow
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        
        <ControlVisualizer
          activeKeys={{ ...activeKeys.current }}
          leftStick={{
            x: controls.Movement[0],
            y: controls.Movement[1]
          }}
          rightStick={{
            x: controls.Rotation[0],
            y: controls.Rotation[1]
          }}
          vrButtonsPressed={{
            left: {
              joystick: false,
              trigger: false,
              grip: false
            },
            right: {
              joystick: false,
              trigger: false,
              grip: false
            }
          }}
          isXRMode={isXRMode}
          cameraActive={cameraActive}
          flightMode={flightMode}
        />
      </div>
    </div>
  );
}

VideoStream.propTypes = {
  style: PropTypes.object,
  isXRMode: PropTypes.bool
};

export default VideoStream;