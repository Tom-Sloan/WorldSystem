/* eslint-disable react/no-unknown-property */
import React, {
  createContext,
  useEffect,
  useState,
  useRef,
  useCallback
} from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';


import RoomModel from './RoomModel';
import Ground from './Ground';
import VideoStream from './VideoStream';
import { WebSocketContext } from './contexts/WebSocketContext';

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarProvider,
} from '@/components/ui/sidebar';

import TrajectoryVisualization from './TrajectoryVisualization';
import OrientationVisualization from './OrientationVisualization';
import TimeSteps from './TimeSteps';

// 1) Import the new ViewCube
import ViewCube from './ViewCube';
// Import the new BarebonesVR component
import BarebonesVR from './BarebonesVR';

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 1000; // 1 second

function App() {
  const [activeTab, setActiveTab] = useState('VideoStream');
  const [showFloatingVideo, setShowFloatingVideo] = useState(true);
  const [isXRSupported, setIsXRSupported] = useState(false);
  const [isInXRSession, setIsInXRSession] = useState(false);

  // We'll keep track of subscribers in a ref to avoid re-renders on updates.
  const subscribers = useRef(new Set());

  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  // We'll store the socket in a ref so we can maintain a stable reference.
  const socketRef = useRef(null);

  // Build the correct WebSocket URL using environment variables.
  // If your site is running over https, we use wss://, else ws://
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/ws/viewer`;

  const connectWebSocket = useCallback(() => {
    // If socket is already open, just return.
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    console.log('Creating new WebSocket connection to', wsUrl);
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected successfully');
      setReconnectAttempts(0);
    };

    ws.onmessage = (event) => {
      // Distribute the message event to all subscribers
      subscribers.current.forEach((handler) => {
        try {
          handler(event);
        } catch (err) {
          console.error('Error in subscriber handler:', err);
        }
      });
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed. code: ${event.code}, reason: ${event.reason}`);
      // Attempt reconnect logic
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        setReconnectAttempts((prev) => prev + 1);
        console.log(
          `Attempting to reconnect (${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...`
        );
        setTimeout(connectWebSocket, RECONNECT_DELAY);
      } else {
        console.error('Max reconnection attempts reached');
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, [wsUrl, reconnectAttempts]);

  // Provide a unified sendMessage function to children.
  const sendMessage = (message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending message:', error);
      }
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  // Subscribers can register a handler function for receiving messages.
  const subscribe = useCallback((handler) => {
    subscribers.current.add(handler);
  }, []);

  // Subscribers can remove their handler when unmounting.
  const unsubscribe = useCallback((handler) => {
    subscribers.current.delete(handler);
  }, []);

  // On initial mount, connect the socket.
  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  // Check for WebXR support when component mounts
  useEffect(() => {
    
    const timer = setTimeout(() => {
      if ('xr' in navigator) {
        navigator.xr.isSessionSupported('immersive-vr')
          .then(supported => {
            console.log("WebXR VR support check result:", supported);
            setIsXRSupported(supported);
          })
          .catch(err => {
            console.error('Error checking WebXR support:', err);
          });
      }
    }, 1000); // Small delay to ensure browser is ready
  
  return () => clearTimeout(timer);
  }, []);

  // Function to start XR session
  const startXRSession = async () => {
    if (!isXRSupported) return;

    try {
      const session = await navigator.xr.requestSession('immersive-vr');
      setIsInXRSession(true);
      
      // Clean up session when it ends
      session.addEventListener('end', () => {
        setIsInXRSession(false);
      });
    } catch (err) {
      console.error('Error starting XR session:', err);
    }
  };

  // Add a small delay when switching tabs to prevent WebSocket issues
  const handleTabChange = (tab) => {
    if (tab === 'VideoStream') {
      setShowFloatingVideo(false);
      setActiveTab(tab);
    } else {
      setActiveTab(tab);
      setTimeout(() => setShowFloatingVideo(true), 50);
    }
  };

  // Printout XR Supported
  useEffect(() => {
    console.log('XR Supported:', isXRSupported);
  }, [isXRSupported]);

  // 2) A ref for our RoomModel, so we can rotate it via <ViewCube />
  const roomRef = useRef(null);

  return (
    <WebSocketContext.Provider
      value={{
        subscribe,
        unsubscribe,
        sendMessage,
      }}
    >
      <SidebarProvider>
        <Sidebar>
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('3DScene')}
                        style={{
                          backgroundColor:
                            activeTab === '3DScene' ? '#2563eb' : 'transparent',
                          color: activeTab === '3DScene' ? 'white' : 'inherit',
                        }}
                      >
                        3D Scene
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('VideoStream')}
                        style={{
                          backgroundColor:
                            activeTab === 'VideoStream' ? '#2563eb' : 'transparent',
                          color: activeTab === 'VideoStream' ? 'white' : 'inherit',
                        }}
                      >
                        Video Stream
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('TrajectoryVisualization')}
                        style={{
                          backgroundColor:
                            activeTab === 'TrajectoryVisualization'
                              ? '#2563eb'
                              : 'transparent',
                          color:
                            activeTab === 'TrajectoryVisualization' ? 'white' : 'inherit',
                        }}
                      >
                        Trajectory Visualization
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('OrientationVisualization')}
                        style={{
                          backgroundColor:
                            activeTab === 'OrientationVisualization'
                              ? '#2563eb'
                              : 'transparent',
                          color:
                            activeTab === 'OrientationVisualization' ? 'white' : 'inherit',
                        }}
                      >
                        Orientation Visualization
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('TimeSteps')}
                        style={{
                          backgroundColor:
                            activeTab === 'TimeSteps' ? '#2563eb' : 'transparent',
                          color: activeTab === 'TimeSteps' ? 'white' : 'inherit',
                        }}
                      >
                        Time Steps
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleTabChange('BarebonesVR')}
                        style={{
                          backgroundColor:
                            activeTab === 'BarebonesVR' ? '#2563eb' : 'transparent',
                          color: activeTab === 'BarebonesVR' ? 'white' : 'inherit',
                        }}
                      >
                        Barebones VR
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>

        <div style={{ flex: 1, position: 'relative' }}>
          {activeTab === '3DScene' && (
            <>
              <Canvas
                shadows
                camera={{ fov: 45, near: 0.1, far: 1000, position: [50, 0, 0] }}
                style={{ backgroundColor: 'grey' }}
              >
                {/* This OrbitControls moves the CAMERA, not the model */}
                <OrbitControls makeDefault />
                <directionalLight position={[1, 2, 3]} intensity={4.5} />
                <ambientLight intensity={1.5} />
                <axesHelper args={[5]} />
                <Ground />

                {/* 3) Forward our ref into <RoomModel> */}
                <RoomModel ref={roomRef} />
              </Canvas>

              {/* 4) The corner cube that rotates roomRef instead of camera */}
              <ViewCube modelRef={roomRef} />

              {showFloatingVideo && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: '20px',
                    right: '20px',
                    width: '15vw',
                    height: 'auto',
                    zIndex: 1000,
                    border: '2px solid white',
                    borderRadius: '10px',
                    overflow: 'hidden',
                  }}
                >
                  <VideoStream isXRMode={isInXRSession} />
                </div>
              )}
            </>
          )}
          {activeTab === 'VideoStream' && <VideoStream isXRMode={isInXRSession} />}
          {activeTab === 'TrajectoryVisualization' && <TrajectoryVisualization />}
          {activeTab === 'OrientationVisualization' && <OrientationVisualization />}
          {activeTab === 'TimeSteps' && <TimeSteps />}
          {activeTab === 'BarebonesVR' && <BarebonesVR />}

          {/* Add XR Session Button when supported */}
          {isXRSupported && activeTab === '3DScene' && !isInXRSession && (
            <button
              onClick={startXRSession}
              style={{
                position: 'absolute',
                bottom: '20px',
                left: '20px',
                padding: '10px 20px',
                backgroundColor: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            >
              Enter VR
            </button>
          )}
        </div>
      </SidebarProvider>
    </WebSocketContext.Provider>
  );
}

export default App;
