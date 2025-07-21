import React, { useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import './ControlVisualizer.css';

// Main control visualizer component
function ControlVisualizer({ 
  activeKeys, 
  leftStick, 
  rightStick, 
  vrButtonsPressed, 
  isXRMode = false, 
  cameraActive = false, 
  flightMode = 'land' 
}) {
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  
  // Measure container size
  useEffect(() => {
    if (containerRef.current) {
      const updateSize = () => {
        setContainerSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      };
      
      // Initial measurement
      updateSize();
      
      // Set up resize observer to update measurements when container changes size
      const resizeObserver = new ResizeObserver(updateSize);
      resizeObserver.observe(containerRef.current);
      
      return () => {
        if (containerRef.current) {
          resizeObserver.unobserve(containerRef.current);
        }
      };
    }
  }, []);

  return (
    <div 
      ref={containerRef}
      className="control-visualizer" 
      style={{ 
        background: 'rgba(0, 0, 0, 0.7)',
        borderRadius: '8px',
        height: '180px',
        width: '100%'
      }}
    >
      <KeyboardLayout 
        activeKeys={activeKeys} 
        leftStick={leftStick} 
        rightStick={rightStick} 
        vrButtonsPressed={vrButtonsPressed}
        cameraActive={cameraActive}
        flightMode={flightMode}
        isXRMode={isXRMode}
      />
    </div>
  );
}

const KeyboardLayout = ({ activeKeys, leftStick, rightStick, cameraActive, flightMode, isXRMode }) => {
  // Convert stick values to key highlights
  const getKeyHighlight = (key) => {
    // Special handling for arrow keys which use different casing in the event system
    if (key === 'ArrowUp' && activeKeys['ArrowUp']) return 'pressed';
    if (key === 'ArrowDown' && activeKeys['ArrowDown']) return 'pressed';
    if (key === 'ArrowLeft' && activeKeys['ArrowLeft']) return 'pressed';
    if (key === 'ArrowRight' && activeKeys['ArrowRight']) return 'pressed';
    
    // Handle direct key presses (case insensitive)
    const keyLower = key.toLowerCase();
    if (activeKeys[keyLower] || activeKeys[key.toUpperCase()]) return 'pressed';
    
    // Handle stick mappings for WASD (using threshold of 0.3)
    switch(key) {
      case 'W':
        return leftStick.y > 0.3 ? 'pressed' : '';
      case 'A':
        return leftStick.x < -0.3 ? 'pressed' : '';
      case 'S':
        return leftStick.y < -0.3 ? 'pressed' : '';
      case 'D':
        return leftStick.x > 0.3 ? 'pressed' : '';
      // Handle stick mappings for arrow keys
      case 'ArrowLeft':
        return rightStick.x < -0.3 ? 'pressed' : '';
      case 'ArrowRight':
        return rightStick.x > 0.3 ? 'pressed' : '';
      case 'ArrowUp':
        return rightStick.y > 0.3 ? 'pressed' : '';
      case 'ArrowDown':
        return rightStick.y < -0.3 ? 'pressed' : '';
      default:
        return '';
    }
  };

  return (
    <div className="keyboard-layout">
      <div className="keyboard-grid">
        {/* Movement Keys (WASD) */}
        <div className="wasd-cluster">
          <div className={`key ${getKeyHighlight('W')}`}>W</div>
          <div className="wasd-bottom-row">
            <div className={`key ${getKeyHighlight('A')}`}>A</div>
            <div className={`key ${getKeyHighlight('S')}`}>S</div>
            <div className={`key ${getKeyHighlight('D')}`}>D</div>
          </div>
        </div>
        
        {/* Middle section */}
        <div className="middle-keys">
          <div className={`key ${getKeyHighlight('T')} ${flightMode === 'land' ? 'inactive' : 'active'}`}>T</div>
          <div className={`key ${getKeyHighlight('P')} ${cameraActive ? 'active' : 'inactive'}`}>P</div>
        </div>
        
        {/* Arrow Keys for Rotation/Altitude */}
        <div className="arrow-keys-cluster">
          <div className={`key ${getKeyHighlight('ArrowUp')}`}>↑</div>
          <div className="arrow-bottom-row">
            <div className={`key ${getKeyHighlight('ArrowLeft')}`}>←</div>
            <div className={`key ${getKeyHighlight('ArrowDown')}`}>↓</div>
            <div className={`key ${getKeyHighlight('ArrowRight')}`}>→</div>
          </div>
        </div>
        
        {/* Controller values display (spans all columns) */}
          <div className="controller-values">
          <div className="controller-label">Movement: {leftStick.x.toFixed(2)}, {leftStick.y.toFixed(2)}</div>
          <div className="controller-label">Rotation: {rightStick.x.toFixed(2)}, {rightStick.y.toFixed(2)}</div>
          <div className="controller-label">Camera: {cameraActive ? 'On' : 'Off'}</div>
          <div className="controller-label">Flight Mode: {flightMode.charAt(0).toUpperCase() + flightMode.slice(1)}</div>
          <div className="controller-label">XR Mode: {isXRMode ? 'On' : 'Off'}</div>
        </div>
      </div>
    </div>
  );
};

ControlVisualizer.propTypes = {
  activeKeys: PropTypes.object.isRequired,
  leftStick: PropTypes.shape({
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired
  }).isRequired,
  rightStick: PropTypes.shape({
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired
  }).isRequired,
  vrButtonsPressed: PropTypes.object.isRequired,
  isXRMode: PropTypes.bool,
  cameraActive: PropTypes.bool,
  flightMode: PropTypes.string
};

KeyboardLayout.propTypes = {
  activeKeys: PropTypes.object.isRequired,
  leftStick: PropTypes.shape({
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired
  }).isRequired,
  rightStick: PropTypes.shape({
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired
  }).isRequired,
  cameraActive: PropTypes.bool,
  flightMode: PropTypes.string,
  isXRMode: PropTypes.bool
};

export default ControlVisualizer;