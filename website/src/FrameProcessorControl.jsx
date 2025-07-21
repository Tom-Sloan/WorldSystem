import { useControls } from "leva";
import { useEffect } from "react";
import { useContext } from "react";
import { WebSocketContext } from "./contexts/WebSocketContext";

function FrameProcessorControl() {
  // Create a dropdown control with two options: "none" and "yolo"
  const { analysisMode } = useControls("Frame Processor", {
    analysisMode: { 
      value: "none", 
      options: { None: "none", YOLO: "yolo" } 
    },
  });
  
  const { sendMessage } = useContext(WebSocketContext);

  // When the value changes, send an update to the server.
  useEffect(() => {
    sendMessage({ type: "analysis_mode", data: analysisMode });
  }, [analysisMode, sendMessage]);

  return null; // This component just sets up the control.
}

export default FrameProcessorControl;
