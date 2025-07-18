import os
import time
import json
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class FileLogger:
    """File-based logger for frame processor service with rotating logs."""
    
    def __init__(self, log_dir: str = "/app/logs", max_file_size_mb: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Different log files for different types
        self.log_files = {
            "general": self.log_dir / "frame_processor.log",
            "detections": self.log_dir / "detections.log",
            "tracking": self.log_dir / "tracking.log",
            "dimensions": self.log_dir / "dimensions.log",
            "errors": self.log_dir / "errors.log",
            "metrics": self.log_dir / "metrics.log",
            "rerun": self.log_dir / "rerun_messages.log"
        }
        
        # Thread lock for file operations
        self.lock = threading.Lock()
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        
        # Write initial session header
        self._write_session_header()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        return obj
    
    def _write_session_header(self):
        """Write session header to all log files."""
        header = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": self.start_time,
            "log_type": "SESSION_START"
        }
        
        for log_type in self.log_files:
            self._write_to_file(log_type, header)
    
    def _rotate_log_if_needed(self, file_path: Path) -> None:
        """Rotate log file if it exceeds max size."""
        try:
            if file_path.exists() and file_path.stat().st_size > self.max_file_size_bytes:
                # Archive old log
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = file_path.with_suffix(f".{timestamp}.log")
                file_path.rename(archive_path)
                
                # Log rotation event
                self.log("general", "LOG_ROTATION", {
                    "archived_file": str(archive_path),
                    "original_file": str(file_path)
                })
        except Exception as e:
            print(f"Error rotating log file: {e}")
    
    def _write_to_file(self, log_type: str, data: Dict[str, Any]) -> None:
        """Write data to specific log file."""
        if log_type not in self.log_files:
            log_type = "general"
        
        file_path = self.log_files[log_type]
        
        with self.lock:
            self._rotate_log_if_needed(file_path)
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = time.time()
            if "datetime" not in data:
                data["datetime"] = datetime.now().isoformat()
            
            # Write as JSON line
            try:
                # Convert numpy types to Python native types for JSON serialization
                data_converted = self._convert_numpy_types(data)
                with open(file_path, "a") as f:
                    f.write(json.dumps(data_converted) + "\n")
            except Exception as e:
                print(f"Error writing to log file {file_path}: {e}")
    
    def log(self, log_type: str, event: str, data: Optional[Union[Dict, str]] = None) -> None:
        """Log an event with optional data."""
        log_entry = {
            "event": event,
            "session_id": self.session_id
        }
        
        if isinstance(data, str):
            log_entry["message"] = data
        elif isinstance(data, dict):
            log_entry.update(data)
        elif data is not None:
            log_entry["data"] = str(data)
        
        self._write_to_file(log_type, log_entry)
    
    def log_frame_processing(self, frame_number: int, processing_time_ms: float, 
                           detections: int = 0, tracked_objects: int = 0) -> None:
        """Log frame processing metrics."""
        self.log("metrics", "FRAME_PROCESSED", {
            "frame_number": frame_number,
            "processing_time_ms": processing_time_ms,
            "detections": detections,
            "tracked_objects": tracked_objects
        })
    
    def log_detection(self, frame_number: int, class_name: str, confidence: float,
                     bbox: tuple, track_id: Optional[int] = None) -> None:
        """Log object detection."""
        self.log("detections", "OBJECT_DETECTED", {
            "frame_number": frame_number,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": list(bbox),
            "track_id": track_id
        })
    
    def log_tracking_event(self, event_type: str, track_id: int, 
                          class_name: str, data: Optional[Dict] = None) -> None:
        """Log tracking events (new track, lost track, etc)."""
        log_data = {
            "track_id": track_id,
            "class_name": class_name
        }
        if data:
            log_data.update(data)
        
        self.log("tracking", event_type, log_data)
    
    def log_dimension_result(self, track_id: int, product_name: str,
                           dimensions: Dict, confidence: float) -> None:
        """Log dimension estimation results."""
        self.log("dimensions", "DIMENSION_ESTIMATED", {
            "track_id": track_id,
            "product_name": product_name,
            "dimensions": dimensions,
            "confidence": confidence
        })
    
    def log_error(self, error_type: str, error_message: str, 
                  context: Optional[Dict] = None) -> None:
        """Log errors with context."""
        log_data = {
            "error_type": error_type,
            "error_message": error_message
        }
        if context:
            log_data["context"] = context
        
        self.log("errors", "ERROR", log_data)
    
    def log_rerun_message(self, path: str, message_type: str, 
                         data: Optional[Dict] = None) -> None:
        """Log Rerun visualization messages."""
        log_data = {
            "rerun_path": path,
            "message_type": message_type
        }
        if data:
            log_data.update(data)
        
        self.log("rerun", "RERUN_MESSAGE", log_data)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        uptime = time.time() - self.start_time
        
        summary = {
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime/3600:.1f} hours",
            "log_files": {
                log_type: str(path) for log_type, path in self.log_files.items()
            }
        }
        
        # Add file sizes
        for log_type, path in self.log_files.items():
            if path.exists():
                summary[f"{log_type}_size_mb"] = path.stat().st_size / (1024 * 1024)
        
        return summary
    
    def close(self):
        """Close logger and write session end."""
        self.log("general", "SESSION_END", self.get_session_summary())