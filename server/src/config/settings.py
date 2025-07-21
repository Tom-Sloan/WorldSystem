import os
import multiprocessing
import logging
import socket
from enum import Enum

# Debug mode flag
DEBUG_MODE = False  # Set to True to see GPU usage info

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,  # Set to DEBUG in debug mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional: Add file handler to also log to file
if DEBUG_MODE:
    file_handler = logging.FileHandler('server.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Worker configuration
CPU_COUNT = os.cpu_count() or multiprocessing.cpu_count()
IO_MULTIPLIER = 2
OPTIMAL_WORKERS = max((CPU_COUNT - 1) * IO_MULTIPLIER, 1)

# Frame processing settings
MAX_QUEUE_SIZE = 10
FRAME_INTERVAL = 1/30  # 30 FPS target

# Model settings
GRID_RESOLUTION = 5
MODEL_DIR = './models/OBJ'

# API settings
API_PORT = int(os.getenv('API_PORT', 5001))
WS_PORT = int(os.getenv('WS_PORT', 5001))  # Default same as API port
BIND_HOST = os.getenv('BIND_HOST', '0.0.0.0')

def get_local_ip():
    try:
        # Get all network interfaces
        interfaces = socket.getaddrinfo(host=socket.gethostname(), port=None, family=socket.AF_INET)
        ips = [ip[-1][0] for ip in interfaces]
        return ips
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
        return []

if DEBUG_MODE:
    logger.info("Network Interfaces:")
    for ip in get_local_ip():
        logger.info(f"  - {ip}")
    logger.info(f"Server will be accessible at:")
    for ip in get_local_ip():
        logger.info(f"  API: http://{ip}:{API_PORT}")
        if WS_PORT != API_PORT:
            logger.info(f"  WebSocket: ws://{ip}:{WS_PORT}/ws/video")
        else:
            logger.info(f"  WebSocket: ws://{ip}:{API_PORT}/ws/video")

# Analysis modes
class AnalysisMode(Enum):
    YOLO = "yolo"
    NONE = "none"

# Recording settings
class RecordingMode(Enum):
    AUTO = "auto"      # Start recording automatically on startup
    MANUAL = "manual"  # Start recording only via websocket command
    OFF = "off"       # Recording disabled

# Set default modes
ANALYSIS_MODE = AnalysisMode.NONE
RECORDING_MODE = RecordingMode.AUTO  # Change to AUTO to start recording automatically
RECORDING_DIR = os.getenv('RECORDING_DIR', './recordings')

# Add to debug logging
if DEBUG_MODE:
    logger.info(f"Analysis Mode: {ANALYSIS_MODE.value}")
    logger.info(f"Recording Mode: {RECORDING_MODE.value}")
    logger.info(f"Recording Directory: {RECORDING_DIR}")