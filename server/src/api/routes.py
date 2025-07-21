from fastapi import APIRouter, BackgroundTasks, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from src.config.settings import logger, DEBUG_MODE, MODEL_DIR, GRID_RESOLUTION
from src.core.model_loader import load_model, map_obj_point_to_numpy
from src.core.pathfinder import find_path
from typing import List
from pydantic import BaseModel, validator
import os
import random
import math
import time
import asyncio
from functools import lru_cache
import json
import aio_pika

router = APIRouter()

# Add this near the top with other environment variables
RESTART_EXCHANGE = os.getenv("RESTART_EXCHANGE", "restart_exchange")

class Point3D(BaseModel):
    x: float
    y: float
    z: float

    @validator('x', 'y', 'z')
    def validate_coordinate(cls, v):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Coordinate must be a valid number")
        return v

# Global state
current_projection = {
    'path': [],
    'path_small': [],
    'current_location': [0, 0, 0],
    'destination': [0, 0, 0],
    'dimensions': [],
    'file_path': '',
    'model': None,
    'min_points': None,
    'max_points': None,
}

@lru_cache(maxsize=10)
def get_cached_model(file_path: str):
    """Cache loaded models to improve performance."""
    try:
        model, min_points, max_points = load_model(file_path, GRID_RESOLUTION)
        logger.info(f"Model cached: {file_path}")
        return model, min_points, max_points
    except Exception as e:
        logger.error(f"Error caching model: {str(e)}")
        return None, None, None
    
@router.get("/")
def root():
    return {"message": "Hello from the server with tracing!"}

@router.get("/obj")
async def download_obj(file_name: str, background_tasks: BackgroundTasks, request: Request):
    """Serve 3D model files with improved caching."""
    logger.info(f"Received request for file: {file_name} from {request.client.host}")
    
    file_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404, 
            content={"message": f"File {file_name} not found"}
        )

    # Start model loading in background immediately
    if file_name.endswith('.obj'):
        background_tasks.add_task(update_model_async, file_path)
        
    # Return file response
    return FileResponse(
        file_path,
        media_type='model/obj' if file_name.endswith('.obj') else 'application/octet-stream',
        filename=file_name
    )

@router.get("/destination")
async def get_destination():
    return JSONResponse({
        'point': {
            'x': current_projection['destination'][0],
            'y': current_projection['destination'][1],
            'z': current_projection['destination'][2]
        },
        'dimensions': current_projection['dimensions']
    })

@router.post("/set_current_location")
async def set_current_location(point: Point3D):
    """Set current location with validation."""
    try:
        # Validate coordinates are not NaN or infinite
        if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) 
               for v in [point.x, point.y, point.z]):
            logger.error(f"Invalid coordinates received: {point}")
            return JSONResponse(
                status_code=400,
                content={"message": "Coordinates must be valid numbers"}
            )
            
        current_projection['current_location'] = [point.x, point.y, point.z]
        logger.info(f"Current location updated to: {current_projection['current_location']}")
        
        return {
            "status": "success",
            "current_location": current_projection['current_location']
        }
    except Exception as e:
        logger.error(f"Error setting location: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error setting location: {str(e)}"}
        )

@router.get("/current_location")
async def get_current_location():
    return {
        'point': Point3D(
            x=current_projection['current_location'][0],
            y=current_projection['current_location'][1],
            z=current_projection['current_location'][2]
        ),
        'dimensions': current_projection['dimensions']
    }

@router.get("/route_random")
async def get_points_random(num_points: int = 10):
    return [
        Point3D(
            x=random.uniform(-5, 5),
            y=random.uniform(-5, 5),
            z=random.uniform(-5, 5)
        ).dict()
        for _ in range(num_points)
    ]

@router.get("/torus")
async def get_torus():
    radius = 5.0
    period = 10.0
    t = (time.time() % period) / period * (2 * math.pi)
    
    return Point3D(
        x=radius * math.cos(t),
        y=radius * math.sin(t),
        z=radius * math.sin(t)
    )

@router.get("/route_projection")
async def get_pathfinder_route(destination: Point3D):
    """Calculate path to destination."""
    if current_projection['model'] is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No model loaded. Please load a model first."}
        )
    
    try:
        # Map coordinates
        mapped_destination = map_obj_point_to_numpy(
            current_projection['model'],
            current_projection['max_points'],
            current_projection['min_points'],
            (destination.x, destination.y, destination.z)
        )
        
        mapped_current = map_obj_point_to_numpy(
            current_projection['model'],
            current_projection['max_points'],
            current_projection['min_points'],
            tuple(current_projection['current_location'])
        )
        
        # Calculate path
        path, path_small, dimensions = find_path(
            mapped_current,
            mapped_destination,
            current_projection['model']
        )
        
        return {
            'path': path,
            'path_small': path_small,
            'dimensions': dimensions,
            'current_location': list(mapped_current),
            'destination': list(mapped_destination)
        }
        
    except Exception as e:
        logger.error(f"Path calculation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Path calculation failed: {str(e)}"}
        )

async def update_model_async(file_path: str):
    """Asynchronously update the current model using cache."""
    if current_projection.get('file_path') != file_path and os.path.exists(file_path):
        try:
            # Use cached model if available
            model, min_points, max_points = get_cached_model(file_path)
            
            if model is not None:
                current_projection.update({
                    'dimensions': model.shape,
                    'model': model,
                    'min_points': min_points,
                    'max_points': max_points,
                    'file_path': file_path
                })
                logger.info("Model updated from cache")
            else:
                logger.error("Failed to load model from cache")
                
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise

@router.get("/test")
async def test_connection():
    """Test endpoint to verify server is running"""
    return {"status": "ok", "message": "Server is running"}

# Add this after the router definition
async def get_amqp_channel():
    """Dependency to get AMQP channel"""
    # This assumes the connection is established in main.py
    from main import amqp_exchanges
    return amqp_exchanges

@router.post("/restart")
async def trigger_restart(amqp_exchanges: dict = Depends(get_amqp_channel)):
    """
    Publish a restart command so that data_storage and slam3r containers
    will refresh their current state.
    """
    try:
        restart_message = json.dumps({"type": "restart"})
        message = aio_pika.Message(
            body=restart_message.encode(),
            content_type="application/json"
        )
        # Publish on the restart exchange
        await amqp_exchanges[RESTART_EXCHANGE].publish(message, routing_key="")
        logger.info("Published restart command on restart_exchange.")
        return {"status": "success", "message": "Restart command published"}
    except Exception as e:
        logger.error(f"Failed to publish restart command: {str(e)}")
        return {"status": "error", "message": f"Failed to publish restart command: {str(e)}"}
