import trimesh
import numpy as np
from src.config.settings import logger, DEBUG_MODE, GRID_RESOLUTION

def load_model(path, grid_resolution=GRID_RESOLUTION, is_debug=DEBUG_MODE):
    """Load and process 3D model."""
    try:
        mesh = trimesh.load(path, force="mesh")
        if is_debug:
            logger.info("Loaded mesh")
        
        mesh = center_obj_file(mesh, path)
        if is_debug:
            logger.info("Centered mesh")

        # Get mesh boundaries
        max_points = np.max(mesh.vertices, axis=0)
        min_points = np.min(mesh.vertices, axis=0)
        
        # Create voxel grid
        voxel_grid = adaptive_voxelize(mesh)
        if is_debug:
            logger.info("Voxelized mesh")
            
        # Convert to numpy array
        np_grid = (~voxel_grid.matrix.astype(np.bool_)).astype(int)
        if is_debug:
            logger.info("Converted to numpy array")
            
        return np_grid, tuple(min_points), tuple(max_points)
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def center_obj_file(mesh, output_path):
    """Center the mesh at origin."""
    center = mesh.bounding_box.centroid
    
    # Check if already centered
    tolerance = 1e-5
    if not all(abs(coord) < tolerance for coord in center):
        mesh.apply_translation(-center)
        mesh.export(output_path)
    
    return mesh

def adaptive_voxelize(mesh, target_voxel_count=300000):
    """Voxelize mesh with adaptive resolution."""
    bbox = mesh.bounding_box.extents
    total_volume = bbox.prod()
    voxel_volume = total_volume / target_voxel_count
    grid_resolution = voxel_volume ** (1/3)
    
    return trimesh.voxel.creation.voxelize(mesh, pitch=grid_resolution)

def map_obj_point_to_numpy(model, max_points, min_points, point, is_debug=False):
    """Map point from obj space to numpy array indices."""
    if is_debug:
        logger.info(f"Mapping point {point} to numpy space")
    
    delta = np.array(max_points) - np.array(min_points)
    point = np.array(point) + np.abs(min_points)
    
    # Map to array indices
    mapped = [
        int(np.floor(p * (s-1) / d)) 
        for p, d, s in zip(point, delta, model.shape)
    ]
    
    return tuple(mapped)

# ... (rest of the model loading functions) 