import scipy.ndimage
import numpy as np
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from src.config.settings import logger, DEBUG_MODE

def find_path(start_pt, end_pt, model, is_debug=DEBUG_MODE):
    """Find path between two points in 3D space."""
    try:
        start_pt = np.array(start_pt)
        end_pt = np.array(end_pt)
        
        # Expand obstacles
        expanded_grid = expand_obstacles(model, 2)
        if is_debug:
            logger.info("Expanded obstacles")
        
        # Create grids
        grid = Grid(matrix=expanded_grid)
        grid_small = Grid(matrix=model)
        
        # Create nodes
        start = grid.node(*start_pt)
        end = grid.node(*end_pt)
        start_small = grid_small.node(*start_pt)
        end_small = grid_small.node(*end_pt)
        
        # Find paths
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, _ = finder.find_path(start, end, grid)
        path_small, _ = finder.find_path(start_small, end_small, grid_small)
        
        return [p.identifier for p in path], [p.identifier for p in path_small], model.shape
        
    except Exception as e:
        logger.error(f"Pathfinding error: {str(e)}")
        raise

def expand_obstacles(grid, expansion_layers):
    """Expand obstacles in the grid."""
    structure = np.ones((3, 3, 3), dtype=np.bool_)
    expanded_grid = grid.copy()
    
    for _ in range(expansion_layers):
        expanded_grid = scipy.ndimage.binary_erosion(
            expanded_grid, 
            structure=structure,
            border_value=1
        )
    
    return expanded_grid.astype(int)

# ... (rest of the pathfinding functions) 