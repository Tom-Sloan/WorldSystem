# Visualize a specific model using pyrender
import pyrender
import numpy as np
import trimesh
import os
from pathlib import Path

def load_obj_models(base_path="data/models", model_names=None):
    """
    Load OBJ files from the specified directory
    Args:
        base_path (str): Path to the directory containing OBJ files
        model_names (list, optional): List of specific model names to load (without .obj extension)
                                    If None, loads all models
    Returns:
        dict: Dictionary of model name: trimesh object
    """
    models = {}
    obj_path = Path(base_path)
    
    # Get list of files to load
    if model_names is not None:
        files = [obj_path / f"{name}.obj" for name in model_names]
    else:
        files = list(obj_path.glob("*.obj"))
    
    # Load each OBJ file along with its MTL file
    for file in files:
        if not file.exists():
            print(f"Warning: {file.stem} not found")
            continue
        try:
            # Check for corresponding MTL file
            mtl_file = file.with_suffix('.mtl')
            if mtl_file.exists():
                mesh = trimesh.load(str(file), force='mesh', resolver=trimesh.visual.resolvers.FilePathResolver(mtl_file))
            else:
                mesh = trimesh.load(str(file), force='mesh')
            models[file.stem] = mesh
            print(f"Successfully loaded: {file.stem}")
        except Exception as e:
            print(f"Error loading {file.stem}: {e}")
    
    return models

def print_model_info(models):
    """
    Print detailed information about each loaded model and its components
    Args:
        models (dict): Dictionary of model name: trimesh object
    """
    for name, mesh in models.items():
        print(f"\n=== Model: {name} ===")
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print(f"Has texture: {hasattr(mesh.visual, 'texture')}")
        
        # Print information about submeshes if the model is a scene
        if isinstance(mesh, trimesh.Scene):
            print("\nSubmeshes/Objects in scene:")
            for geometry_name, geometry in mesh.geometry.items():
                print(f"\nObject: {geometry_name}")
                print(f"  - Vertices: {len(geometry.vertices)}")
                print(f"  - Faces: {len(geometry.faces)}")
                print(f"  - Material: {getattr(geometry.visual, 'material', 'None')}")
                if hasattr(geometry.visual, 'texture'):
                    print(f"  - Has texture: Yes")
                    if hasattr(geometry.visual.texture, 'name'):
                        print(f"  - Texture name: {geometry.visual.texture.name}")
        else:
            print("\nSingle mesh object (no submeshes)")
            if hasattr(mesh.visual, 'material'):
                print(f"Material: {mesh.visual.material}")

def visualize_model(models, model_name="RoomInitial"):
    """
    Visualize a specific model if it exists in the loaded models
    Args:
        models (dict): Dictionary of model name: trimesh object
        model_name (str): Name of the model to visualize
    """
    if model_name in models:
        mesh = models[model_name]
        mesh.vertices -= mesh.centroid
        scale = 2.0 / mesh.extents.max()
        mesh.vertices *= scale

        # Create a scene with grey background
        scene = trimesh.Scene(mesh)
        scene.show(background=(128, 128, 128, 100))  # RGB + alpha values for grey

def main():
    room_name = "RoomInitial"
    # Load single model
    selected_models = load_obj_models(model_names=[room_name])
    
    # Use the last loaded models for visualization
    models = selected_models  # or all_models or room_model
    
    # Print information about each loaded model
    print_model_info(models)

    # Visualize the Room model if it's in the loaded models
    visualize_model(models, room_name)

if __name__ == "__main__":
    main()
