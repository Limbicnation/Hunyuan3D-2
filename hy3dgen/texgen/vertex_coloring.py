# Simple vertex coloring utility for memory-limited systems
# This is used as a fallback when texture generation fails due to memory constraints

import numpy as np
import trimesh
from PIL import Image


def apply_vertex_colors(mesh, image):
    """
    Apply simple vertex coloring to a mesh based on vertex positions and an input image.
    This is a low-memory alternative to full texture generation.
    
    Args:
        mesh: A trimesh.Trimesh object
        image: A PIL.Image.Image object
        
    Returns:
        A trimesh.Trimesh object with vertex colors
    """
    print("Applying simple vertex coloring (low memory mode)")
    
    # Convert image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
        
    # Get mesh vertices and normalize them to [0, 1] range
    vertices = mesh.vertices.copy()
    
    # Calculate bounding box
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    size = max_bounds - min_bounds
    
    # Normalize vertices to [0, 1]
    normalized_verts = (vertices - min_bounds) / size
    
    # Initialize vertex colors (RGBA)
    vertex_colors = np.ones((len(vertices), 4), dtype=np.uint8) * 255
    
    # Simple projection mapping: use the X and Y coordinates to sample the image
    # This is much simpler than proper UV mapping but uses much less memory
    img_height, img_width = image_array.shape[:2]
    
    # Project to front view
    img_x = (normalized_verts[:, 0] * img_width).astype(int)
    img_y = (normalized_verts[:, 1] * img_height).astype(int)
    
    # Clamp coordinates to image boundaries
    img_x = np.clip(img_x, 0, img_width - 1)
    img_y = np.clip(img_y, 0, img_height - 1)
    
    # Sample colors from image
    for i in range(len(vertices)):
        if image_array.shape[2] == 4:  # RGBA
            vertex_colors[i, :] = image_array[img_y[i], img_x[i], :]
        else:  # RGB
            vertex_colors[i, :3] = image_array[img_y[i], img_x[i], :]
    
    # Create a new mesh with vertex colors
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, 
        faces=mesh.faces,
        vertex_colors=vertex_colors
    )
    
    # Copy metadata from original mesh
    if hasattr(mesh, 'metadata') and mesh.metadata is not None:
        new_mesh.metadata = mesh.metadata.copy()
    
    print("Vertex coloring complete")
    return new_mesh