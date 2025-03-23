# vertex_color.py
import trimesh
import numpy as np
from PIL import Image
import sys
import os

def apply_vertex_colors(mesh, image):
    """
    Apply vertex colors to a mesh using a more robust multi-directional projection.
    
    Args:
        mesh: A trimesh mesh object
        image: PIL Image object
        
    Returns:
        A mesh with vertex colors applied
    """
    # Convert PIL image to numpy array if needed
    if not isinstance(image, np.ndarray):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Get image dimensions
    img_h, img_w = img_array.shape[:2]
    
    # Get mesh properties
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    
    # Create vertex colors array (RGBA)
    vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    
    # Calculate mesh bounding box and center
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    
    # Normalize vertices around center
    centered_verts = vertices - center
    
    # Get maximum distance from center for scaling
    max_dist = np.max(np.abs(centered_verts))
    scaled_verts = centered_verts / max_dist if max_dist > 0 else centered_verts
    
    # Create multiple projection directions for better coverage
    # Front, back, left, right, top, bottom projections
    directions = [
        [0, 0, 1],   # Front (Z+)
        [0, 0, -1],  # Back (Z-)
        [-1, 0, 0],  # Left (X-)
        [1, 0, 0],   # Right (X+)
        [0, 1, 0],   # Top (Y+)
        [0, -1, 0]   # Bottom (Y-)
    ]
    
    # Initialize weights for each direction (based on normal alignment)
    weights = np.zeros((len(vertices), len(directions)))
    
    # Calculate weights for each direction based on normal alignment
    for i, direction in enumerate(directions):
        # Convert direction to unit vector
        dir_vector = np.array(direction, dtype=float)
        dir_vector = dir_vector / np.linalg.norm(dir_vector)
        
        # Calculate alignment of each vertex normal with this direction
        # Dot product of normal and direction gives cosine of angle
        alignment = np.abs(np.dot(normals, dir_vector))
        
        # Higher value = better alignment
        weights[:, i] = alignment
    
    # Normalize weights to sum to 1 for each vertex
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, where=row_sums!=0)
    
    # For each direction, calculate color contribution
    for i, direction in enumerate(directions):
        # Skip directions with no weight
        if np.max(weights[:, i]) < 0.01:
            continue
            
        # Project vertices onto image plane perpendicular to this direction
        if direction[0] != 0:  # X direction (side view)
            # Use Y and Z for image coordinates
            x_img = (scaled_verts[:, 1] * 0.5 + 0.5) * img_w
            y_img = (0.5 - scaled_verts[:, 2] * 0.5) * img_h
        elif direction[1] != 0:  # Y direction (top/bottom view)
            # Use X and Z for image coordinates
            x_img = (scaled_verts[:, 0] * 0.5 + 0.5) * img_w
            y_img = (0.5 - scaled_verts[:, 2] * 0.5) * img_h
        else:  # Z direction (front/back view)
            # Use X and Y for image coordinates
            x_img = (scaled_verts[:, 0] * 0.5 + 0.5) * img_w
            y_img = (0.5 - scaled_verts[:, 1] * 0.5) * img_h
        
        # Clamp to image boundaries
        x_img = np.clip(x_img, 0, img_w - 1).astype(np.int32)
        y_img = np.clip(y_img, 0, img_h - 1).astype(np.int32)
        
        # Sample colors from image
        for j, (x, y) in enumerate(zip(x_img, y_img)):
            # Only process if this direction has weight for this vertex
            if weights[j, i] > 0.01:
                # Get color at this pixel
                color = img_array[y, x]
                
                # Convert to RGBA if needed
                if len(color) == 3:
                    color_rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
                else:
                    color_rgba = color
                
                # Blend color based on weight
                if vertex_colors[j, 3] == 0:  # If no color yet, use this one directly
                    vertex_colors[j] = color_rgba
                else:
                    # Otherwise blend with existing color
                    w = weights[j, i]
                    vertex_colors[j] = (vertex_colors[j] * (1 - w) + color_rgba * w).astype(np.uint8)
    
    # For any vertices with no color (weight sum was 0), use nearest neighbor sampling
    mask = vertex_colors[:, 3] == 0
    if np.any(mask):
        # For these vertices, just use front projection
        x_img = (scaled_verts[mask, 0] * 0.5 + 0.5) * img_w
        y_img = (0.5 - scaled_verts[mask, 1] * 0.5) * img_h
        
        x_img = np.clip(x_img, 0, img_w - 1).astype(np.int32)
        y_img = np.clip(y_img, 0, img_h - 1).astype(np.int32)
        
        for idx, (x, y) in zip(np.where(mask)[0], zip(x_img, y_img)):
            color = img_array[y, x]
            if len(color) == 3:
                vertex_colors[idx] = [color[0], color[1], color[2], 255]
            else:
                vertex_colors[idx] = color
    
    # Apply to mesh
    colored_mesh = mesh.copy()
    colored_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    
    return colored_mesh

def process_files(mesh_path, image_path, output_path):
    """
    Load mesh and image files, apply vertex colors, and save the result.
    
    Args:
        mesh_path: Path to the input mesh file
        image_path: Path to the input image file
        output_path: Path to save the colored mesh
    """
    print(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    print(f"Loading image from {image_path}")
    image = Image.open(image_path)
    
    print("Applying vertex colors...")
    colored_mesh = apply_vertex_colors(mesh, image)
    
    print(f"Saving colored mesh to {output_path}")
    colored_mesh.export(output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python vertex_color.py [mesh_path] [image_path] [output_path]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]
    
    process_files(mesh_path, image_path, output_path)