#!/usr/bin/env python3
"""
Batch Texture Generator for Hunyuan3D-2

This script generates 3D models with textures from images.
It uses advanced multi-directional vertex coloring as a fallback.
"""

import os
import argparse
import glob
import traceback
import time
from PIL import Image
import numpy as np
import trimesh
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def apply_vertex_colors(mesh, image):
    """
    Apply vertex colors to a mesh using a more robust multi-directional projection.
    
    Args:
        mesh: A trimesh mesh object
        image: PIL Image or numpy array
        
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

def process_image(image_path, output_dir, model_path, shape_subfolder, texture_subfolder, 
                 use_texgen=True, remove_bg=True, verbose=False, use_vertex_coloring=False):
    """Process a single image and save the resulting 3D model."""
    if verbose:
        print(f"\n=== Processing image: {image_path} ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Load and prepare image
    if verbose:
        print("Loading image...")
    image = Image.open(image_path).convert("RGBA")
    
    # Remove background if needed
    if remove_bg and image.mode == 'RGB':
        if verbose:
            print("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # Generate shape
    if verbose:
        print(f"Generating 3D shape using {shape_subfolder}...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, 
        subfolder=shape_subfolder
    )
    
    shape_start = time.time()
    mesh = pipeline_shapegen(image=image)[0]
    shape_time = time.time() - shape_start
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    untextured_path = os.path.join(output_dir, f"{base_name}_untextured.glb")
    
    # Save untextured mesh
    if verbose:
        print(f"Saving untextured mesh to {untextured_path}")
    mesh.export(untextured_path)
    
    # Apply texture if requested and not using vertex coloring directly
    if use_texgen and not use_vertex_coloring:
        try:
            if verbose:
                print(f"Applying texture using {texture_subfolder}...")
            
            texture_start = time.time()
            pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
                model_path, 
                subfolder=texture_subfolder
            )
            
            textured_mesh = pipeline_texgen(mesh, image=image)
            texture_time = time.time() - texture_start
            
            textured_path = os.path.join(output_dir, f"{base_name}_textured.glb")
            if verbose:
                print(f"Saving textured mesh to {textured_path}")
            textured_mesh.export(textured_path)
            
            if verbose:
                print(f"Texture generation took {texture_time:.2f} seconds")
            
            total_time = time.time() - start_time
            if verbose:
                print(f"Total processing time: {total_time:.2f} seconds")
            
            return textured_path
        
        except Exception as e:
            if verbose:
                print(f"Error applying texture: {e}")
                print("Falling back to multi-directional vertex coloring...")
            
            # Use improved vertex coloring as fallback
            textured_mesh = None
    
    # If we need to use vertex coloring (either by choice or as fallback)
    if use_vertex_coloring or textured_mesh is None:
        try:
            if verbose:
                print("Applying multi-directional vertex coloring...")
            
            vertex_start = time.time()
            colored_mesh = apply_vertex_colors(mesh, image)
            vertex_time = time.time() - vertex_start
            
            colored_path = os.path.join(output_dir, f"{base_name}_colored.glb")
            if verbose:
                print(f"Saving vertex-colored mesh to {colored_path}")
            colored_mesh.export(colored_path)
            
            if verbose:
                print(f"Vertex coloring took {vertex_time:.2f} seconds")
            
            total_time = time.time() - start_time
            if verbose:
                print(f"Total processing time: {total_time:.2f} seconds")
            
            return colored_path
        
        except Exception as e:
            if verbose:
                print(f"Error applying vertex colors: {e}")
                print(traceback.format_exc())
    
    total_time = time.time() - start_time
    if verbose:
        print(f"Shape generation took {shape_time:.2f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")
    
    return untextured_path

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models from images with texture")
    
    # Main parameters
    parser.add_argument("--input_dir", help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated models")
    
    # Model paths and subfolders
    parser.add_argument("--model_path", default="/home/gero/GitHub/Hunyuan3D-2/models/tencent/Hunyuan3D-2", 
                        help="Path to model directory")
    parser.add_argument("--shape_subfolder", default="hunyuan3d-dit-v2-0-turbo", 
                       help="Subfolder for shape generation model")
    parser.add_argument("--texture_subfolder", default="hunyuan3d-paint-v2-0", 
                       help="Subfolder for texture generation model")
    
    # Processing options
    parser.add_argument("--no_texgen", action="store_true", help="Skip texture generation")
    parser.add_argument("--keep_bg", action="store_true", help="Keep image background")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--single_file", help="Process a single image file instead of a directory")
    parser.add_argument("--use_vertex_coloring", action="store_true", 
                       help="Use improved vertex coloring instead of texture generation")
    
    args = parser.parse_args()
    
    if not args.input_dir and not args.single_file:
        parser.error("Either --input_dir or --single_file must be specified")
    
    if args.verbose:
        print("=== Hunyuan3D-2 Batch Texture Generator ===")
        print(f"Model path: {args.model_path}")
        print(f"Shape subfolder: {args.shape_subfolder}")
        print(f"Texture subfolder: {args.texture_subfolder}")
        print(f"Output directory: {args.output_dir}")
        if args.use_vertex_coloring:
            print("Using multi-directional vertex coloring instead of texture generation")
    
    # Process single file or all files in directory
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} not found")
            return
        
        image_files = [args.single_file]
        if args.verbose:
            print(f"Processing single file: {args.single_file}")
    else:
        # Get all image files in input directory
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            return
            
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        
        if not image_files:
            print(f"No image files found in {args.input_dir}")
            return
        
        if args.verbose:
            print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files):
        try:
            if args.verbose:
                print(f"\nProcessing image {i+1}/{len(image_files)}: {image_path}")
            else:
                print(f"Processing {os.path.basename(image_path)}... ", end="", flush=True)
                
            output_path = process_image(
                image_path, 
                args.output_dir, 
                args.model_path,
                args.shape_subfolder,
                args.texture_subfolder,
                use_texgen=not args.no_texgen,
                remove_bg=not args.keep_bg,
                verbose=args.verbose,
                use_vertex_coloring=args.use_vertex_coloring
            )
            
            if not args.verbose:
                print("Done!")
                
            successful += 1
            
        except Exception as e:
            if args.verbose:
                print(f"Error processing {image_path}: {e}")
                print(traceback.format_exc())
            else:
                print(f"Failed! Error: {e}")
                
            failed += 1
    
    print(f"\nProcessing complete! Successfully processed {successful} images, failed: {failed}")
    if successful > 0:
        print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()