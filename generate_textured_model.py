from PIL import Image
import argparse
import os
import inspect

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, default="output.glb", help="Output model filename")
parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Model path")
parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-0-turbo", help="Subfolder")
parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
parser.add_argument("--low_vram_mode", action="store_true", help="Enable low VRAM mode")
parser.add_argument("--enable_flashvdm", action="store_true", help="Enable FlashVDM")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

print(f"Loading models from {args.model_path}...")

# Initialize pipeline for shape generation
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    args.model_path, 
    subfolder=args.subfolder,
    use_safetensors=True
)

if args.enable_flashvdm:
    pipeline_shapegen.enable_flashvdm()

# Try CPU offload only if supported
if args.low_vram_mode:
    try:
        pipeline_shapegen.enable_model_cpu_offload()
        print("Enabled CPU offloading for shape generation")
    except AttributeError:
        print("CPU offloading not supported for shape generation pipeline")

# Initialize pipeline for texture generation
print("Loading texture pipeline...")
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(args.model_path)

if args.low_vram_mode:
    try:
        pipeline_texgen.enable_model_cpu_offload()
        print("Enabled CPU offloading for texture generation")
    except AttributeError:
        print("CPU offloading not supported for texture generation pipeline")

# Verify the image path exists
if not os.path.exists(args.image):
    raise ValueError(f"Image path does not exist: {args.image}")

# Load and preprocess image
print(f"Processing image: {args.image}")
image = Image.open(args.image).convert("RGBA")
if image.mode == 'RGB':
    print("Removing background...")
    rembg = BackgroundRemover()
    image = rembg(image)

# Generate shape
print("Generating 3D shape...")
mesh = pipeline_shapegen(
    image=image,
    num_inference_steps=args.steps
)[0]

# Extract untextured model
mesh.export("untextured_" + args.output)
print(f"Saved untextured model to: untextured_{args.output}")

# Try texture generation with multiple approaches
print("Generating texture...")
try:
    # Debug output
    if args.debug:
        # Dump the structure of the texture pipeline
        print("Texture pipeline structure:")
        for attr_name in dir(pipeline_texgen):
            if not attr_name.startswith('_'):
                attr = getattr(pipeline_texgen, attr_name)
                print(f"  {attr_name}: {type(attr)}")
                
                # If this is an object, also show its attributes
                if hasattr(attr, '__dict__'):
                    for sub_attr_name in dir(attr):
                        if not sub_attr_name.startswith('_'):
                            try:
                                sub_attr = getattr(attr, sub_attr_name)
                                print(f"    - {sub_attr_name}: {type(sub_attr)}")
                            except Exception as e:
                                print(f"    - {sub_attr_name}: Error: {e}")
    
    # Try to find any method directly on the model that might work
    if hasattr(pipeline_texgen, 'models'):
        print("Pipeline has models attribute")
        # Check if there's a relevant model that might process meshes
        for model_name, model in pipeline_texgen.models.items():
            print(f"Checking model: {model_name}")
            
            # Try to find methods that might be useful
            if hasattr(model, 'process') and callable(getattr(model, 'process')):
                print(f"Using {model_name}.process")
                textured_mesh = model.process(mesh, image)
                break
            elif hasattr(model, 'render') and callable(getattr(model, 'render')):
                print(f"Using {model_name}.render")
                textured_mesh = model.render(mesh, image)
                break
            elif hasattr(model, '__call__') and callable(getattr(model, '__call__')):
                print(f"Using {model_name}.__call__")
                textured_mesh = model(mesh, image=image)
                break
    else:
        raise RuntimeError("No suitable method found for texturing")
    
    # Save textured model
    textured_mesh.export(args.output)
    print(f"Successfully saved textured model to: {args.output}")
except Exception as e:
    print(f"Error generating texture: {e}")
    print("Saving untextured model only.")
    
    if args.debug:
        import traceback
        traceback.print_exc()