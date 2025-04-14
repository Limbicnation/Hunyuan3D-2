#!/usr/bin/env python3
"""
This script downloads Hunyuan3D models directly to the local models directory
instead of using the HuggingFace cache.
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import shutil
from huggingface_hub import snapshot_download

def download_model(model_name, subfolder=None, local_dir=None, force=False):
    """Download a model directly to the specified local directory."""
    
    if local_dir is None:
        # Use project root directory
        script_dir = Path(__file__).parent
        local_dir = script_dir.parent / "models"
    else:
        local_dir = Path(local_dir)
    
    local_dir.mkdir(exist_ok=True, parents=True)
    
    # Construct the target directory
    if model_name.startswith("tencent/"):
        target_dir = local_dir / model_name
    else:
        target_dir = local_dir / f"tencent/{model_name}"
    
    if subfolder:
        target_dir = target_dir / subfolder
        
    target_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nDownloading {model_name} {'/' + subfolder if subfolder else ''} to {target_dir}")
    
    # Check if files already exist
    if not force and any(target_dir.iterdir()):
        print(f"Files already exist in {target_dir}. Use --force to overwrite.")
        return
    
    try:
        # Download directly to the target directory
        allow_patterns = [f"{subfolder}/*"] if subfolder else None
        
        result = snapshot_download(
            repo_id=model_name,
            local_dir=target_dir.parent,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False
        )
        
        print(f"Successfully downloaded {model_name} to {result}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download Hunyuan3D models to local directory")
    parser.add_argument("--t2i", action="store_true", help="Download the text-to-image model")
    parser.add_argument("--main", action="store_true", help="Download the main shape generation model")
    parser.add_argument("--turbo", action="store_true", help="Download the turbo model")
    parser.add_argument("--paint", action="store_true", help="Download the paint/texture model")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    
    args = parser.parse_args()
    
    # Set HUGGINGFACE_HUB_CACHE to a temporary location
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path.home() / ".cache" / "huggingface_temp")
    
    # If no specific models are requested, download all
    if not (args.t2i or args.main or args.turbo or args.paint) and not args.all:
        args.all = True
    
    # Download text-to-image model
    if args.t2i or args.all:
        download_model(
            model_name="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
            local_dir=Path.home() / "GitHub" / "Hunyuan3D-2" / "models",
            force=args.force
        )
    
    # Download main model
    if args.main or args.all:
        download_model(
            model_name="tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-dit-v2-0",
            force=args.force
        )
    
    # Download turbo model
    if args.turbo or args.all:
        download_model(
            model_name="tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-dit-v2-0-turbo",
            force=args.force
        )
        
    # Download paint/texture model
    if args.paint or args.all:
        download_model(
            model_name="tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-paint-v2-0",
            force=args.force
        )
    
    print("\nModel downloads complete!")

if __name__ == "__main__":
    main()