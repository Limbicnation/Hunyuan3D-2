#!/bin/bash
# Direct download from Hugging Face for specific model components

# Set target directory
TARGET_DIR="/home/gero/GitHub/Hunyuan3D-2/models"
mkdir -p "$TARGET_DIR"

# Install huggingface_hub if not already installed
pip install huggingface_hub

# Use Python to download specific model components
python3 <<EOF
import os
import huggingface_hub

# Define base target directory
target_dir = "$TARGET_DIR"

# Models and components to download
downloads = [
    # Shape models
    {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-dit-v2-0-turbo", "local_dir": f"{target_dir}/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0-turbo"},
    
    # Texture models
    {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-delight-v2-0", "local_dir": f"{target_dir}/tencent/Hunyuan3D-2/hunyuan3d-delight-v2-0"},
    {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-paint-v2-0", "local_dir": f"{target_dir}/tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0"},
]

for download in downloads:
    repo = download["repo"]
    subfolder = download["subfolder"]
    local_dir = download["local_dir"]
    
    print(f"\nDownloading {repo}/{subfolder} to {local_dir}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download files for this specific subfolder
        huggingface_hub.snapshot_download(
            repo_id=repo,
            allow_patterns=[f"{subfolder}/**"],
            local_dir=os.path.dirname(local_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"Successfully downloaded {repo}/{subfolder}")
    except Exception as e:
        print(f"Error downloading {repo}/{subfolder}: {e}")
EOF

echo "Download process complete. Check output for any errors."
