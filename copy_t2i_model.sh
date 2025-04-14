#!/bin/bash
# This script copies text-to-image models from the HuggingFace cache to the local models directory

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Determine model directories
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
TARGET_DIR="$MODELS_DIR/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
SOURCE_DIR="$HOME/.cache/huggingface/hub/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled"

echo "="
echo "Text-to-Image Model Copy Utility"
echo "="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory not found: $SOURCE_DIR"
    echo "The model doesn't appear to be in the HuggingFace cache yet."
    echo "Run the gradio app once to download it, then run this script."
    exit 1
fi

# Check if snapshots directory exists in the source
if [ ! -d "$SOURCE_DIR/snapshots" ]; then
    echo "Snapshots directory not found: $SOURCE_DIR/snapshots"
    echo "Looking for model files in the root directory..."
    
    # If no snapshots folder, try copying all files directly
    echo "Copying files from $SOURCE_DIR to $TARGET_DIR"
    cp -r "$SOURCE_DIR"/* "$TARGET_DIR/" 2>/dev/null || true
else
    # Copy files from snapshots folder
    echo "Copying files from $SOURCE_DIR/snapshots to $TARGET_DIR"
    cp -r "$SOURCE_DIR/snapshots"/* "$TARGET_DIR/" 2>/dev/null || true
fi

echo ""
echo "Done! The text-to-image model has been copied to your local models directory."
echo "You can now use it with: python gradio_app.py --enable_t23d"