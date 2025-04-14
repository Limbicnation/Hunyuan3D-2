#!/bin/bash
# This script compiles the C++ renderers needed for texture generation in the hunyuan3d conda environment

# Check if we're in the hunyuan3d conda environment
if [[ "$CONDA_DEFAULT_ENV" != "hunyuan3d" ]]; then
    echo "ERROR: You need to activate the 'hunyuan3d' conda environment first!"
    echo "Run: conda activate hunyuan3d"
    exit 1
fi

set -e  # Exit on any error

echo "============================"
echo "Compiling Renderers for Hunyuan3D"
echo "Using conda environment: $CONDA_DEFAULT_ENV"
echo "============================"

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Compile differentiable renderer
echo ""
echo "Compiling differentiable renderer..."
cd hy3dgen/texgen/differentiable_renderer
pip install -e .

# Compile custom rasterizer if CUDA is available
if command -v nvcc &> /dev/null; then
    echo ""
    echo "Compiling custom rasterizer (CUDA)..."
    cd ../custom_rasterizer
    pip install -e .
else
    echo ""
    echo "CUDA compiler (nvcc) not found. Skipping custom rasterizer compilation."
    echo "Texture generation may still work with limited functionality."
fi

echo ""
echo "Compilation complete!"
echo "You can now run the gradio app with texture support:"
echo "python gradio_app.py --enable_t23d"