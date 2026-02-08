# VSRFI - Video Super Resolution + Frame Interpolation for ComfyUI

A unified ComfyUI custom node that combines **FlashVSR** (video super-resolution) and **GIMM-VFI** (frame interpolation) into a single node.

## Features

- **Video Super Resolution**: Upscale videos 2x-8x using FlashVSR's diffusion-based approach
- **Frame Interpolation**: Generate smooth intermediate frames using GIMM-VFI's implicit neural representation
- **Automatic Model Downloading**: Models are downloaded automatically from HuggingFace on first use
- **Tiled Processing**: Process large videos with limited VRAM using automatic tiling
- **Flexible Processing**: Control tile size and resolution limits

## Installation

### 1. Clone this repository into your ComfyUI custom nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/VSRFI.git
```

### 2. Install dependencies:

```bash
cd VSRFI
pip install -r requirements.txt
```

**Note**: CuPy (required for GIMM-VFI frame interpolation) must be installed separately with the version matching your CUDA toolkit:
```bash
# Check your CUDA version first:
nvcc --version

# Then install the matching CuPy package:
pip install cupy-cuda12x   # For CUDA 12.x
# OR
pip install cupy-cuda11x   # For CUDA 11.x
```

### 3. Restart ComfyUI

The models will be downloaded automatically when you first use the node.

## Model Downloads

On first use, the node will automatically download:

1. **FlashVSR-v1.1** (~4GB) from [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)
   - Downloaded to: `ComfyUI/models/FlashVSR-v1.1/`

2. **GIMM-VFI** (~500MB) from [Kijai/GIMM-VFI_safetensors](https://huggingface.co/Kijai/GIMM-VFI_safetensors)
   - Downloaded to: `ComfyUI/models/interpolation/gimm-vfi/`

Models are cached and reused across sessions.

## Usage

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **video_path** | STRING | "" | Path to input video file |
| **output_path** | STRING | "" | Path for output video (auto-generated if empty) |
| **scale** | INT | 2 | Super-resolution scale factor (1-8x) |
| **frames_per_chunk** | INT | 100 | Number of frames processed at once (reduce for lower VRAM) |
| **max_tile_kilopixels** | INT | 4000 | Max tile size in kilopixels for spatial tiling (0=no tiling) |
| **interpolation_factor** | INT | 2 | Frame interpolation multiplier (1=no interpolation, 2=double framerate, etc.) |


