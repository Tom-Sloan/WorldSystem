#!/bin/bash
# frame_processor/install_models.sh

echo "Installing SAM2 and FastSAM models..."

# Create models directory
mkdir -p models

# Function to download if not exists
download_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -f "$dest" ]; then
    echo "Skipping download, already exists: $dest"
  else
    echo "Downloading $(basename "$dest")..."
    wget -O "$dest" "$url"
  fi
}

# Download SAM2 Hiera Large checkpoint
download_if_missing "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" "models/sam2_hiera_large.pt"

# Optional: Download other SAM2 models
echo "Available SAM2 models (download manually if needed):"
echo "  - Hiera Tiny: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
echo "  - Hiera Small: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
echo "  - Hiera Base+: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

download_if_missing "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" "models/sam2_hiera_tiny.pt"
download_if_missing "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt" "models/sam2_hiera_small.pt"
download_if_missing "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt" "models/sam2_hiera_base_plus.pt"

# Download FastSAM model
download_if_missing "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.0.1/FastSAM-x.pt" "models/FastSAM-x.pt"

# Verify downloads
echo "Verifying downloads..."
ls -lh models/

echo "Model installation complete!"