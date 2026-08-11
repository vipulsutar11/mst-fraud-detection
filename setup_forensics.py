import os
import sys
import requests

MODELS_DIR = "models"
MVSSNET_URL = "https://github.com/WZTUST/MVSSNet/releases/download/v1.0/mvssnet.pth"  # Example release URL
TRUFOR_URL = "https://github.com/grip-unina/TruFor/releases/download/weights/trufor.pth"  # Example release URL

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed successfully.")
        return True
    except Exception as e:
        print(f"Failed to download weights: {e}")
        return False

def setup():
    print("=" * 60)
    print("Local Forensics (MVSS-Net / TruFor) Setup Helper")
    print("=" * 60)

    # 1. Check Python dependencies
    print("\n[1] Checking local Python environment...")
    try:
        import torch
        print("  - PyTorch: Installed (version:", torch.__version__, ")")
        print("  - Device available:", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
    except ImportError:
        print("  - Warning: PyTorch ('torch') is not installed in your current .venv.")
        print("  - To install PyTorch run:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/ml/cu118")
        print("      (or simply: pip install torch)")

    # 2. Check/Download model weights
    print("\n[2] Checking model weights...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    mvssnet_path = os.path.join(MODELS_DIR, "mvssnet.pth")
    trufor_path = os.path.join(MODELS_DIR, "trufor.pth")

    if os.path.exists(mvssnet_path) or os.path.exists(trufor_path):
        print("  - Local model weights found in models/ directory.")
    else:
        print("  - No weights found. Would you like to download them?")
        print("  - Run the following command to download MVSS-Net weights:")
        print("      python setup_forensics.py --download")

if __name__ == "__main__":
    if "--download" in sys.argv:
        os.makedirs(MODELS_DIR, exist_ok=True)
        dest = os.path.join(MODELS_DIR, "mvssnet.pth")
        download_file(MVSSNET_URL, dest)
    else:
        setup()
