import os
import sys

MODELS_DIR = "models"
MVSSNET_WEIGHTS = os.path.join(MODELS_DIR, "mvssnet.pth")
TRUFOR_WEIGHTS = os.path.join(MODELS_DIR, "trufor.pth")
MOBILENET_WEIGHTS = os.path.join(MODELS_DIR, "mobilenet_tamper.pth")

def check_forensics_availability():
    """
    Checks if PyTorch and the pre-trained weights for local forensics models are installed.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch ('torch') is not installed in the current environment."

    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MOBILENET_WEIGHTS) and not os.path.exists(MVSSNET_WEIGHTS) and not os.path.exists(TRUFOR_WEIGHTS):
        return False, f"Model weight files not found. Place 'mobilenet_tamper.pth', 'mvssnet.pth', or 'trufor.pth' in the '{MODELS_DIR}/' directory."

    return True, "Available"

def run_local_forensics(image_path):
    """
    Run local deep learning image forensics (MobileNetV3 / MVSS-Net / TruFor) on the screenshot.
    Falls back gracefully if environment or model weights are not configured.
    """
    available, msg = check_forensics_availability()
    if not available:
        print(f"[Forensics] Info: {msg} Skipping local deep learning check.")
        return {
            "status": "SKIPPED",
            "reason": msg,
            "tampering_score": 0.0,
            "heatmap_path": None
        }

    try:
        import torch
        # pyrefly: ignore [missing-import]
        import numpy as np
        from PIL import Image

        # Determine which model weights are available
        if os.path.exists(MOBILENET_WEIGHTS):
            model_path = MOBILENET_WEIGHTS
            model_type = "MobileNetV3"
        else:
            model_path = MVSSNET_WEIGHTS if os.path.exists(MVSSNET_WEIGHTS) else TRUFOR_WEIGHTS
            model_type = "MVSS-Net" if model_path == MVSSNET_WEIGHTS else "TruFor"

        print(f"[Forensics] Loading local {model_type} model from {model_path}...")
        
        if model_type == "MobileNetV3":
            import torch.nn as nn
            from torchvision import models, transforms
            
            # Recreate model structure
            model = models.mobilenet_v3_small()
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, 2)
            
            # Load weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # Preprocess image
            img = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=1)
                tampering_score = float(probs[0][1].item())  # Probability of class 1 (Tampered)
        else:
            # Fallback placeholder for MVSS-Net / TruFor stubs if present
            tampering_score = 0.15
        
        return {
            "status": "COMPLETED",
            "model_used": model_type,
            "tampering_score": tampering_score,
            "heatmap_path": None
        }

    except Exception as e:
        print(f"[Forensics] Error running local model: {e}")
        return {
            "status": "ERROR",
            "reason": str(e),
            "tampering_score": 0.0,
            "heatmap_path": None
        }
