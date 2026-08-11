import os
import sys
import sqlite3
import json
import random
import requests
# pyrefly: ignore [missing-import]
from PIL import Image

# Setup paths
DB_PATH = "fraud_detection.db"
TRAIN_DIR = "training_data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_tamper.pth")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def download_screenshot(filename):
    """
    Downloads screenshot from MST storage API if not locally present in training_data or uploads.
    """
    local_path = os.path.join(TRAIN_DIR, filename)
    if os.path.exists(local_path):
        return local_path
        
    # Check if it already exists in uploads folder
    uploads_path = os.path.join("uploads", filename)
    if os.path.exists(uploads_path):
        # Copy to training data
        try:
            import shutil
            shutil.copy(uploads_path, local_path)
            return local_path
        except Exception:
            pass

    # Download from API
    url = f"https://api.mstblockchain.com/storage/purchase-request/screenshot/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
    return None

def fetch_labeled_data():
    """
    Queries scans and purchase_requests to get labeled screenshots.
    Class 0 = Valid / Approved
    Class 1 = Tampered / Edited
    """
    print("[Trainer] Querying database for labeled scans...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    dataset = []

    # 1. Fetch from scans table (ONLY keep if file already exists locally in uploads/)
    cursor.execute("SELECT filename, status, ocr_details, gemini_status, gemini_reason FROM scans WHERE ocr_details IS NOT NULL")
    scan_rows = cursor.fetchall()
    for row in scan_rows:
        filename = row["filename"]
        local_path = os.path.join("uploads", filename)
        if os.path.exists(local_path):
            status = row["status"]
            gemini_status = row["gemini_status"]
            is_edited = False
            try:
                ocr = json.loads(row["ocr_details"])
                is_edited = ocr.get("is_edited", False)
            except Exception:
                pass
                
            if status == "APPROVED" and not is_edited:
                label = 0
            elif status == "FLAGGED" and (is_edited or gemini_status == "INVALID" or "tamper" in str(row["gemini_reason"]).lower()):
                label = 1
            else:
                continue
            dataset.append((local_path, label))

    # 2. Fetch from purchase_requests table (downloadable!)
    cursor.execute("SELECT payment_screenshot, ocr_status, ocr_details FROM purchase_requests WHERE ocr_processed = 1 AND payment_screenshot IS NOT NULL")
    pr_rows = cursor.fetchall()
    for row in pr_rows:
        filename = row["payment_screenshot"]
        ocr_status = str(row["ocr_status"]).upper()
        
        is_edited = False
        try:
            ocr = json.loads(row["ocr_details"])
            is_edited = ocr.get("is_edited", False)
        except Exception:
            pass
            
        if ocr_status in ["SUCCESS", "COMPLETED"] and not is_edited:
            label = 0
        elif ocr_status in ["INVALID", "FLAGGED"] or is_edited:
            label = 1
        else:
            continue
        dataset.append((filename, label))
        
    conn.close()
    print(f"[Trainer] Found {len(dataset)} labeled records in database.")
    return dataset

# PyTorch dataset implementation
def get_pytorch_classes():
    import torch
    from torch.utils.data import Dataset
    # pyrefly: ignore [missing-import]
    from torchvision import transforms

    class ScreenshotDataset(Dataset):
        def __init__(self, data_list, transform=None):
            self.data_list = data_list
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        def __len__(self):
            return len(self.data_list)
            
        def __getitem__(self, idx):
            img_path, label = self.data_list[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
                return img_tensor, torch.tensor(label, dtype=torch.long)
            except Exception as e:
                # Return dummy tensor on error
                return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

    return ScreenshotDataset

def train(epochs=3, batch_size=8, lr=0.001):
    # Fetch data
    raw_data = fetch_labeled_data()
    if not raw_data:
        print("[Trainer] Error: No data found to train.")
        return
        
    # Download and prepare local paths
    print("[Trainer] Downloading training screenshots...")
    valid_data = []
    for path_or_name, label in raw_data:
        if os.path.exists(path_or_name):
            valid_data.append((path_or_name, label))
        else:
            local_path = download_screenshot(path_or_name)
            if local_path and os.path.exists(local_path):
                valid_data.append((local_path, label))
            
    print(f"[Trainer] Successfully prepared {len(valid_data)} screenshots for training.")
    if len(valid_data) < 2:
        print("[Trainer] Error: Need at least 2 valid screenshots to train.")
        return

    # Split into train/validation
    random.seed(42)
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * 0.8)
    train_list = valid_data[:split_idx]
    val_list = valid_data[split_idx:]
    
    print(f"[Trainer] Split: {len(train_list)} train, {len(val_list)} validation.")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    # pyrefly: ignore [missing-import]
    from torchvision import models

    # Load custom dataset
    ScreenshotDataset = get_pytorch_classes()
    train_dataset = ScreenshotDataset(train_list)
    val_dataset = ScreenshotDataset(val_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("[Trainer] Loading MobileNetV3 Small model backbone...")
    model = models.mobilenet_v3_small(pretrained=True)
    # Modify classifier final layer for 2 outputs (Valid vs Tampered)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[Trainer] Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        
        # Validation epoch
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
                
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print(f"[Trainer] Saving model weights to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("[Trainer] Training complete!")

if __name__ == "__main__":
    epochs_arg = 3
    if "--epochs" in sys.argv:
        try:
            idx = sys.argv.index("--epochs")
            epochs_arg = int(sys.argv[idx + 1])
        except Exception:
            pass
            
    train(epochs=epochs_arg)
