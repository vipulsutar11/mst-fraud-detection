import os
import sys
import json
from datetime import datetime
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

import detector
import db

def test_image(image_path, expected_amount=250.00, expected_datetime_str=None):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    print("=" * 60)
    print(f"Testing detection pipeline for: {image_path}")
    print("=" * 60)

    # 1. Perceptual Hash / Duplicate Detection
    print("\n[1] Running Perceptual Hash Detection...")
    img_hash = detector.get_image_hash(image_path)
    print(f"  Perceptual Hash: {img_hash}")
    is_duplicate, parent_id = db.check_duplicate_hash(img_hash)
    print(f"  Is Duplicate (Hash): {is_duplicate} (Parent ID: {parent_id})")

    # 2. EXIF Metadata analysis
    print("\n[2] Running EXIF Metadata Analysis...")
    exif_warning = detector.analyze_exif(image_path)
    print(f"  EXIF Warning: {exif_warning}")

    # 3. Gemini Multimodal OCR and Fraud Detection
    print("\n[3] Running Gemini Multimodal OCR and Fraud Detection...")
    if not expected_datetime_str:
        # Default to current time formatted
        expected_datetime_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    
    print(f"  Expected Amount: {expected_amount}")
    print(f"  Expected DateTime: {expected_datetime_str}")
    
    gemini_result = detector.analyze_screenshot_with_gemini(
        image_path=image_path,
        expected_amount=expected_amount,
        expected_datetime_str=expected_datetime_str
    )
    
    print("\n[Results]")
    print(json.dumps(gemini_result, indent=2))
    print("=" * 60)

    # 4. Save to Database
    print("\n[4] Saving scan to the database...")
    ocr_details = gemini_result.get("ocr_details", {})
    status = "APPROVED" if gemini_result.get("gemini_status") == "VALID" else "FLAGGED"
    if is_duplicate:
        status = "REJECTED"
        
    scan_data = {
        "filename": os.path.basename(image_path),
        "buyer_id": "TEST-USER",
        "fraction_id": "FRAC-LIVE",
        "num_fractions": 1,
        "expected_amount": expected_amount,
        "actual_amount": ocr_details.get("amount"),
        "image_hash": img_hash,
        "is_duplicate": is_duplicate,
        "duplicate_parent_id": parent_id,
        "exif_warning": exif_warning,
        "gemini_status": gemini_result.get("gemini_status"),
        "gemini_reason": gemini_result.get("gemini_reason"),
        "fraud_probability": gemini_result.get("fraud_probability", 0.0),
        "ocr_details": ocr_details,
        "status": status
    }
    
    scan_id = db.save_scan(scan_data)
    print(f"  Scan successfully saved with ID: {scan_id}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <path_to_image> [expected_amount] [expected_datetime_str]")
        print("Example: python test_detector.py uploads/test_payment.png 250.00 '2026-07-17 10:30 AM'")
    else:
        path = sys.argv[1]
        amount = float(sys.argv[2]) if len(sys.argv) > 2 else 250.00
        dt_str = sys.argv[3] if len(sys.argv) > 3 else None
        test_image(path, amount, dt_str)

