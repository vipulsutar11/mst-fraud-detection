import os
import sys
import argparse
import posixpath
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

import db
import detector

# Configuration
BASE_URL = "https://api.mstblockchain.com/storage/purchase-request/screenshot/"
DEFAULT_CSV = "purchase_request (1).csv"
UPLOADS_DIR = "uploads"
MAX_WORKERS = 8

os.makedirs(UPLOADS_DIR, exist_ok=True)

def process_record(row_idx, row, use_gemini=True):
    """
    Downloads the screenshot and executes the fraud detection pipeline.
    """
    screenshot_name = row.get("payment_screenshot")
    if not isinstance(screenshot_name, str) or not screenshot_name.strip() or screenshot_name == "NULL":
        return {"index": row_idx, "status": "SKIPPED", "reason": "No screenshot name"}

    # Form URL and save path
    image_url = BASE_URL + screenshot_name
    temp_path = os.path.join(UPLOADS_DIR, f"batch_{screenshot_name}")
    
    # 1. Download image if it doesn't exist locally
    if not os.path.exists(temp_path):
        try:
            resp = requests.get(image_url, timeout=15)
            if resp.status_code == 200:
                with open(temp_path, "wb") as f:
                    f.write(resp.content)
            else:
                return {"index": row_idx, "status": "FAILED", "reason": f"Download failed (HTTP {resp.status_code})"}
        except Exception as e:
            return {"index": row_idx, "status": "FAILED", "reason": f"Download error: {str(e)}"}

    # 2. Extract transaction parameters from CSV
    expected_amount = row.get("paid_amount")
    if pd.isna(expected_amount):
        expected_amount = row.get("amount")
    
    created_at_val = row.get("created_at")
    expected_datetime_str = "Not Specified"
    if isinstance(created_at_val, str) and created_at_val != "NULL":
        try:
            # Parse CSV timestamp: 2024-11-07 09:34:04.121
            dt = datetime.strptime(created_at_val.split('.')[0], "%Y-%m-%d %H:%M:%S")
            expected_datetime_str = dt.strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            pass

    # 3. Run Pipeline
    ela_image_path = None
    try:
        # Calculate image hash for metadata storage only
        img_hash = detector.get_image_hash(temp_path)
        
        # EXIF analysis
        exif_warning = detector.analyze_exif(temp_path)
        
        # ELA analysis
        ela_result = detector.analyze_ela(temp_path)
        ela_warning = ela_result.get("ela_warning")
        ela_image_path = ela_result.get("ela_image_path")
        
        # Gemini logic
        if use_gemini and os.getenv("GEMINI_API_KEY"):
            gemini_result = detector.analyze_screenshot_with_gemini(
                image_path=temp_path,
                expected_amount=str(expected_amount),
                expected_datetime_str=expected_datetime_str
            )
            ocr_details = gemini_result.get("ocr_details", {})
            actual_amount = ocr_details.get("amount")
            fraud_probability = gemini_result.get("fraud_probability", 0.0)
            gemini_status = gemini_result.get("gemini_status", "VALID")
            gemini_reason = gemini_result.get("gemini_reason", "")
        else:
            # Gemini-skipped fallback
            ocr_details = {"amount": expected_amount, "reference_id": row.get("transaction_id")}
            actual_amount = expected_amount
            fraud_probability = 0.0
            gemini_status = "VALID"
            gemini_reason = "Gemini audit bypassed or API key not present."

        # Ensure UTR is set in ocr_details
        reference_id = ocr_details.get("reference_id") or row.get("transaction_id")
        if not ocr_details.get("reference_id"):
            ocr_details["reference_id"] = reference_id
            
        # Check duplicate against purchase_requests table
        is_duplicate, parent_id = db.check_duplicate_against_purchase_requests(ocr_details)
        
        # Determine status
        if is_duplicate:
            fraud_probability = 100.0
            gemini_status = "DUPLICATE"
            gemini_reason = f"{gemini_reason} | Duplicate detected: YES (Matches purchase request #{parent_id})"
        else:
            gemini_reason = f"{gemini_reason} | Duplicate detected: NO"

        if ocr_details.get("is_ai_generated"):
            fraud_probability = max(fraud_probability, 95.0)
            gemini_status = "AI_GENERATED"
            gemini_reason = f"{gemini_reason} | AI-Generated Image Detected!"
        elif ela_warning:
            fraud_probability = max(fraud_probability, 80.0)
            gemini_reason = f"{gemini_reason} | ELA warning: {ela_warning}"
        elif exif_warning:
            fraud_probability = max(fraud_probability, 70.0)
            gemini_reason = f"{gemini_reason} | Metadata warning: {exif_warning}"

        if gemini_status in ["SUSPECTED_FRAUD", "INVALID"]:
            fraud_probability = max(fraud_probability, 85.0)

        # Flag only if fraud probability > 70
        if fraud_probability > 70:
            status = "FLAGGED"
        else:
            status = "APPROVED"

        # Check screenshot presence
        ss_present = os.path.exists(temp_path)
        ss_status_prefix = "[Screenshot Present: Yes] " if ss_present else "[Screenshot Present: No] "
        gemini_reason = ss_status_prefix + gemini_reason

        # Save scan to database
        scan_data = {
            "filename": f"batch_{screenshot_name}",
            "buyer_id": f"CSV-USER-{row.get('user_id', 'UNKNOWN')}",
            "fraction_id": "FRAC-LIVE",
            "num_fractions": int(row.get("fractions_count", 1)),
            "expected_amount": expected_amount,
            "actual_amount": actual_amount,
            "image_hash": img_hash,
            "is_duplicate": is_duplicate,
            "duplicate_parent_id": parent_id,
            "exif_warning": exif_warning,
            "ela_warning": ela_warning,
            "ela_image_path": ela_image_path,
            "gemini_status": gemini_status,
            "gemini_reason": gemini_reason,
            "fraud_probability": fraud_probability,
            "ocr_details": ocr_details,
            "status": status
        }
        
        scan_id = db.save_scan(scan_data)
        return {"index": row_idx, "status": "SUCCESS", "scan_id": scan_id, "screenshot": screenshot_name}
    except Exception as e:
        return {"index": row_idx, "status": "ERROR", "reason": str(e)}
    finally:
        # Cleanup downloaded batch screenshot to save disk space
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        # Cleanup ELA image
        if ela_image_path and os.path.exists(ela_image_path):
            try:
                os.remove(ela_image_path)
            except Exception:
                pass

def run_batch(csv_path=DEFAULT_CSV, limit=10, offset=0, use_gemini=True):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    total_records = len(df)
    print(f"Loaded {total_records} records.")
    
    # Slice the dataframe according to offset and limit
    end_idx = min(offset + limit, total_records)
    subset = df.iloc[offset:end_idx]
    
    print(f"Processing records {offset} to {end_idx - 1} (Total: {len(subset)}) using Gemini={use_gemini}...")
    
    results = []
    # Use ThreadPoolExecutor to download/process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_record, offset + idx, row, use_gemini)
            for idx, row in subset.iterrows()
        ]
        
        for future in tqdm(futures, desc="Batch Scans Progress"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                results.append({"status": "CRITICAL_ERROR", "reason": str(e)})
                
    successes = sum(1 for r in results if r.get("status") == "SUCCESS")
    failures = sum(1 for r in results if r.get("status") in ["FAILED", "ERROR", "CRITICAL_ERROR"])
    
    print(f"\nBatch processing finished!")
    print(f"Successfully processed: {successes}")
    print(f"Failed: {failures}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process transaction screenshots from CSV file.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--limit", type=int, default=10, help="Number of records to process")
    parser.add_argument("--offset", type=int, default=0, help="Record index to start from")
    parser.add_argument("--no-gemini", action="store_true", help="Bypass Gemini LLM checks to run faster without API calls")
    
    args = parser.parse_args()
    
    run_batch(
        csv_path=args.csv,
        limit=args.limit,
        offset=args.offset,
        use_gemini=not args.no_gemini
    )
