import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import db
import detector

def ocr_db_records(limit=5, months=None):
    if months:
        print(f"Fetching up to {limit} unprocessed records from purchase_requests table within the last {months} months...")
    else:
        print(f"Fetching up to {limit} unprocessed records from purchase_requests table...")
        
    records = db.get_unprocessed_purchase_requests(limit, months_back=months)
    if not records:
        print("No unprocessed purchase requests found.")
        return

    print(f"Found {len(records)} unprocessed records. Starting download and OCR...")
    for record in records:
        pr_id = record["id"]
        screenshot_name = record["payment_screenshot"]
        expected_amount = record["paid_amount"] or record["amount"]
        created_at = record["created_at"]

        print("-" * 50)
        print(f"Processing ID: {pr_id} | Screenshot: {screenshot_name}")
        
        # This will download the screenshot, run OCR via Gemini, and update the purchase_requests table
        ocr_details = detector.process_and_ocr_purchase_request(
            db_id=pr_id,
            screenshot_name=screenshot_name,
            expected_amount=expected_amount,
            expected_datetime_str=created_at
        )

        if ocr_details and ocr_details.get("reference_id"):
            print("Successfully processed and saved:")
            print(f"  Extracted UTR: {ocr_details.get('reference_id')}")
            print(f"  Extracted Amount: {ocr_details.get('amount')}")
        else:
            print("Failed to process screenshot.")
            
    print("-" * 50)
    print("Batch processing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="Number of records to process")
    parser.add_argument("--months", type=int, default=None, help="Filter for only last N months of records")
    args = parser.parse_args()
    ocr_db_records(args.limit, args.months)
