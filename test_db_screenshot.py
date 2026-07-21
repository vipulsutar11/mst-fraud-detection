import os
import requests
import db
import detector

def test_existing_screenshot():
    # 1. Fetch a purchase request from the database that has a transaction ID
    conn = db.sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, payment_screenshot, transaction_id, amount 
        FROM purchase_requests 
        WHERE transaction_id IS NOT NULL AND transaction_id != 'NULL' 
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No purchase request found with a transaction ID in the database.")
        return

    pr_id, screenshot_name, transaction_id, amount = row
    print("=" * 60)
    print("Found screenshot record in database:")
    print(f"  ID: {pr_id}")
    print(f"  Screenshot Name: {screenshot_name}")
    print(f"  Transaction ID (UTR): {transaction_id}")
    print(f"  Amount: {amount}")
    print("=" * 60)

    # 2. Construct the URL and download the screenshot
    print("\nDownloading screenshot to test...")
    local_path = detector.download_purchase_request_screenshot(screenshot_name)
    if not local_path or not os.path.exists(local_path):
        print(f"Could not download screenshot from: https://api.mstblockchain.com/storage/purchase-request/screenshot/{screenshot_name}")
        return
    print(f"Downloaded to: {local_path}")

    # 3. Simulate OCR result with matching transaction_id (UTR)
    # (Since Gemini OCR will read this exact transaction ID from the screenshot)
    simulated_ocr = {
        "reference_id": transaction_id,
        "amount": amount
    }

    # 4. Run the duplicate check logic
    print("\nRunning check_duplicate_against_purchase_requests...")
    is_duplicate, matched_id = db.check_duplicate_against_purchase_requests(simulated_ocr)
    
    print("\n[Results]")
    print(f"  Is Duplicate: {is_duplicate}")
    print(f"  Matched Purchase Request ID: {matched_id}")
    print("=" * 60)

    # Cleanup downloaded test file
    if os.path.exists(local_path):
        os.remove(local_path)

if __name__ == "__main__":
    test_existing_screenshot()
