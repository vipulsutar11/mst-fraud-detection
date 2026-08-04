import sqlite3
import json
import os

DB_PATH = os.environ.get("DB_PATH", "fraud_detection.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            buyer_id TEXT,
            fraction_id TEXT,
            num_fractions INTEGER DEFAULT 1,
            expected_amount REAL,
            actual_amount REAL,
            image_hash TEXT,
            is_duplicate INTEGER,
            duplicate_parent_id INTEGER,
            exif_warning TEXT,
            ela_warning TEXT,
            ela_image_path TEXT,
            gemini_status TEXT,
            gemini_reason TEXT,
            fraud_probability REAL,
            ocr_details TEXT,
            status TEXT
        )
    """)
    # Migration: Add num_fractions column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN num_fractions INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    # Migration: Add ela_warning column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN ela_warning TEXT")
    except sqlite3.OperationalError:
        pass
    # Migration: Add ela_image_path column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN ela_image_path TEXT")
    except sqlite3.OperationalError:
        pass

    # Create purchase_requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS purchase_requests (
            id INTEGER PRIMARY KEY,
            payment_screenshot TEXT,
            transaction_id TEXT,
            amount REAL,
            paid_amount REAL,
            created_at TEXT,
            user_id TEXT,
            ocr_processed INTEGER DEFAULT 0,
            ocr_reference_id TEXT,
            ocr_amount REAL,
            ocr_date TEXT,
            ocr_time TEXT,
            ocr_status TEXT,
            ocr_details TEXT
        )
    """)
    conn.commit()
    conn.close()

def seed_purchase_requests(csv_path="Updated_Data.csv"):
    """
    Seed or update purchase_requests table from the CSV file.
    """
    if not os.path.exists(csv_path):
        csv_path = "purchase_request (1).csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found, skipping seeding.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM purchase_requests")
    count = cursor.fetchone()[0]
    if count > 0 and csv_path != "Updated_Data.csv":
        conn.close()
        return

    print("Seeding purchase_requests table from CSV...")
    import pandas as pd
    try:
        # Load CSV using pandas
        df = pd.read_csv(csv_path)
        # Select relevant columns and handle NaN
        records = []
        for _, row in df.iterrows():
            rec_id = int(row.get("id"))
            screenshot = str(row.get("payment_screenshot")) if pd.notna(row.get("payment_screenshot")) else None
            tx_id = str(row.get("transaction_id")) if pd.notna(row.get("transaction_id")) else None
            amount = float(row.get("amount")) if pd.notna(row.get("amount")) else None
            paid_amount = float(row.get("paid_amount")) if pd.notna(row.get("paid_amount")) else None
            created_at = str(row.get("created_at")) if pd.notna(row.get("created_at")) else None
            user_id = str(row.get("user_id")) if pd.notna(row.get("user_id")) else None

            records.append((rec_id, screenshot, tx_id, amount, paid_amount, created_at, user_id))

        cursor.executemany("""
            INSERT INTO purchase_requests (id, payment_screenshot, transaction_id, amount, paid_amount, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        print(f"Seeded {len(records)} purchase requests.")
    except Exception as e:
        print(f"Error seeding purchase requests: {e}")
    finally:
        conn.close()

def save_scan(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scans (
            filename, buyer_id, fraction_id, num_fractions, expected_amount, actual_amount,
            image_hash, is_duplicate, duplicate_parent_id, exif_warning, ela_warning, ela_image_path,
            gemini_status, gemini_reason, fraud_probability, ocr_details, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("filename"),
        data.get("buyer_id"),
        data.get("fraction_id"),
        data.get("num_fractions", 1),
        data.get("expected_amount"),
        data.get("actual_amount"),
        data.get("image_hash"),
        1 if data.get("is_duplicate") else 0,
        data.get("duplicate_parent_id"),
        data.get("exif_warning"),
        data.get("ela_warning"),
        data.get("ela_image_path"),
        data.get("gemini_status"),
        data.get("gemini_reason"),
        data.get("fraud_probability"),
        json.dumps(data.get("ocr_details", {})),
        data.get("status")
    ))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id

def get_scans():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        d = dict(r)
        d["ocr_details"] = json.loads(d["ocr_details"]) if d["ocr_details"] else {}
        result.append(d)
    return result

def check_duplicate_hash(new_hash_str, threshold=0):
    """
    Check if a matching perceptual hash already exists.
    Returns (is_duplicate, parent_id)
    (Disabled: now always returns False, None to rely strictly on Reference ID)
    """
    return False, None

def check_duplicate_reference_id(reference_id):
    """
    Check if a matching reference_id (UTR) already exists in past scans.
    Returns (is_duplicate, parent_id)
    """
    if not reference_id:
        return False, None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, ocr_details FROM scans WHERE ocr_details IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    for row_id, ocr_details_str in rows:
        try:
            details = json.loads(ocr_details_str)
            if details.get("reference_id") == reference_id:
                return True, row_id
        except Exception:
            continue

    return False, None

def get_purchase_request_by_tx_id(transaction_id):
    """
    Retrieve a purchase request by transaction_id (UTR).
    """
    if not transaction_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchase_requests WHERE transaction_id = ? OR ocr_reference_id = ?", (transaction_id, transaction_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_unprocessed_purchase_requests(limit=10, months_back=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if months_back:
        from datetime import datetime, timedelta
        threshold_date = datetime.now() - timedelta(days=months_back * 30)
        threshold_str = threshold_date.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            SELECT * FROM purchase_requests 
            WHERE ocr_processed = 0 
              AND payment_screenshot IS NOT NULL 
              AND payment_screenshot != 'NULL'
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (threshold_str, limit))
    else:
        cursor.execute("SELECT * FROM purchase_requests WHERE ocr_processed = 0 AND payment_screenshot IS NOT NULL AND payment_screenshot != 'NULL' LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_purchase_request_ocr(db_id, ocr_details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE purchase_requests
        SET ocr_processed = 1,
            ocr_reference_id = ?,
            ocr_amount = ?,
            ocr_date = ?,
            ocr_time = ?,
            ocr_status = ?,
            ocr_details = ?
        WHERE id = ?
    """, (
        ocr_details.get("reference_id"),
        ocr_details.get("amount"),
        ocr_details.get("date"),
        ocr_details.get("time"),
        ocr_details.get("payment_status"),
        json.dumps(ocr_details),
        db_id
    ))
    conn.commit()
    conn.close()

def check_duplicate_against_purchase_requests(new_ocr_details):
    """
    Compare new OCR details against purchase_requests AND past audit scans.
    Returns (is_duplicate, parent_id/purchase_request_id)
    """
    reference_id = new_ocr_details.get("reference_id")
    if not reference_id:
        return False, None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Search in purchase_requests OCR processed records
    cursor.execute("""
        SELECT id FROM purchase_requests 
        WHERE ocr_processed = 1 AND (
            ocr_reference_id = ? 
            OR (LENGTH(?) >= 8 AND ocr_reference_id LIKE ?) 
            OR (LENGTH(ocr_reference_id) >= 8 AND ? LIKE '%' || ocr_reference_id || '%')
        )
    """, (reference_id, reference_id, f"%{reference_id}%", reference_id))
    row = cursor.fetchone()
    if row:
        conn.close()
        return True, row[0]

    # 2. Search in purchase_requests CSV transaction_id records
    cursor.execute("""
        SELECT id FROM purchase_requests 
        WHERE transaction_id = ? 
           OR (LENGTH(?) >= 8 AND transaction_id LIKE ?) 
           OR (LENGTH(transaction_id) >= 8 AND ? LIKE '%' || transaction_id || '%')
    """, (reference_id, reference_id, f"%{reference_id}%", reference_id))
    row = cursor.fetchone()
    if row:
        pr_id = row[0]
        conn.close()
        return True, pr_id

    # 3. Search in past audit scans (scans table) by reference_id or image_hash
    cursor.execute("SELECT id, ocr_details FROM scans WHERE ocr_details IS NOT NULL ORDER BY id ASC")
    scan_rows = cursor.fetchall()
    for s_id, ocr_json in scan_rows:
        try:
            details = json.loads(ocr_json) if ocr_json else {}
            past_ref = details.get("reference_id")
            if past_ref and (past_ref == reference_id or (len(reference_id) >= 8 and reference_id in str(past_ref))):
                conn.close()
                return True, s_id
        except Exception:
            continue

    conn.close()
    return False, None

init_db()
seed_purchase_requests()

