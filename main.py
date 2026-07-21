import os
import shutil
import uuid
import requests
from datetime import datetime
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, BackgroundTasks
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

import db
import detector

app = FastAPI(title="MST Fraction Purchase Verification Engine")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

FRACTIONS_DB = {
    "FRAC-LIVE": {"name": "Live Node Fraction (MST Blockchain)", "price": 0.0}
}

def fetch_live_fraction_price() -> float:
    try:
        response = requests.post("https://api.mstblockchain.com/purchase/node-fraction-price", timeout=10)
        if response.status_code == 201:
            data = response.json()
            if data.get("status") == "success" and "currentNodeFractionPrice" in data:
                return float(data["currentNodeFractionPrice"])
    except Exception as e:
        print(f"Error fetching live fraction price: {e}")
    raise HTTPException(status_code=502, detail="Failed to retrieve live fraction price from MST Blockchain API.")

def safe_float(val, default=None) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("₹", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return default

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Serve the dashboard static HTML page directly to avoid Jinja2 compatibility issues on Python 3.14.
    """
    html_path = os.path.join("templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/api/fractions")
async def get_fractions():
    """
    Mock API to retrieve current fraction details and their prices.
    """
    try:
        live_price = fetch_live_fraction_price()
        expected_amount = round(live_price * 1.18, 2)
        FRACTIONS_DB["FRAC-LIVE"]["price"] = expected_amount
    except Exception as e:
        print(f"Warning: could not update live fraction price: {e}")
    return JSONResponse(content=FRACTIONS_DB)

@app.get("/api/history")
async def get_history():
    """
    Retrieve historical transaction validation results.
    """
    history = db.get_scans()
    return JSONResponse(content=history)

@app.get("/api/health")
async def health_check():
    """
    Perform diagnostic checks on database, environment, and external APIs.
    """
    checks = {}
    
    # 1. Database Check
    try:
        scans = db.get_scans()
        checks["database"] = {
            "status": "healthy",
            "message": f"Successfully connected. Found {len(scans)} records."
        }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "message": f"Failed to connect: {str(e)}"
        }

    # 2. Gemini API Check
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        masked_key = gemini_key[:4] + "..." + gemini_key[-4:] if len(gemini_key) > 8 else "configured"
        checks["gemini_api"] = {
            "status": "configured",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "key": masked_key
        }
    else:
        checks["gemini_api"] = {
            "status": "unconfigured",
            "message": "GEMINI_API_KEY environment variable is missing."
        }

    # 3. MST Blockchain API Check
    try:
        response = requests.post("https://api.mstblockchain.com/purchase/node-fraction-price", timeout=5)
        checks["blockchain_api"] = {
            "status": "accessible" if response.status_code == 201 else "unhealthy",
            "status_code": response.status_code
        }
    except Exception as e:
        checks["blockchain_api"] = {
            "status": "inaccessible",
            "message": str(e)
        }

    # Overall status
    overall = "healthy"
    for check in checks.values():
        if check.get("status") in ["unhealthy", "unconfigured", "inaccessible"]:
            overall = "degraded"
            
    return JSONResponse(status_code=200 if overall == "healthy" else 207, content={
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "checks": checks
    })

@app.post("/api/detect")
async def detect_screenshot(screenshot: UploadFile = File(...)):
    """
    Perform a complete screenshot audit (OCR, Visual Manipulation, Hash, EXIF, ELA)
    requiring ONLY the screenshot file, checks against duplicates in the database,
    saves the results to the database, and returns the verdict.
    """
    file_ext = os.path.splitext(screenshot.filename)[1] or ".png"
    temp_filename = f"detect_{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join("uploads", temp_filename)
    
    ela_image_path = None
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(screenshot.file, buffer)
        
    try:
        # 1. Calculate image hash for metadata storage only
        img_hash = detector.get_image_hash(temp_path)
        
        # 2. Metadata Analysis
        exif_warning = detector.analyze_exif(temp_path)
        
        # 3. ELA Analysis
        ela_result = detector.analyze_ela(temp_path)
        ela_warning = ela_result["ela_warning"]
        ela_image_path = ela_result["ela_image_path"]
        
        # 4. Gemini Vision Audit
        gemini_result = detector.analyze_screenshot_with_gemini(
            image_path=temp_path,
            expected_amount="Not Specified",
            expected_datetime_str="Not Specified"
        )
        
        # Extract details from Gemini response
        ocr_details = gemini_result.get("ocr_details", {})
        actual_amount = ocr_details.get("amount")
        
        # Fetch current live base price of the fraction
        try:
            live_price = fetch_live_fraction_price()
        except Exception as e:
            print(f"Error fetching live fraction price: {e}")
            live_price = 100.0  # Fallback base price if API is offline
            
        actual_amount_val = safe_float(actual_amount)
        if actual_amount_val is not None:
            unit_price = live_price * 1.18
            num_fractions = max(1, round(actual_amount_val / unit_price))
            expected_amount = round(unit_price * num_fractions, 2)
            
            # Smart Notch Occlusion Recovery for Camera Cutouts / Dynamic Island:
            # If mismatch > 5.00, test if an obscured digit (e.g., '6' or '0' misread for '8') resolves to expected pricing
            if abs(actual_amount_val - expected_amount) > 5.00:
                s_val = str(int(actual_amount_val))
                for search_digit, replace_digit in [('6', '8'), ('0', '8')]:
                    if search_digit in s_val:
                        candidate_val = float(s_val.replace(search_digit, replace_digit))
                        cand_fractions = max(1, round(candidate_val / unit_price))
                        cand_expected = round(unit_price * cand_fractions, 2)
                        if abs(candidate_val - cand_expected) <= 5.00:
                            actual_amount_val = candidate_val
                            num_fractions = cand_fractions
                            expected_amount = cand_expected
                            ocr_details["amount"] = actual_amount_val
                            break
        else:
            num_fractions = 1
            expected_amount = None

        buyer_id = "API-USER"
        fraction_id = "FRAC-LIVE"
        expected_datetime_str = "Not Specified"
        
        # Determine status & scores
        status = "APPROVED"
        fraud_probability = gemini_result.get("fraud_probability", 0.0)
        gemini_status = gemini_result.get("gemini_status", "VALID")
        gemini_reason = gemini_result.get("gemini_reason", "")
        
        # Check duplicate against purchase_requests table
        is_duplicate, parent_id = db.check_duplicate_against_purchase_requests(ocr_details)
        
        # Python datetime check for 20-minute window
        is_older_ss = False
        receipt_date = ocr_details.get("date")
        receipt_time = ocr_details.get("time")
        if receipt_date:
            try:
                from dateutil import parser
                dt_str = f"{receipt_date} {receipt_time}".strip() if receipt_time else str(receipt_date).strip()
                parsed_dt = parser.parse(dt_str, fuzzy=True)
                if abs((datetime.now() - parsed_dt).total_seconds()) > 1200:
                    is_older_ss = True
            except Exception:
                pass

        if not gemini_result.get("datetime_match", True):
            is_older_ss = True

        # Check if amount difference is within ₹5.00 buffer tolerance
        if expected_amount is not None and actual_amount_val is not None:
            amt_diff = abs(actual_amount_val - expected_amount)
            if amt_diff <= 5.00:
                if gemini_status == "AMOUNT_MISMATCH":
                    gemini_status = "VALID"
                if gemini_reason:
                    cleaned_parts = [
                        part.strip() for part in gemini_reason.split("|")
                        if "amount mismatch" not in part.lower() and "amount match" not in part.lower()
                    ]
                    gemini_reason = " | ".join([p for p in cleaned_parts if p])

        reasons_list = []
        if gemini_reason:
            reasons_list.append(gemini_reason)

        # Final Decision Engine
        if is_duplicate:
            fraud_probability = 100.0
            gemini_status = "DUPLICATE"
            reasons_list.append(f"Duplicate detected: YES (Matches purchase request #{parent_id})")
        else:
            reasons_list.append("Duplicate detected: NO")
        
        if ocr_details.get("is_ai_generated"):
            fraud_probability = max(fraud_probability, 95.0)
            gemini_status = "AI_GENERATED"
            reasons_list.append("AI-Generated Image Detected!")
        
        # Amount mismatch check with ₹5.00 buffer tolerance for live market price fluctuations
        if expected_amount is not None and actual_amount_val is not None and abs(actual_amount_val - expected_amount) > 5.00:
            amount_str = f"₹{actual_amount_val:.2f}"
            if gemini_status not in ["DUPLICATE", "AI_GENERATED"]:
                gemini_status = "AMOUNT_MISMATCH"
            fraud_probability = max(fraud_probability, 85.0)
            reasons_list.append(f"Amount mismatch! Screenshot has {amount_str}, but expected integer multiple of live price + 18% GST (nearest expected amount: ₹{expected_amount:.2f} for {num_fractions} fractions, exceeding ₹5.00 buffer tolerance)")

        if is_older_ss:
            if gemini_status not in ["DUPLICATE", "AI_GENERATED", "AMOUNT_MISMATCH"]:
                gemini_status = "DATETIME_MISMATCH"
            fraud_probability = max(fraud_probability, 75.0)
            reasons_list.append("Screenshot is older than the allowed 20-minute transaction window")

        if ela_warning:
            fraud_probability = max(fraud_probability, 80.0)
            reasons_list.append(f"ELA warning: {ela_warning}")

        if exif_warning:
            fraud_probability = max(fraud_probability, 70.0)
            reasons_list.append(f"Metadata warning: {exif_warning}")

        gemini_reason = " | ".join([r for r in reasons_list if r])

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

        # Save record in SQLite
        scan_data = {
            "filename": temp_filename,
            "buyer_id": buyer_id,
            "fraction_id": fraction_id,
            "num_fractions": num_fractions,
            "expected_amount": expected_amount,
            "actual_amount": ocr_details.get("amount"),
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
        scan_data["id"] = scan_id
        scan_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Remove buyer_id and fraction_id from API output response
        scan_data.pop("buyer_id", None)
        scan_data.pop("fraction_id", None)
        
        return JSONResponse(content=scan_data)
        
    except Exception as e:
        print(f"Error in detection pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        # Clean up the original uploaded file to not save it
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file {temp_path}: {e}")
        # Clean up the ELA image file to not save it
        if ela_image_path and os.path.exists(ela_image_path):
            try:
                os.remove(ela_image_path)
            except Exception as e:
                print(f"Error removing ELA image file {ela_image_path}: {e}")


@app.post("/api/batch-process")
async def batch_process_endpoint(
    background_tasks: BackgroundTasks,
    limit: int = 10,
    offset: int = 0,
    use_gemini: bool = True
):
    """
    Trigger batch processing of CSV screenshot records in the background.
    """
    import batch_process
    
    background_tasks.add_task(
        batch_process.run_batch,
        csv_path="purchase_request (1).csv",
        limit=limit,
        offset=offset,
        use_gemini=use_gemini
    )
    
    return JSONResponse(content={
        "status": "started",
        "message": f"Background batch processing started for records {offset} to {offset + limit - 1}."
    })

def seed_data():
    conn = db.sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scans")
    if cursor.fetchone()[0] == 0:
        # Insert a successful transaction
        cursor.execute("""
            INSERT INTO scans (
                filename, buyer_id, fraction_id, expected_amount, actual_amount,
                image_hash, is_duplicate, duplicate_parent_id, exif_warning,
                gemini_status, gemini_reason, fraud_probability, ocr_details, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "clean_payment.png", "BUYER-8812", "FRAC-ALPHA", 250.00, 250.00,
            "a8e1b4f2c3d4e5f6", 0, None, None,
            "VALID", "Visual check passed, fonts are consistent and amount matches exactly.", 4.2,
            '{"amount": 250.00, "date": "2026-07-16", "time": "11:30 AM", "reference_id": "UTR829104820", "payment_status": "SUCCESS", "is_edited": false}',
            "APPROVED"
        ))
        
        # Insert an edited screenshot transaction
        cursor.execute("""
            INSERT INTO scans (
                filename, buyer_id, fraction_id, expected_amount, actual_amount,
                image_hash, is_duplicate, duplicate_parent_id, exif_warning,
                gemini_status, gemini_reason, fraud_probability, ocr_details, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "tampered_payment.png", "BUYER-1029", "FRAC-GAMMA", 1200.00, 120.00,
            "b7e3a2f8c5d1e2f9", 0, None, "Edited with software: Canva",
            "SUSPECTED_FRAUD", "Amount mismatch and visual anomalies detected. Fonts around amount field are misaligned. Metadata shows editing software: Canva.", 89.5,
            '{"amount": 120.00, "date": "2026-07-15", "time": "10:15 PM", "reference_id": "TXN74810294", "payment_status": "SUCCESS", "is_edited": true}',
            "REJECTED"
        ))
        conn.commit()
    conn.close()

def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    import subprocess
    import sys
    
    try:
        # pyrefly: ignore [missing-import]
        from cryptography import x509
        # pyrefly: ignore [missing-import]
        from cryptography.x509.oid import NameOID
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives import hashes
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives.asymmetric import rsa
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        print("Installing 'cryptography' library for self-signed SSL certificate generation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "cryptography"], check=True)
        # pyrefly: ignore [missing-import]
        from cryptography import x509
        # pyrefly: ignore [missing-import]
        from cryptography.x509.oid import NameOID
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives import hashes
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives.asymmetric import rsa
        # pyrefly: ignore [missing-import]
        from cryptography.hazmat.primitives import serialization

    import datetime

    print("Generating self-signed SSL certificate...")
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write private key to file
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MST"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow() - datetime.timedelta(days=1)
    ).not_valid_after(
        # Valid for 365 days
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.DNSName("127.0.0.1")]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write certificate to file
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print("Self-signed SSL certificate generated successfully.")

seed_data()

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    ssl_keyfile = os.getenv("SSL_KEY_FILE")
    ssl_certfile = os.getenv("SSL_CERT_FILE")
    
    kwargs = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": port,
        "reload": True
    }
    
    if ssl_keyfile and ssl_certfile:
        if not os.path.exists(ssl_keyfile) or not os.path.exists(ssl_certfile):
            try:
                generate_self_signed_cert(ssl_certfile, ssl_keyfile)
            except Exception as e:
                print(f"Failed to generate self-signed SSL cert: {e}")
                print("Starting server in HTTP mode instead.")
                ssl_keyfile = None
                ssl_certfile = None
                
        if ssl_keyfile and ssl_certfile:
            kwargs["ssl_keyfile"] = ssl_keyfile
            kwargs["ssl_certfile"] = ssl_certfile
            print(f"Starting HTTPS server on port {port}...")
        else:
            print(f"Starting HTTP server on port {port}...")
    else:
        print(f"Starting HTTP server on port {port}...")
        
    uvicorn.run("main:app", **kwargs)
