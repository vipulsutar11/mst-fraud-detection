import sys
import requests
# pyrefly: ignore [missing-import]
import urllib3
import json

# Suppress self-signed certificate warnings for local testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_verify_api(image_path, buyer_id="BUYER-99", fraction_id="FRAC-LIVE", num_fractions=1, expected_datetime="2026-07-16T10:41", paid_amount=None):
    url = "http://127.0.0.1:8001/api/detect"
    
    print("=" * 60)
    print(f"Calling Detect API: {url}")
    print("=" * 60)
    print(f"Payload Params:")
    print(f"  - Image: {image_path}")
    print(f"  - Buyer ID: {buyer_id}")
    print(f"  - Fraction ID: {fraction_id}")
    print(f"  - Number of Fractions: {num_fractions}")
    print(f"  - Expected Datetime: {expected_datetime}")
    print(f"  - Paid Amount: {paid_amount}")
    print("-" * 60)
    
    try:
        with open(image_path, "rb") as f:
            files = {"screenshot": (image_path, f, "image/png")}
            data = {
                "buyer_id": buyer_id,
                "fraction_id": fraction_id,
                "num_fractions": num_fractions,
                "expected_datetime": expected_datetime,
                "fractionsCount": num_fractions
            }
            if paid_amount is not None:
                data["paidAmount"] = paid_amount
            # verify=False is used because local server uses self-signed SSL certificates
            response = requests.post(url, files=files, data=data, verify=False, timeout=60)
            
            print(f"Status Code: {response.status_code}")
            print("Response Body:")
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to API server: {e}")
        print("Please ensure the FastAPI server is running (python main.py)")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image> [buyer_id] [expected_datetime_iso] [paid_amount]")
        print("Example: python test_api.py uploads/old_transaction.png BUYER-99 2026-07-16T10:41 49967.0")
    else:
        img = sys.argv[1]
        buyer = sys.argv[2] if len(sys.argv) > 2 else "BUYER-99"
        dt = sys.argv[3] if len(sys.argv) > 3 else "2026-07-16T10:41"
        amt = float(sys.argv[4]) if len(sys.argv) > 4 else None
        fracs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        test_verify_api(img, buyer_id=buyer, expected_datetime=dt, paid_amount=amt, num_fractions=fracs)
