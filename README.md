# MST Fraction Purchase Verification Engine

An AI-powered transaction screenshot audit and fraud detection engine. This system verifies fractional asset purchase screenshots uploaded by buyers against real-time blockchain pricing and detects tampering, duplicate submissions, and visual discrepancies.

## Features

1. **Live Blockchain Price Sync**: Queries `https://api.mstblockchain.com/purchase/node-fraction-price` in real time to fetch the active fraction price at the time of screenshot upload.
2. **18% GST & Quantity Calculation**: Automatically applies 18% GST to the unit price and validates the total paid amount based on the number of fractions purchased ($(\text{Price} \times \text{Quantity}) \times 1.18$).
3. **Dynamic Buffer Tolerance**: Allows for price/payment fluctuations of up to **₹10.00 per fraction** (calculated dynamically as `10.0 * num_fractions`).
4. **Transaction ID (UTR) Verification**:
   - Checks for duplicate Transaction Reference IDs (UTRs) in the database to prevent double-spending.
   - Flags screenshots immediately if the UTR is missing, invalid, or hidden.
5. **24-Hour Transaction Datetime Window**: Matches screenshot transaction date/time against the current server time in **Indian Standard Time (IST)** with a circular 12-hour clock difference calculation (tolerating AM/PM discrepancies), flagging receipts older than 24 hours.
6. **Strict Image File Validation**: Rejects corrupted, fake, or non-image files (such as PDFs or text files) before processing.
7. **Perceptual Image Hashing**: Uses perceptual hashing (`imagehash.phash`) to detect identical duplicate receipt images in the database.
8. **EXIF Metadata Forensics & AI Warnings**: Scans EXIF tags for software footprints of image editing tools (Canva, Photoshop, Lightroom, GIMP, Snapseed) and generative AI signatures (DALL-E, Midjourney, Stable Diffusion, Firefly, OpenAI, ChatGPT, Flux, Leonardo, Craiyon).
9. **Advanced Visual Audit & Tampering Detection**: Uses OpenAI's Vision API (`gpt-4o-mini` or `gpt-4o`) to perform advanced optical forensic scans:
   - Detects visual anomalies: text misalignments, font mismatches, pixel artifacting.
   - Identifies manual drawings, paint brush strokes, scribbles, and same-color shape cover-ups used to hide UTRs, amounts, or text details.
   - Recognizes synthetic textures, perfect linear gradients, or soft painted artifacts from fake screenshot generators.
10. **Multi-Tier Fallback System**:
    - **Tier 1: Groq VLM Fallback**: If the OpenAI API is offline or returns an error, the system retries the analysis using Groq's Llama 3.2 11B Vision model (`llama-3.2-11b-vision-preview`).
    - **Tier 2: Hugging Face Fallback**: If Groq fails, the request is redirected to the Hugging Face Serverless API using `Qwen/Qwen2.5-VL-7B-Instruct`.
    - **Tier 3: Local Offline Fallback**: If all cloud APIs fail, the engine triggers local OCR (`easyocr`) offline to extract amounts and reference numbers.
    - **Tier 4: Mock Heuristics**: If all automated pipelines fail, it safely flags the scan with simulated parameters for manual review.

---

## Installation & Setup

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (or edit the existing one):
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini   # Use gpt-4o for highest visual accuracy
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_hugging_face_token_here
PORT=8001
HOST=0.0.0.0
```

---

## Running the Application

To start the FastAPI web server, run:
```bash
.venv\Scripts\python main.py
```
Once started, open your browser and navigate to:
[http://localhost:8001](http://localhost:8001)

---

## Running Verification Tests

To test the detection API locally with a sample transaction screenshot:
```bash
.venv\Scripts\python test_api.py <path_to_screenshot> [buyer_id] [expected_datetime_iso]
```
For example:
```bash
.venv\Scripts\python test_api.py uploads/2..jpeg BUYER-99 2026-07-16T10:41
```
