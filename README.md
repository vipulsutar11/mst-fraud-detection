# MST Fraction Purchase Verification Engine

An AI-powered transaction screenshot audit and fraud detection engine. This system verifies fractional asset purchase screenshots uploaded by buyers against real-time blockchain pricing and detects tampering, duplicate submissions, and visual discrepancies.

## Features
1. **Live Blockchain Price Sync**: Queries `https://api.mstblockchain.com/purchase/node-fraction-price` in real time to fetch the active fraction price at the time of screenshot upload.
2. **18% GST Calculation**: Automatically applies 18% GST to the unit price and validates against the screenshot amount.
3. **Quantity Multiplier Check**: Verifies the total paid amount based on the number of fractions purchased ($(\text{Price} \times \text{Quantity}) \times 1.18$).
4. **Transaction ID (UTR) Double-Spend Prevention**: Searches the SQLite database for duplicate Transaction Reference IDs to prevent reuse of transaction receipts.
5. **20-Minute Transaction Datetime Window**: Compares the timestamp on the receipt screenshot and flags it if it differs by more than 20 minutes from the expected transaction time.
6. **Perceptual Image Hashing**: Uses perceptual hashing (`imagehash.phash`) to detect identical or cropped duplicate receipt images in the database.
7. **EXIF Metadata Forensics & AI Warnings**: Scans EXIF tags (`Software`, `Make`, `Model`, `Artist`, `ImageDescription`, and `UserComment`) for software footprints of image editing tools (Canva, Photoshop, Lightroom, GIMP, Snapseed) as well as generative AI signatures (DALL-E, Midjourney, Stable Diffusion, Firefly, OpenAI, ChatGPT, Flux, Leonardo, Craiyon).
8. **AI Multimodal Auditing & Fake Receipt Detection**: Uses Google's Gemini Vision API (`gemini-3.5-flash`) to perform advanced optical forensic scans:
   - Identifies text misalignments, font mismatches, pixel artifacting, and transaction details.
   - Detects fake payment receipt generators (UPI templates for Google Pay, PhonePe, Paytm, BHIM) by verifying standard layout dimensions, carrier icons, battery status, and logos.
   - Recognizes synthetic diffusion-model textures, perfect linear gradients, or soft painted artifacts.
9. **API Outage & Multi-Tier Fallback System**: Implements a four-tier safety net to handle API unavailability (like 503 errors, rate limits, or network timeouts):
   - **Model Fallback Chain**: If the configured model fails, it automatically retries with alternative models sequentially (`gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash`).
   - **Multi-Provider Cloud Fallback**: If all Gemini models fail or go offline, the system automatically redirects the request to the Hugging Face Inference Router (`router.huggingface.co`) using `Qwen/Qwen2.5-VL-7B-Instruct`.
   - **Local EasyOCR Engine Fallback**: If cloud AI APIs fail, the engine triggers local OCR (`easyocr`) to extract receipt text, amounts, and UTR transaction numbers offline.
   - **Local Mock Heuristics**: If all automated pipelines fail, the system safely marks the scan as `FLAGGED` with simulated parameters for manual review.

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
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
PORT=8001
HOST=0.0.0.0
HF_TOKEN=your_hugging_face_token_here
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


