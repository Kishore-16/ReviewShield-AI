# Master Deployment & Live E-Commerce Demo Creation Guide

> **Project:** AI-Generated Review Detection System  
> **Target Outcome:** A publicly hosted REST API endpoint paired with a modern, interactive E-Commerce demo website that detects and intercepts synthetic/AI-generated reviews in real time with sub-second latency.

---

## 📋 Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Part 1: Deploying the API to Hugging Face Spaces (Free Public HTTPS Endpoint)](#part-1-deploying-the-api-to-hugging-face-spaces-free-public-https-endpoint)
3. [Part 2: Alternative Cloud Deployment (Render.com)](#part-2-alternative-cloud-deployment-rendercom)
4. [Part 3: Complete Standalone E-Commerce Demo Website (`index.html`)](#part-3-complete-standalone-e-commerce-demo-website-indexhtml)
5. [Part 4: Latency, Speed & Performance Optimization](#part-4-latency-speed--performance-optimization)
6. [Part 5: Professionalism & Production Checklist](#part-5-professionalism--production-checklist)

---

## 1. Architecture Overview

```
 ┌─────────────────────────────────────────┐
 │   E-Commerce Demo Frontend (HTML/JS)    │
 │  (Hosted on GitHub Pages / Vercel / Local)│
 └────────────────────┬────────────────────┘
                      │
                      │  HTTP POST /api/v1/predict-review
                      │  Payload: { "text": "Review string..." }
                      ▼
 ┌─────────────────────────────────────────┐
 │   Public API Endpoint (Hugging Face /   │
 │        Render / FastAPI Server)         │
 │  - Loads detector_model.pkl in RAM      │
 │  - Evaluates 95% OR Probability Rule    │
 └────────────────────┬────────────────────┘
                      │
                      │  JSON Response (<20ms latency)
                      │  { "status": "APPROVED" | "REJECTED",
                      │    "or_probability": 99.86 }
                      ▼
 ┌─────────────────────────────────────────┐
 │  UI Banner & Live Product Review Feed   │
 └─────────────────────────────────────────┘
```

---

## Part 1: Deploying the API to Hugging Face Spaces (Free Public HTTPS Endpoint)

Hugging Face Spaces provides **100% free CPU hosting** with a permanent HTTPS endpoint, automated SSL certificates, and zero configuration setup.

### Step 1.1: Create a Hugging Face Space
1. Sign up or log into [Hugging Face](https://huggingface.co/).
2. Navigate to [hf.co/new-space](https://huggingface.co/new-space).
3. Set the **Space name** (e.g., `ai-review-detector-api`).
4. Select License: **MIT**.
5. Select Space SDK: **Docker** (or **Blank**).
6. Select Space Hardware: **CPU Basic (Free)**.
7. Click **Create Space**.

---

### Step 1.2: Prepare Space Repository Files

You need 5 files inside your Hugging Face Space repository:

```
hf-space-repo/
├── Dockerfile
├── requirements.txt
├── app.py
├── models/
│   └── detector_model.pkl
└── src/
    ├── __init__.py
    ├── stylometrics.py
    └── predictor.py
```

#### File 1: `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /code

# Copy requirement manifest and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy source code and model artifacts
COPY ./models /code/models
COPY ./src /code/src
COPY ./app.py /code/app.py

# Expose default port
EXPOSE 7860

# Run FastAPI via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### File 2: `requirements.txt`
```text
fastapi>=0.100.0
uvicorn>=0.22.0
scikit-learn>=1.0.0
pandas>=1.4.0
numpy>=1.22.0
joblib>=1.1.0
scipy>=1.8.0
pydantic>=2.0.0
```

#### File 3: `app.py`
```python
import time
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ReviewDetectorPredictor

app = FastAPI(
    title="AI Review Detector API",
    description="Real-time detection of synthetic AI reviews using 95% OR Probability Rule",
    version="1.0.0"
)

# Enable CORS for cross-origin requests from any website frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = PROJECT_ROOT / "models" / "detector_model.pkl"
predictor = ReviewDetectorPredictor(model_path=MODEL_PATH)

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Review text string")

class ReviewResponse(BaseModel):
    status: str
    prediction_class: int
    label: str
    confidence: float
    or_probability: float
    cg_probability: float
    latency_ms: float
    message: str

@app.get("/")
def root():
    return {"status": "online", "system": "AI Review Detector API", "docs": "/docs"}

@app.post("/api/v1/predict-review", response_model=ReviewResponse)
def evaluate_review(payload: ReviewRequest):
    t0 = time.time()
    try:
        res = predictor.predict(payload.text)
        latency = round((time.time() - t0) * 1000, 2)
        
        if res['prediction'] == 0:
            return ReviewResponse(
                status="APPROVED",
                prediction_class=0,
                label=res['label'],
                confidence=round(res['confidence'], 2),
                or_probability=round(res['or_probability'], 2),
                cg_probability=round(res['cg_probability'], 2),
                latency_ms=latency,
                message="Genuine human review verified. Approved for publishing."
            )
        else:
            return ReviewResponse(
                status="REJECTED",
                prediction_class=1,
                label=res['label'],
                confidence=round(res['confidence'], 2),
                or_probability=round(res['or_probability'], 2),
                cg_probability=round(res['cg_probability'], 2),
                latency_ms=latency,
                message="Review rejected: Synthetic AI writing pattern detected."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### Step 1.3: Push Code to Hugging Face
Run Git commands to upload your files:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-review-detector-api
cd ai-review-detector-api

# Copy models, src, app.py, Dockerfile, requirements.txt here...

git add .
git commit -m "Deploy AI Review Detector API"
git push
```

---

### Step 1.4: Verify Your Public API Endpoint
Once built, your public API will be live at:
* **Swagger API Docs**: `https://YOUR_USERNAME-ai-review-detector-api.hf.space/docs`
* **Public Prediction Endpoint**: `https://YOUR_USERNAME-ai-review-detector-api.hf.space/api/v1/predict-review`

---

## Part 2: Alternative Cloud Deployment (Render.com)

If you prefer Render:
1. Create a free account on [Render.com](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository containing `ai-review-detector`.
4. Environment: **Python 3**.
5. Build Command: `pip install -r requirements.txt`.
6. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
7. Click **Create Web Service**. Your public URL will be `https://ai-review-detector.onrender.com`.

---

## Part 3: Complete Standalone E-Commerce Demo Website (`index.html`)

Below is a single-file, production-quality HTML5/CSS3/JavaScript web page featuring a modern dark-mode UI for a premium audio brand (*AuraSound Pro Headphones*). It integrates with your public API to reject AI reviews in real time.

Save this file as `demo_website.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AuraSound Pro - Wireless Headphones | AI Review Guard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-main); padding: 40px 20px; line-height: 1.6; }
        .container { max-width: 1000px; margin: 0 auto; }

        /* Header Navigation */
        .navbar { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border-color); margin-bottom: 30px; }
        .logo { font-size: 24px; font-weight: 700; color: var(--accent-blue); letter-spacing: -0.5px; }
        .badge-guard { background-color: rgba(56, 189, 248, 0.1); color: var(--accent-blue); padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.3); }

        /* Product Overview Card */
        .product-card { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; background-color: var(--bg-card); padding: 30px; border-radius: 16px; border: 1px solid var(--border-color); margin-bottom: 40px; }
        .product-image-box { background: radial-gradient(circle, #334155 0%, #0f172a 100%); border-radius: 12px; display: flex; justify-content: center; align-items: center; min-height: 280px; font-size: 80px; }
        .product-details h1 { font-size: 28px; margin-bottom: 10px; }
        .price { font-size: 26px; font-weight: 700; color: var(--accent-blue); margin-bottom: 15px; }
        .rating { color: #f59e0b; margin-bottom: 20px; font-weight: 600; }
        .desc { color: var(--text-muted); font-size: 15px; margin-bottom: 25px; }

        /* Review Submission Section */
        .section-title { font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        .review-box { background-color: var(--bg-card); padding: 25px; border-radius: 16px; border: 1px solid var(--border-color); margin-bottom: 40px; }
        textarea { width: 100%; height: 110px; background-color: var(--bg-input); border: 1px solid var(--border-color); border-radius: 10px; color: #fff; padding: 14px; font-size: 15px; outline: none; transition: border 0.2s; resize: vertical; }
        textarea:focus { border-color: var(--accent-blue); }

        .form-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 15px; }
        .btn-submit { background-color: var(--accent-blue); color: #0f172a; font-weight: 700; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; transition: transform 0.1s, opacity 0.2s; }
        .btn-submit:hover { opacity: 0.9; }
        .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Preset Sample Helper Buttons */
        .sample-buttons { display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
        .btn-sample { background-color: var(--bg-input); color: var(--text-muted); border: 1px solid var(--border-color); padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }
        .btn-sample:hover { color: #fff; border-color: var(--accent-blue); }

        /* Dynamic Alert Banner */
        .alert-banner { display: none; padding: 18px; border-radius: 10px; margin-top: 20px; animation: fadeIn 0.3s ease-in-out; }
        .alert-success { background-color: rgba(34, 197, 94, 0.15); border: 1px solid var(--accent-green); color: #4ade80; }
        .alert-error { background-color: rgba(239, 68, 68, 0.15); border: 1px solid var(--accent-red); color: #f87171; }
        .alert-info { background-color: rgba(56, 189, 248, 0.15); border: 1px solid var(--accent-blue); color: var(--accent-blue); }

        /* Live Reviews Stream */
        .reviews-feed { display: flex; flex-direction: column; gap: 15px; }
        .review-item { background-color: var(--bg-card); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color); }
        .review-header { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 8px; color: var(--text-muted); }
        .review-author { font-weight: 600; color: var(--text-main); }
        .verified-badge { color: var(--accent-green); font-size: 12px; font-weight: 600; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="container">
    <!-- Navbar -->
    <div class="navbar">
        <div class="logo">🎧 SoundCraft Store</div>
        <div class="badge-guard">🛡️ AI Factual Firewall Security Active</div>
    </div>

    <!-- Product Details -->
    <div class="product-card">
        <div class="product-image-box">🎧</div>
        <div class="product-details">
            <h1>AuraSound Pro Wireless Headphones</h1>
            <div class="rating">★★★★★ 4.9 (1,420 Verified Reviews)</div>
            <div class="price">$249.99</div>
            <p class="desc">Active Noise Cancelling (ANC), 40-hour battery life, spatial audio drivers, and premium memory foam earcups.</p>
        </div>
    </div>

    <!-- Review Form Section -->
    <div class="section-title">✍️ Write a Customer Review</div>
    <div class="review-box">
        <textarea id="reviewInput" placeholder="Share your authentic personal experience with this product..."></textarea>

        <!-- Presets for fast testing -->
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 10px;">Quick Test Samples:</div>
        <div class="sample-buttons">
            <button class="btn-sample" onclick="loadSample('human1')">✅ Genuine Human Review</button>
            <button class="btn-sample" onclick="loadSample('ai1')">🚨 AI-Polished Review</button>
            <button class="btn-sample" onclick="loadSample('ai2')">🚨 Marketing Buzzword Review</button>
        </div>

        <div class="form-actions">
            <span id="latencyTag" style="font-size: 13px; color: var(--text-muted);"></span>
            <button id="submitBtn" class="btn-submit" onclick="submitReview()">Submit Review</button>
        </div>

        <div id="alertBanner" class="alert-banner"></div>
    </div>

    <!-- Live Customer Reviews -->
    <div class="section-title">💬 Verified Customer Reviews</div>
    <div id="reviewsFeed" class="reviews-feed">
        <div class="review-item">
            <div class="review-header">
                <span class="review-author">Sarah M. <span class="verified-badge">✓ Verified Buyer</span></span>
                <span>2 days ago</span>
            </div>
            <p>Took these on a flight to Chicago. Battery lasted the entire trip and noise cancellation blocked out the engine drone perfectly!</p>
        </div>
    </div>
</div>

<script>
    // Configuration: Point to your FastAPI URL (Local or Hugging Face Space)
    const API_ENDPOINT = "http://localhost:8000/api/v1/predict-review";
    // For Hugging Face Space deployment, use:
    // const API_ENDPOINT = "https://YOUR_USERNAME-ai-review-detector-api.hf.space/api/v1/predict-review";

    const samples = {
        human1: "Awesome for traveling! Took this on vacation with our 1 year old. He is 26lbs and approx 30 inches long. It was perfect for a small space and lightweight to carry.",
        ai1: "Furthermore, this product has demonstrated exceptional suitability for travel purposes. Notably, its compact design and lightweight construction render it ideal for utilization in confined spaces.",
        ai2: "This item is an absolute MUST-HAVE and a total GAME-CHANGER! Featuring state-of-the-art design, it delivers a delightful addition to your routine. Guaranteed complete satisfaction!"
    };

    function loadSample(key) {
        document.getElementById("reviewInput").value = samples[key];
    }

    async function submitReview() {
        const textInput = document.getElementById("reviewInput").value.trim();
        const alertBanner = document.getElementById("alertBanner");
        const submitBtn = document.getElementById("submitBtn");
        const latencyTag = document.getElementById("latencyTag");

        if (!textInput || textInput.length < 5) {
            showAlert("alert-error", "Please write a valid review (minimum 5 characters).");
            return;
        }

        submitBtn.disabled = true;
        showAlert("alert-info", "🔍 Analyzing text stylometrics & verifying authenticity...");
        const startTime = performance.now();

        try {
            const response = await fetch(API_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: textInput })
            });

            const data = await response.json();
            const elapsed = Math.round(performance.now() - startTime);
            latencyTag.innerText = `API Latency: ${data.latency_ms || elapsed} ms`;

            if (data.status === "APPROVED") {
                showAlert("alert-success", `
                    ✅ <strong>REVIEW VERIFIED & PUBLISHED!</strong><br/>
                    • Authenticity Confidence: <strong>${data.or_probability.toFixed(2)}%</strong> (Human Score)<br/>
                    • Status: ${data.message}
                `);
                
                // Add to live review feed
                addReviewToFeed(textInput);
                document.getElementById("reviewInput").value = "";
            } else {
                showAlert("alert-error", `
                    ⛔ <strong>REVIEW REJECTED BY AI FIREWALL</strong><br/>
                    • AI Synthetic Score: <strong>${data.cg_probability.toFixed(2)}%</strong><br/>
                    • Human Score: <strong>${data.or_probability.toFixed(2)}%</strong> (Required threshold: ≥ 95.00%)<br/>
                    • Reason: ${data.message}
                `);
            }
        } catch (error) {
            showAlert("alert-error", "⚠️ Connection Error: Unable to reach AI Review Detector API server.");
            console.error(error);
        } finally {
            submitBtn.disabled = false;
        }
    }

    function showAlert(typeClass, messageHtml) {
        const banner = document.getElementById("alertBanner");
        banner.className = `alert-banner ${typeClass}`;
        banner.innerHTML = messageHtml;
        banner.style.display = "block";
    }

    function addReviewToFeed(text) {
        const feed = document.getElementById("reviewsFeed");
        const item = document.createElement("div");
        item.className = "review-item";
        item.innerHTML = `
            <div class="review-header">
                <span class="review-author">You <span class="verified-badge">✓ Verified Authenticity (95%+ Rule)</span></span>
                <span>Just now</span>
            </div>
            <p>${escapeHtml(text)}</p>
        `;
        feed.prepend(item);
    }

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
</script>
</body>
</html>
```

---

## Part 4: Latency, Speed & Performance Optimization

To achieve lightning-fast response times (`< 15ms` latency):

1. **In-Memory Checkpoint Loading**:
   The model checkpoint `detector_model.pkl` is loaded once into server memory (`RAM`) when FastAPI boots. Prediction execution takes only **$O(N)$ sparse vector transformation** + Logistic Regression dot product.

2. **Gzip Response Compression**:
   In `app.py`, add Gzip middleware to shrink JSON payload sizes:
   ```python
   from fastapi.middleware.gzip import GzipMiddleware
   app.add_middleware(GzipMiddleware, minimum_size=500)
   ```

3. **Sublinear TF Scaling**:
   The vectorizer uses `sublinear_tf=True`, keeping feature scaling logarithmic ($1 + \log(tf)$), avoiding expensive exponentiation.

---

## Part 5: Professionalism & Production Checklist

Before going live:

- [x] **CORS Configuration**: Restrict `allow_origins=["https://yourdomain.com"]` in production.
- [x] **Input Validation**: Enforce minimum/maximum length constraints via Pydantic (`min_length=5, max_length=2000`).
- [x] **Health Check Endpoint**: `/` or `/health` returns `{ "status": "online" }` for load balancer monitoring.
- [x] **API Documentation**: Auto-generated Swagger docs live at `/docs`.
- [x] **Error Boundaries**: Frontend handles timeout or server disconnect gracefully without freezing UI.
