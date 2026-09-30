# Aurelia Studio Pro — AI-Verified E-Commerce Storefront

This folder contains the standalone **Aurelia Studio Pro E-Commerce Web Application** integrated with the **95.00% Threshold AI Review Detector (`detector_model.pkl`)**.

---

## 🚀 Quick Start Guide

### 1. Launch the FastAPI Backend API Server

From the root directory or inside `ecommerce_app/`, run:

```bash
python ecommerce_app/app.py
```

Or using `uvicorn` directly:

```bash
uvicorn ecommerce_app.app:app --reload --port 8000
```

* **Swagger API Documentation**: `http://localhost:8000/docs`
* **Health Check**: `GET http://localhost:8000/`
* **Prediction Endpoint**: `POST http://localhost:8000/api/v1/predict-review`

---

## 2. Launch the E-Commerce Storefront

Open `ecommerce_app/index.html` directly in any web browser:

```bash
# Windows
start ecommerce_app/index.html

# Mac
open ecommerce_app/index.html
```

Or serve it with Python's HTTP server:

```bash
python -m http.server 3000 --directory ecommerce_app
```

Then open `http://localhost:3000` in your browser.

---

## 🛡️ AI Review Guard Integration Features

1. **Real-time REST API Verification**:
   When a user clicks **Verify & Submit Review**, the web page calls `POST http://localhost:8000/api/v1/predict-review` with the review text.

2. **95.00% OR Threshold Policy Enforcement**:
   - **Genuine Human Review ($P(\text{OR}) \ge 95.00\%$)**: Approved, assigned a green `Human-Verified 99.8%` trust badge, and immediately prepended to the live customer review feed!
   - **Synthetic / AI-Generated Review ($P(\text{OR}) < 95.00\%$)**: Intercepted by the AI Firewall, displaying an error banner, threshold gauge, and writing guidance.

3. **Built-in Guard Simulator**:
   Test all 6 system states (Idle, Preset Test, Analyzing Radar Scan, Approved, Rejected, API Error) using the top switcher bar in the review modal!

4. **Curated Quick-Test Samples**:
   - **✅ Human Sample**: *"Awesome for traveling! Took this on vacation with our 1 year old..."*
   - **🚨 AI-Polished Sample**: *"Furthermore, this product has demonstrated exceptional suitability..."*
   - **🚨 Marketing Buzzwords Sample**: *"This item is an absolute MUST-HAVE and a total GAME-CHANGER!..."*
