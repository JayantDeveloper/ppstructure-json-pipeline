# OCR Demo

A FastAPI web service that accepts an image upload and returns structured JSON with extracted text and table data.

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r backend/requirements.txt
```

**3. Install Tesseract**

Tesseract must be installed on the system separately from pip.
Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki

## Running the server

```bash
cd backend
fastapi dev main.py
```

Open your browser and go to `http://localhost:8000`.

## Usage

1. Upload an image using the file picker or drag and drop
2. Click **Run OCR**
3. The extracted text and table data appear as JSON below the image
