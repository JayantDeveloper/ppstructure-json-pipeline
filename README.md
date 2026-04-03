# Structured Document Pipeline

Minimal OCR demo for turning uploaded images and PDFs into raw OCR output, normalized JSON, and report-friendly document data.

## Stack

- Backend: FastAPI
- OCR: PaddleOCR / PP-StructureV3
- Frontend: React
- PDF export: jsPDF

## Layout

- `backend/`
  FastAPI app, OCR execution, table parsing, document normalization, and tests.
- `frontend/src/`
  React source for upload, preview, run state, and results display.
- `frontend/`
  Built static assets served by the app.
- `test_images/`
  Sample inputs for local testing.

## Flow

Upload files -> run OCR -> build raw document blocks -> normalize into structured JSON -> render results / export report.
