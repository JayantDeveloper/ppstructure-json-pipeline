import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ocr_service import run_ocr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_VALID_OCR_MODES = frozenset({"fast", "light", "safe", "full", "standard", "accurate"})

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    ocr_mode: str = Form("fast"),
):
    if not _is_supported_upload(file):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or common image format.",
        )

    if (ocr_mode or "").strip().lower() not in _VALID_OCR_MODES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported OCR mode. Use 'fast' or 'full'.",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB demo limit.",
        )

    try:
        result = run_ocr(
            contents,
            filename=file.filename,
            content_type=file.content_type,
            mode=ocr_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("OCR processing failed for %s (%s)", file.filename, ocr_mode)
        raise HTTPException(
            status_code=500,
            detail="Could not process file with PP-StructureV3.",
        )
    return JSONResponse(result)


def _is_supported_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    if content_type.startswith("image/") or content_type == "application/pdf":
        return True

    suffix = Path(file.filename or "").suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp"}
