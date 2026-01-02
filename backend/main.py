"""
FastAPI main application for Legal Contract Parser
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.document_processor import LegalDocumentProcessor
from app.services.clause_classifier import ClauseClassifier


app = FastAPI(
    title="Legal Contract Parser API",
    description="Advanced legal contract clause extraction system",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
processor = LegalDocumentProcessor()
classifier = ClauseClassifier()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ParseResponse(BaseModel):
    clauses: list
    metadata: dict
    total_pages: int
    processing_time: float


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Legal Contract Parser API",
        "version": "1.0.0"
    }


@app.get("/api/clause-types")
async def get_clause_types():
    """Return list of supported clause types from CUAD"""
    return {
        "categories": classifier.get_all_categories(),
        "total": len(classifier.CUAD_CATEGORIES)
    }


@app.post("/api/parse-contract", response_model=ParseResponse)
async def parse_contract(file: UploadFile = File(...)):
    """
    Upload contract PDF and receive structured clause extraction
    
    Returns:
    {
        "clauses": [
            {
                "number": "1.1",
                "title": "Definitions",
                "content": "...",
                "level": 2,
                "children": [...]
            }
        ],
        "metadata": {...},
        "total_pages": 10,
        "processing_time": 12.3
    }
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process contract
        result = await processor.process_contract(str(file_path))
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = round(processing_time, 2)
        
        # Update metadata
        result['metadata']['processing_date'] = datetime.now().isoformat()
        result['metadata']['filename'] = file.filename
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing contract: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            try:
                os.remove(file_path)
            except:
                pass


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": {
            "tesseract_ocr": processor.use_tesseract,
            "layout_model": processor.layout_model is not None,
            "docformer": processor.docformer_model is not None,
            "contracts_bert": processor.contracts_bert is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

