from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import json
from pathlib import Path
import tempfile
from src.core.pdf_qa_system import PDFQuestionAnsweringSystem

router = APIRouter()

# 1) Create Global QA System
GLOBAL_QA_SYSTEM = PDFQuestionAnsweringSystem()

# -----------------------------
# 2) Upload PDF Endpoint
# -----------------------------
@router.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, process it into chunks, compute embeddings, and store in memory.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use the global QA system to process the PDF
        GLOBAL_QA_SYSTEM.load_document(tmp_path)
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink()

# -----------------------------
# 3) Ask Questions Endpoint
# -----------------------------
class QuestionRequest(BaseModel):
    questions: List[str]

@router.post("/api/ask-questions")
async def ask_questions(payload: QuestionRequest):
    """
    Ask questions after the PDF has already been uploaded and processed.
    """
    if len(GLOBAL_QA_SYSTEM.vector_store.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No PDF data found. Please upload and process a PDF first."
        )

    results = GLOBAL_QA_SYSTEM.answer_questions(payload.questions)
    return json.loads(results)

# -----------------------------
# 4) Combined Endpoint
# -----------------------------
@router.post("/api/upload-and-query")
async def upload_and_query(
    file: UploadFile = File(...),
    questions_json: str = File(...)
):
    # 1) Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # 2) Parse questions
    try:
        questions_data = json.loads(questions_json)
        questions_list = questions_data["questions"]
    except:
        raise HTTPException(400, "Invalid questions format")

    # 3) Save + embed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_content = await file.read()
        tmp.write(pdf_content)
        tmp_path = tmp.name

    try:
        GLOBAL_QA_SYSTEM.load_document(tmp_path)
        results = GLOBAL_QA_SYSTEM.answer_questions(questions_list)
        return json.loads(results)
    finally:
        Path(tmp_path).unlink()
