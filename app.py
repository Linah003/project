from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pipeline import (
    extract_text_clean,
    chunk_text,
    render_pages,
    detect_visual_blocks,
    extract_text_blocks,
    find_caption,
    visual_bbox,
    describe_image,
    ask_llm
)

app = FastAPI(title="Research Paper Chatbot Backend")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "file_path": file_path
    }


@app.post("/ask")
def ask_question(pdf_path: str, question: str):
    """
    pdf_path: path of uploaded pdf
    question: user question
    """

  
    answer = ask_llm(question)

    return {
        "question": question,
        "answer": answer
    }
@app.get("/")
def root():
    return {"status": "ok"}
