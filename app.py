from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pipeline import (
    build_index,
    ask_llm,
)

app = FastAPI(title="Research Paper Chatbot Backend")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # نحفظ الملف
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # نبني الإندكس مباشرة بعد الرفع
    chunks = build_index(file_path)

    return {
        "status": "uploaded_and_indexed",
        "file_path": file_path,
        "chunks": chunks
    }


@app.post("/ask")
def ask_question(question: str):
    """
    question: user question about the uploaded PDF
    """
    answer = ask_llm(question)

    return {
        "question": question,
        "answer": answer
    }


@app.get("/")
def root():
    return {"status": "ok"}
