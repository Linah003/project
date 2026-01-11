from fastapi import FastAPI, UploadFile, File
import os, shutil, io, base64, re

import pdfplumber
import fitz
import cv2
import numpy as np
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI


# ===================== APP =====================
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_PDF = None
all_docs = []
index = None

client = OpenAI()  # API KEY من ENV

# ===================== MODELS (load once) =====================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ===================== TEXT =====================
def extract_text_clean(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False,
                x_tolerance=2,
                y_tolerance=2
            )
            if words:
                full_text += " ".join(w["text"] for w in words) + "\n"

    full_text = re.sub(r"-\s*\n\s*", "", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
    return full_text


def chunk_text(text, chunk_size=600, overlap=120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks


# ===================== VISION =====================
def render_pages(pdf_path, dpi=200):  # ↓ خفّضنا dpi
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append({"page": i+1, "image": pix.tobytes("png")})
    return pages


def detect_visual_blocks(page_img):
    img = cv2.imdecode(np.frombuffer(page_img, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 0.03 * w * h:
            boxes.append((x, y, bw, bh))
    return boxes, img


# ===================== IMAGE DESCRIPTION =====================
def describe_image(image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode()
    prompt = "Describe this scientific figure or table in precise academic language."

    res = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        max_completion_tokens=200
    )
    return res.choices[0].message.content


# ===================== API =====================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global CURRENT_PDF, index, all_docs
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    CURRENT_PDF = path
    index = None
    all_docs = []
    return {"status": "uploaded"}


@app.post("/process")
def process_pdf():
    global all_docs, index

    if not CURRENT_PDF:
        return {"error": "No PDF uploaded"}

    # TEXT ONLY (خفيف)
    text = extract_text_clean(CURRENT_PDF)
    chunks = chunk_text(text)

    all_docs = [{"type": "text", "content": c} for c in chunks]

    embeddings = embed_model.encode(
        [c["content"] for c in all_docs],
        convert_to_numpy=True
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return {"status": "processed", "chunks": len(all_docs)}


def get_context(question, k=5):
    q_emb = embed_model.encode([question])
    _, ids = index.search(q_emb, k)
    return "\n\n".join(all_docs[i]["content"] for i in ids[0])


@app.post("/chat")
def chat(question: str):
    if index is None:
        return {"error": "PDF not processed yet"}

    context = get_context(question)
    prompt = f"{context}\n\nQuestion: {question}"

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return {"answer": res.choices[0].message.content}
