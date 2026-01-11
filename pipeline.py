
# requirements.txt:
# fastapi
# uvicorn
# pdfplumber
# sentence-transformers
# faiss-cpu
# pymupdf
# opencv-python
# pillow
# openai
# numpy
# matplotlib

from fastapi import FastAPI, UploadFile, File
import os, shutil, io, base64, re

import pdfplumber
import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI


app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_PDF = None
all_docs = []
index = None


client = OpenAI()  # المفتاح من ENV فقط


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
                page_text = " ".join(w["text"] for w in words)
                full_text += page_text + "\n"

    full_text = re.sub(r"-\s*\n\s*", "", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
    return full_text


def chunk_text(text, chunk_size=600, overlap=120):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def render_pages(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append({"page": i + 1, "image": pix.tobytes("png")})
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
        if bw * bh > 0.02 * w * h:
            boxes.append((x, y, bw, bh))
    return boxes, img


def extract_text_blocks(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    scale = dpi / 72
    pages_text = []

    for page in doc:
        blocks = page.get_text("blocks")
        page_blocks = []
        for b in blocks:
            x0, y0, x1, y1, text, _, _ = b
            text = text.strip()
            if text:
                page_blocks.append({
                    "bbox": (x0*scale, y0*scale, x1*scale, y1*scale),
                    "text": text
                })
        pages_text.append(page_blocks)
    return pages_text


def visual_bbox(x, y, w, h):
    return (x, y, x+w, y+h)


def find_caption(vb, text_blocks, max_distance=150):
    vx0, vy0, vx1, vy1 = vb
    candidates = []

    for block in text_blocks:
        tx0, ty0, tx1, ty1 = block["bbox"]
        if ty0 > vy1:
            d = ty0 - vy1
            if d < max_distance:
                candidates.append((d, block["text"]))

    if not candidates:
        return None
    return sorted(candidates)[0][1]



def describe_image(image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode()
    prompt = """Describe this scientific figure or table in precise academic language."""
    res = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{
  "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        max_completion_tokens=300
    )
    return res.choices[0].message.content



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global CURRENT_PDF
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    CURRENT_PDF = path
    return {"status": "uploaded", "file": file.filename}



@app.post("/process")
def process_pdf():
    global all_docs, index

    text = extract_text_clean(CURRENT_PDF)
    text_chunks = chunk_text(text)

    pages = render_pages(CURRENT_PDF)
    text_pages = extract_text_blocks(CURRENT_PDF)

    image_docs = []
    for i, page in enumerate(pages):
        boxes, img = detect_visual_blocks(page["image"])
        for (x, y, w, h) in boxes:
            crop = img[y:y+h, x:x+w]
            pil = Image.fromarray(crop)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            desc = describe_image(buf.getvalue())
            image_docs.append(f"Figure (page {i+1}): {desc}")

    all_docs = [{"type": "text", "content": c} for c in text_chunks]
    all_docs += [{"type": "figure", "content": f} for f in image_docs]

    texts = [d["content"] for d in all_docs]
    emb = embed_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    return {"status": "processed", "elements": len(all_docs)}



def get_context(q, k=5):
    q_emb = embed_model.encode([q])
    _, ids = index.search(q_emb, k)
    return "\n\n".join(all_docs[i]["content"] for i in ids[0])


@app.post("/chat")
def chat(question: str):
    context = get_context(question)
    prompt = f"{context}\n\nQuestion: {question}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return {"answer": res.choices[0].message.content}
