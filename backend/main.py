# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, numpy as np, faiss, sqlite3, os, torch
from transformers import CLIPModel, CLIPProcessor

# Config
FAISS_INDEX = "products.index"
IDS_FILE = "ids.npy"
SQLITE_DB = "products.db"
TOP_K = 10

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS index + id mapping
index = faiss.read_index(FAISS_INDEX)
ids = np.load(IDS_FILE)


def embed_pil_image(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")


def fetch_metadata(product_id):
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, product_title, product_rating, discounted_price,
               original_price, product_image_url, product_page_url, product_category
        FROM products WHERE id=?
    """, (int(product_id),))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "title": row[1],
            "rating": row[2],
            "discounted_price": row[3],
            "original_price": row[4],
            "image_url": row[5],
            "page_url": row[6],
            "category": row[7],
        }
    return None


@app.post("/search")
async def search(file: UploadFile = File(...)):
    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert("RGB")
    q = embed_pil_image(pil)  # shape (1, dim)

    D, I = index.search(q, TOP_K)  # cosine similarity scores + indices
    results = []
    for score, idx in zip(D[0], I[0]):
        pid = int(ids[int(idx)])   # map FAISS index -> product id
        meta = fetch_metadata(pid)
        if meta:
            meta["score"] = float(score)
            results.append(meta)

    return {"results": results}
