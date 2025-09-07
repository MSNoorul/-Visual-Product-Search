# backend/embed_and_index.py
import os
import numpy as np
import pandas as pd
import faiss
import sqlite3
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Config
PRODUCTS_CSV = "../data/archieves/amazon_100products.csv"   # test file with 100 rows
PRODUCTS_DIR = "../data/products"      # where we cache images
SQLITE_DB = "products.db"
FAISS_INDEX_FILE = "products.index"
IDS_FILE = "ids.npy"
EMB_DIM = 512   # CLIP ViT-B/32 -> 512 dims

os.makedirs(PRODUCTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Helper: download + embed image ---
def download_image(url, save_path):
    if os.path.exists(save_path):
        return save_path
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception as e:
        print(f"❌ Failed to download {url} -> {e}")
        return None

def embed_image(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)  # shape (1, emb_dim)
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")

# --- Main ---
def main():
    df = pd.read_csv(PRODUCTS_CSV)
    embeddings = []
    ids = []

    print("Embedding images...")
    for _, row in df.iterrows():
        pid = int(row["id"])
        url = row["product_image_url"]

        # local filename (cache)
        filename = os.path.join(PRODUCTS_DIR, f"{pid}.jpg")
        img_path = download_image(url, filename)
        if not img_path:
            continue

        try:
            emb = embed_image(img_path)
        except Exception as e:
            print(f"❌ Failed to embed {pid}: {e}")
            continue

        embeddings.append(emb)
        ids.append(pid)

    if not embeddings:
        raise SystemExit("No embeddings generated. Check CSV and image URLs.")
    emb_matrix = np.vstack(embeddings)  # (N, 512)

    # --- FAISS index ---
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine similarity
    index.add(emb_matrix)
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(IDS_FILE, np.array(ids))
    print("✅ Saved FAISS index and ids:", FAISS_INDEX_FILE, IDS_FILE)

    # --- SQLite metadata ---
    print("Writing metadata to SQLite:", SQLITE_DB)
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            product_title TEXT,
            product_rating REAL,
            total_reviews INTEGER,
            purchased_last_month INTEGER,
            discounted_price REAL,
            original_price REAL,
            is_best_seller TEXT,
            is_sponsored TEXT,
            has_coupon TEXT,
            buy_box_availability TEXT,
            delivery_date TEXT,
            sustainability_tags TEXT,
            product_image_url TEXT,
            product_page_url TEXT,
            data_collected_at TEXT,
            product_category TEXT,
            discount_percentage REAL
        )
    """)

    conn.commit()

    for _, row in df.iterrows():
        cur.execute("""
            INSERT OR REPLACE INTO products (
                id, product_title, product_rating, total_reviews, purchased_last_month,
                discounted_price, original_price, is_best_seller, is_sponsored,
                has_coupon, buy_box_availability, delivery_date, sustainability_tags,
                product_image_url, product_page_url, data_collected_at,
                product_category, discount_percentage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["id"]),
            row["product_title"],
            float(row.get("product_rating", 0)) if not pd.isna(row.get("product_rating")) else None,
            int(row.get("total_reviews", 0)) if not pd.isna(row.get("total_reviews")) else None,
            int(row.get("purchased_last_month", 0)) if not pd.isna(row.get("purchased_last_month")) else None,
            float(row.get("discounted_price", 0)) if not pd.isna(row.get("discounted_price")) else None,
            float(row.get("original_price", 0)) if not pd.isna(row.get("original_price")) else None,
            row.get("is_best_seller"),
            row.get("is_sponsored"),
            row.get("has_coupon"),
            row.get("buy_box_availability"),
            row.get("delivery_date"),
            row.get("sustainability_tags"),
            row["product_image_url"],
            row.get("product_page_url"),
            row.get("data_collected_at"),
            row.get("product_category"),
            float(row.get("discount_percentage", 0)) if not pd.isna(row.get("discount_percentage")) else None
        ))

    conn.commit()
    conn.close()
    print("✅ Done. Metadata + embeddings ready.")

if __name__ == "__main__":
    main()
