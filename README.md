## 📽️ Demo Video

[![Watch the video](https://img.youtube.com/vi/yKcP_nHxsU0/0.jpg)](https://youtu.be/yKcP_nHxsU0)



# 🖼️ Visual Search with AI

This project is a **Visual Search Application** built using **FastAPI** (backend) and a simple **HTML/CSS/JS frontend**.  
It allows users to **upload an image**, preview it in the UI, and perform AI-powered similarity search to find matching products.

---

## ✨ Features

- 🔍 **AI-Powered Visual Search**  
  Extracts embeddings (CLIP) from product images and stores them in a vector database (FAISS/SQL).  
  Users can upload an image → the system finds **similar products**.

- ⚡ **Backend with FastAPI**  
  REST APIs with auto-generated Swagger UI at `/docs`.

- 🌐 **Frontend**  
  Built with HTML/CSS/JS to:
  - Upload and preview product images  
  - Display product results in a styled UI  
  - Click on a product to view product details page  

---

## 🛠️ Technologies Used

### Backend
- [FastAPI](https://fastapi.tiangolo.com/) → for APIs  
- [Uvicorn](https://www.uvicorn.org/) → ASGI server  
- [Pandas](https://pandas.pydata.org/) → data handling  
- [SentenceTransformers / OpenAI embeddings] → for AI vector embeddings  
- [FAISS](https://faiss.ai/) → similarity search (vector database)  
- [MongoDB Atlas](https://www.mongodb.com/) → product metadata storage  

### Frontend
- HTML5  
- CSS3 (custom styles)  
- Vanilla JavaScript  

### Tools
- Git + GitHub → version control  
- Python venv → environment management  

---

## 🚀 Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
