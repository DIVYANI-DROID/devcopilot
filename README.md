# 🚀 DevCopilot – AI Assistant for Developer Notes (RAG Demo)

DevCopilot is a lightweight AI assistant built with a **Retrieval-Augmented Generation (RAG)** pipeline that lets you query your own developer notes using natural language. Powered by **LangChain**, **HuggingFace**, and **FAISS**, it runs 100% locally — no OpenAI dependency.

---

## 💡 Features

- ✅ Embed your dev notes using HuggingFace transformers
- 🔍 Search relevant chunks with FAISS vector similarity
- 🧠 Get question-answering from your own documents
- 🔐 No OpenAI or API cost — fully offline-friendly
- 🛠️ Modular: plug in real LLMs or wrap in a Streamlit UI

---

## 🛠️ Tech Stack

- **LangChain (Community & HuggingFace)**
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **FAISS** for vector search
- **dotenv** for local environment config
- **Python 3.9+**

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/devcopilot.git
cd devcopilot
