1. # 🚀 DevCopilot – AI Assistant for Developer Notes (RAG Demo)

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
git clone https://github.com/DIVYANI-DROID/devcopilot.git
cd devcopilot

2. Create and Activate a Virtual Environment

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

📌 Usage
Step 1: Add your notes
Place your notes as .txt files in the data/ folder. For example:
data/dev_notes.txt

Step 2: Build the Vector DB

python rag/vectorstore.py

Step 3: Ask Questions
python rag/retriever.py
Example prompt:
Ask a question: How do I activate a Python virtual environment?

📁 Project Structure
devcopilot/
├── data/             # Your developer notes (.txt)
├── faiss_index/      # Saved FAISS vector store
├── rag/
│   ├── vectorstore.py    # Builds the vector index
│   └── retriever.py      # Retrieves answers from the vector DB
├── .env              # Optional (for future API keys)
├── requirements.txt
└── README.md

🔮 Coming Soon (Optional Extensions)
✅ Real Hugging Face LLM (FLAN-T5, Mistral)

💻 Streamlit UI chatbot interface

📦 HuggingFace Spaces deployment

🧪 Auto-updating vector store on file change

🙋‍♀️ Author
Built by Divyani Audichya
If you like this, ⭐️ the repo or share it on LinkedIn!