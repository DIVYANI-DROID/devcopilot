# 🧠 DevCopilot – AI Assistant for Developer Notes (RAG + HuggingFace + Streamlit)

DevCopilot is a lightweight RAG-based assistant that lets you query your personal `.txt` developer notes using natural language.

✨ Built with:
- **LangChain (Community)**
- **HuggingFace Transformers** – no OpenAI API needed!
- **FAISS** for local semantic search
- **Streamlit** UI (optional)

---

## 💡 Features

- 🧠 Ask natural language questions over your `.txt` notes
- 🪄 Embeds text using `all-MiniLM-L6-v2`
- 🔍 Retrieves matching chunks with FAISS
- 🧩 Chains retrieval with `google/flan-t5-base` (runs locally via Transformers)
- ✅ 100% runs offline, no OpenAI, no cost

---

## 🛠️ Tech Stack

- Python 3.9+
- LangChain (Community)
- HuggingFace Transformers
- FAISS
- Streamlit (optional UI)
- dotenv

---

## 🚀 Getting Started

Clone this repo:

```bash
git clone https://github.com/DIVYANI-DROID/devcopilot.git
cd devcopilot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Place your notes file (e.g. dev_notes.txt) in the data/ folder.

Then:
python rag/vectorstore.py   # ✅ Build FAISS vector DB
python rag/retriever.py     # ✅ Run retrieval + generation locally
streamlit run app.py        # (optional) Launch Streamlit UI

📁 Project Structure

devcopilot/
├── data/                # Text files with your notes
├── faiss_index/         # Vector store saved here
├── rag/
│   ├── vectorstore.py   # Ingest + embed documents
│   └── retriever.py     # RAG pipeline using flan-t5-base
├── app.py               # Streamlit UI
├── requirements.txt
└── README.md

🖼️ Demo Preview
Coming soon: Hosted on HuggingFace Spaces
(Or feel free to clone and run locally!)

🙋‍♀️ Author
👩‍💻 Divyani Audichya
📍 Bengaluru, India
🔗 https://github.com/DIVYANI-DROID  | https://www.linkedin.com/in/divyaniaudichya/

⭐️ Support
If this project inspired you or helped you learn:

⭐️ Star the repo

🗣️ Share it on LinkedIn or X (Twitter)

📬 Or connect to chat about data science and AI!

This is part of my journey toward a Data Scientist / ML Engineer role at top product companies like Atlassian, Google, and Microsoft.


