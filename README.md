# 🧠 DevCopilot – AI Assistant for Developer Notes (RAG + Streamlit)

DevCopilot is a lightweight AI assistant that lets you query your own developer notes using natural language.

Built using:
- **LangChain** (Community edition)
- **HuggingFace Embeddings**
- **FAISS vector search**
- **Streamlit UI**

No OpenAI API required.

---

## 💡 Features

- 📄 Query your `.txt` notes locally using RAG
- 🧠 Embed text with HuggingFace transformers (`all-MiniLM-L6-v2`)
- 🔍 Retrieve answers using FAISS
- 🖥️ Ask questions via a clean Streamlit interface
- ✅ Zero OpenAI dependency

---

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit
- LangChain (community)
- HuggingFace Sentence Transformers
- FAISS
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

Add your .txt files (like dev_notes.txt) inside the data/ folder.

Then run:
python rag/vectorstore.py   # Build FAISS vector DB
python rag/retriever.py     # Test RAG logic
streamlit run app.py        # Launch the Streamlit app

📁 Project Structure
devcopilot/
├── data/                # Input .txt files
├── faiss_index/         # Stored vector database
├── rag/
│   ├── vectorstore.py   # Build vector store from notes
│   └── retriever.py     # RAG retriever + answer logic
├── app.py               # Streamlit UI
├── requirements.txt
└── README.md

🤖 Demo Preview
Coming soon: Hosted on Hugging Face Spaces

🙋‍♀️ Author
👩‍💻 Divyani Audichya
📍 Bengaluru, India
🔗 GitHub

⭐️ If you like this project...
Give it a ⭐️ on GitHub, or connect with me on LinkedIn!
This is part of my journey toward a Data Scientist / ML Engineer role at companies like Atlassian, Google, and Microsoft.


---

Once you paste that into `README.md`, save it, then commit & push:

```bash
git add README.md
git commit -m "Update README to reflect Streamlit UI and project structure"
git push
