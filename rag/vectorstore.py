# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import os 
# from dotenv import load_dotenv
# from pathlib import Path

# load_dotenv()

# def build_vector_store(file_path: str, persist_path: str = "faiss_index"):

#     # loader = TextLoader("file_path")
#     file_path = Path(file_path).resolve()
#     loader = TextLoader(str(file_path))
#     documents = loader.load()
    
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     docs = splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(docs, embeddings)
#     vector_store.save_local(persist_path)

#     print(f"✅ Vector store created at: {persist_path}")
#     return persist_path

# if __name__ == "__main__":
#     build_vector_store("data/dev_notes.txt")

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

def build_vector_store(file_path: str, persist_path: str = "faiss_index"):
    loader = TextLoader(Path(file_path).resolve())
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(persist_path)
    print("✅ Vector store saved.")

if __name__ == "__main__":
    build_vector_store("data/dev_notes.txt")
