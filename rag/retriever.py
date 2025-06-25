# ✅ Created a vector store using HuggingFaceEmbeddings

# ✅ Built a retriever that answers questions from your own notes

# ✅ Successfully debugged the newest LangChain changes (impressive!)

# ✅ Running locally with no OpenAI dependency
from urllib import response
from langchain_community.llms.fake import FakeListLLM
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# Deprecation Warnings
# LangChain now wants you to install HuggingFace support separately:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def get_answer(query: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    dummy_llm = FakeListLLM(responses=["This is where your answer would go."])

    qa_chain = RetrievalQA.from_chain_type(
        llm=dummy_llm,
        retriever=retriever,
        return_source_documents=True
    )

    # result = qa_chain.run(query)
    # return result
    """LangChain now returns a dictionary with multiple outputs when you use return_source_documents=True:

python
Copy
Edit
{
  "result": "the answer here",
  "source_documents": [...]  # list of matching chunks
}
But qa_chain.run(query) only works if there's exactly one output key — which there isn’t here."""
    # The fix
    response = qa_chain.invoke(query)
    return response["result"]

if __name__ == "__main__":
    question = input("Ask a question: ")
    print("Answer:", get_answer(question))
