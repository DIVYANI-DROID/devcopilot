# # import requests
# # import os
# # from dotenv import load_dotenv
# # from langchain_community.vectorstores import FAISS
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.chains import RetrievalQA
# # from langchain_core.language_models.llms import LLM
# # from pydantic import Field
# # from typing import Optional
# # from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# # from pathlib import Path
# # from huggingface_hub import InferenceClient

# # # Load API token
# # load_dotenv()

# # # ✅ Custom LLM class using plain requests
# # class HFTextGenLLM(LLM):
# #     model_id: str = Field(...)
# #     temperature: float = Field(default=0.3)
# #     token: Optional[str] = Field(default=None)

# #     def _call(self, prompt: str, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
# #         url = f"https://api-inference.huggingface.co/models/{self.model_id}"
# #         headers = {
# #             "Authorization": f"Bearer {self.token}"
# #         }
# #         payload = {
# #             "inputs": prompt,
# #             "parameters": {"temperature": self.temperature, "max_new_tokens": 256}
# #         }
# #         res = requests.post(url, headers=headers, json=payload)
# #         try:
# #             return res.json()[0]["generated_text"]
# #         except Exception as e:
# #             return f"❌ Error from LLM: {res.text}"

# #     @property
# #     def _llm_type(self) -> str:
# #         return "hf-textgen-custom"

# # # ✅ Main retrieval logic
# # def get_answer(query: str) -> str:
# #     # Load vector DB
# #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     retriever = vector_store.as_retriever()

# #     # Use Flan-T5 which is 100% supported
# #     # llm = HFTextGenLLM(
# #     #     model_id="google/flan-t5-base",
# #     #     temperature=0.3,
# #     #     # hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# #     #     # client = InferenceClient(model=model_id, token=hf_token)
# #     # )
# #     llm = HFTextGenLLM(
# #         model_id = "google/flan-t5-base"
# #         hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# #         client = InferenceClient(model=model_id, token=hf_token)
        
# #         response = client.text_generation(
# #         prompt="Translate English to French: I love Python.",
# #         max_new_tokens=50,
# #         temperature=0.3
# #         )
# #     )

# #     print("✅ Response:", response.generated_text.strip())


# #     print("✅ LLM test:", llm.invoke("Translate English to Hindi: I love coding."))

# #     qa_chain = RetrievalQA.from_chain_type(
# #         llm=llm,
# #         retriever=retriever,
# #         return_source_documents=False
# #     )

# #     result = qa_chain.invoke({"query": query})
# #     return result["result"]

# # print("🔍 .env loaded from:", Path('.env').resolve())
# # print("🔑 Token (from os.getenv):", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# # if __name__ == "__main__":
# #     question = input("Ask a question: ")
# #     print("💬 Answer:", get_answer(question))

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# # from langchain import HuggingFacePipeline
# from langchain_community.llms import HuggingFaceHub
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# def get_answer(query: str) -> str:
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = db.as_retriever()

#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

#     # llm = HuggingFacePipeline(pipeline=pipe)
#     llm = HuggingFaceHub(
#         repo_id="google/flan-t5-base",
#         model_kwargs={
#             "temperature": 0.3,
#             "max_new_tokens": 100
#         }
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=False
#     )

#     return qa_chain.invoke({"query": query})["result"]

# if __name__ == "__main__":
#     question = input("Ask a question: ")
#     print("💬 Answer:", get_answer(question))

"""
Option 1: Run Model Locally
Use HuggingFace pipeline directly with HuggingFacePipeline from langchain_community.
Works offline (no API errors, no rate limits)
HuggingFace’s flan-t5-base is fast and light enough for local usage on Mac
"""
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain.schema import StrOutputParser

# def get_answer(query: str) -> str:
#     # Load vector store
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = db.as_retriever()

#     # Load model locally
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

#     # Wrap in LangChain-compatible LLM
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Build RAG chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=False
#     )

#     return qa_chain.invoke({"query": query})["result"]

# if __name__ == "__main__":
#     question = input("Ask a question: ")
#     print("💬 Answer:", get_answer(question))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ New import
from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# def get_answer(query: str) -> str:
#     # Load FAISS index
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = db.as_retriever()

#     # Load local model
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

#     # Wrap pipeline in LangChain
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Build RAG chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=False
#     )

#     # Ask question
#     # result = qa_chain.invoke({"query": query})["result"]
#     formatted_prompt = f"Answer the question clearly and concisely:\n\n{query}"
#     result = qa_chain.invoke({"query": formatted_prompt})["result"]


#     # ✅ Trim repetitive outputs
#     cleaned = result.strip().split("\n")[0].strip()

#     return cleaned

def get_answer(query: str) -> str:
    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # Load local FLAN-T5
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # Wrap in LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # 🧠 Format prompt
    # formatted_prompt = f"Answer the question clearly and concisely:\n\n{query}"
    # formatted_prompt = f"""Answer the question using the notes below.

    # Notes:
    # To create a Python virtual environment:
    # python3 -m venv venv

    # To activate it on macOS/Linux:
    # source venv/bin/activate

    # Use pip install -r requirements.txt to install dependencies.

    # Question: {query}
    # """

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = f"""You are an assistant answering technical questions from notes.
    Use the notes below to answer the question.

    Notes:
    {context}

    Question: {query}

    Answer:"""


    # Run inference
    # result = qa_chain.invoke({"query": formatted_prompt})["result"]

    # # 🧹 Clean up repetition
    # cleaned = result.strip().split("\n")[0].strip()

    # return cleaned
    # Generate answer
    result = llm.invoke(formatted_prompt)
    return result.strip().split("\n")[0].strip()



if __name__ == "__main__":
    question = input("Ask a question: ")
    print("💬 Answer:", get_answer(question))

