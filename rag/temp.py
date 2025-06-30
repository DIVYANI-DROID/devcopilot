# from huggingface_hub import InferenceClient
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = InferenceClient(
#     model="mistralai/Mistral-7B-Instruct-v0.1",
#     token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# response = client.text_generation("What is the capital of France?", max_new_tokens=50)
# print("✅ Response:", response)


from dotenv import load_dotenv
import os
load_dotenv()
print("🔑 Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
