from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

#Setup LLM
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
print(f"Hugging Face Key: {TOKEN}")

HUGGINGFACE_REPOID = "HuggingFaceH4/zephyr-7b-beta"

def load_llm(hugging_face_repoid, promt):
    #llm = HuggingFaceEndpoint(repo_id=hugging_face_repoid, temperature=0.5,huggingfacehub_api_token = TOKEN, max_new_tokens= 512)
    client = InferenceClient(model=hugging_face_repoid, token=TOKEN)

    llm = client.text_generation(promt)
    return llm

#Connect with FAISS
PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
DB_PATH="vectorstore/db_faiss"

def set_customprompt(prompt_template):
    prompt=PromptTemplate(template=prompt_template,input_variables=["context", "question"])
    return prompt

model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)


# QA CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPOID, PROMPT_TEMPLATE),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True
    #chain_type_kwargs={'prompt': set_customprompt(PROMPT_TEMPLATE)}
)

#Invoke - single query

user_query = input("Write Query Here:")
response=qa_chain.invoke({'query': user_query})
print("Result: ", response["result"])
print("Source Documents: ", response["source_documents"])
