from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load PDF

DATA_PATH="data"
def load_pdf(data):
    loader=DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

loadedDocuments = load_pdf(DATA_PATH);
print("Length of pdf pages:", len(loadedDocuments))

#create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

chunks = create_chunks(loadedDocuments)
print("Length of Text Chunks", len(chunks))
#Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

model = get_embedding_model()

#Store in FAISS - store data in local

DB_FAISS_PATH='vectorstore/db_faiss'

db = FAISS.from_documents(chunks, model)
db.save_local(DB_FAISS_PATH)



