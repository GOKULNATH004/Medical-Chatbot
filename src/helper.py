from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
import os

def load_pdf_file(data):
    text = ""
    if not os.path.exists(data):
        raise FileNotFoundError(f"Directory '{data}' not found.")
    for filename in os.listdir(data):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(data, filename))
            pages = loader.load_and_split()
            for page in pages:
                text += page.page_content
    return text

def text_split(text):
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')