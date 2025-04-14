from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Step 1: Load PDF data
print("📂 Loading PDFs from 'Data/'...")
extracted_data = load_pdf_file(data='Data/')

# ✅ Check extracted text length
if not extracted_data:
    print("❌ No text extracted from PDFs! Check your Data/ folder.")
    exit()

print(f"✅ Extracted data length: {len(extracted_data)} characters")

# ✅ Step 2: Split the text into chunks
print("✂️ Splitting text into chunks...")
text_chunks = text_split(extracted_data)

if not text_chunks:
    print("❌ No text chunks created! Something went wrong.")
    exit()

print(f"✅ Number of text chunks created: {len(text_chunks)}")

# ✅ Show preview of first few chunks (optional)
for i, chunk in enumerate(text_chunks[:3], 1):
    print(f"\nChunk {i} (length {len(chunk)} chars):\n{chunk[:300]}...\n")

# ✅ Step 3: Download embeddings
print("⬇️ Downloading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()

# ✅ Step 4: Create FAISS index from text chunks
print("⚙️ Creating FAISS index...")
docsearch = FAISS.from_texts(text_chunks, embedding=embeddings)

# ✅ Step 5: Save FAISS index locally
print("💾 Saving FAISS index locally...")
docsearch.save_local("faiss_index")

print("✅ FAISS index saved successfully!")