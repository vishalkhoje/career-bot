import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def test_chunking_locally():
    path = "2_rag_bot/me/linkedin.pdf"
    print(f"--- Testing Chunking for {path} ---")
    
    loader = PyPDFLoader(path)
    pages = loader.load()
    
    # Clean and merge pages
    cleaned_pages = []
    for p in pages:
        text = p.page_content
        text = re.sub(r"Page \d+ of \d+", "", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        cleaned_pages.append(text.strip())
    
    full_text = "\n\n".join(cleaned_pages)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=1000,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    doc = Document(page_content=full_text)
    chunks = text_splitter.split_documents([doc])
    
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ---")
        if "HRMS" in chunk.page_content:
            print(">> HRMS FOUND <<")
            # Find index of HRMS
            idx = chunk.page_content.find("HRMS")
            start = max(0, idx - 500)
            end = min(len(chunk.page_content), idx + 500)
            print("CONTEXT AROUND HRMS:")
            print(chunk.page_content[start:end])
            
            if "sumHR" in chunk.page_content:
                print(">> sumHR FOUND IN SAME CHUNK! <<")
            else:
                print(">> sumHR NOT FOUND in this chunk. <<")

if __name__ == "__main__":
    test_chunking_locally()
