import pypdf
import os

def inspect_pdf(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found.")
        return
        
    reader = pypdf.PdfReader(path)
    for i, page in enumerate(reader.pages):
        print(f"\n--- PAGE {i+1} ---")
        print(page.extract_text())

if __name__ == "__main__":
    inspect_pdf("me/linkedin.pdf")
