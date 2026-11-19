from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# 1. PDF se text nikalo
reader = PdfReader("notes.pdf")
text = ""
for page in reader.pages:
    text += (page.extract_text() or "") + "\n"

# 2. Chunking
def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

# 3. Embeddings (normalize karke, taaki similarity = dot product ho)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, normalize_embeddings=True)

# 4. File mein save karo
np.save("embeddings.npy", embeddings)
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

print("Ingest done!")