import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("embeddings.npy")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

while True:
    q = input("\nSawal (exit likho band karne ke liye): ")
    if q.lower() == "exit":
        break
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = embeddings @ qv
    for i in np.argsort(scores)[::-1][:3]:
        print(f"\nScore: {scores[i]:.3f}")
        print(chunks[i][:300], "...")