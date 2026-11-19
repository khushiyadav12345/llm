import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("embeddings.npy")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
llm = anthropic.Anthropic()  # key .env se leta hai

def retrieve(question, k=4):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q                 # har chunk ka similarity score
    top_idx = np.argsort(scores)[::-1][:k]  # sabse zyada score wale k chunks
    return [chunks[i] for i in top_idx]

def ask(question):
    retrieved = retrieve(question)
    context = "\n\n---\n\n".join(retrieved)

    prompt = f"""Neeche diye context ke basis par hi sawal ka jawab do.
Agar jawab context mein nahi hai, to bolo "Mujhe ye notes mein nahi mila."

Context:
{context}

Sawal: {question}"""

    response = llm.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, retrieved

while True:
    q = input("\nSawal (exit likho band karne ke liye): ")
    if q.lower() == "exit":
        break
    answer, sources = ask(q)
    print("\nJawab:", answer)
    print("\n[Top chunk ka pehla hissa]:", sources[0][:200], "...")