import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

THRESHOLD = 0.4   # isse kam score = "nahi mila"
TOP_K = 4

_model = SentenceTransformer("all-MiniLM-L6-v2")
_embeddings = np.load("embeddings.npy")
with open("chunks.json", encoding="utf-8") as f:
    _chunks = json.load(f)


def retrieve(question, k=TOP_K):
    q = _model.encode([question], normalize_embeddings=True)[0]
    scores = _embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [{"text": _chunks[i], "score": round(float(scores[i]), 3)} for i in top]


def build_prompt(question, results):
    context = "\n\n---\n\n".join(r["text"] for r in results)
    return f"""Neeche diye context ke basis par hi sawal ka jawab do.
Agar jawab context mein nahi hai, to bolo "Mujhe ye notes mein nahi mila."
Jawab chhota aur saaf rakho.

Context:
{context}

Sawal: {question}"""


def generate(prompt):
    provider = os.getenv("LLM_PROVIDER", "none").lower()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    if provider == "gemini":
        from google import genai
        client = genai.Client()   # GEMINI_API_KEY .env se leta hai
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return resp.text

    return None   # provider "none" ho to LLM use nahi hoga


def answer(question):
    results = retrieve(question)
    good = [r for r in results if r["score"] >= THRESHOLD]

    if not good:
        return {"answer": "Mujhe ye notes mein nahi mila.", "sources": results}

    text = generate(build_prompt(question, good))
    if text is None:
        # LLM nahi hai to sabse achha chunk hi dikha do
        text = "(LLM off hai) Sabse relevant hissa:\n\n" + good[0]["text"]

    return {"answer": text, "sources": good}