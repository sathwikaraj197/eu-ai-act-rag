import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_PATH = "eu_ai_act.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400   # words per chunk
OVERLAP = 50       # word overlap between chunks


def load_and_chunk_pdf(pdf_path: str = PDF_PATH) -> list[str]:
    """Extract text from PDF and split into overlapping word chunks."""
    doc = fitz.open(pdf_path)
    full_text = " ".join(page.get_text() for page in doc)
    doc.close()

    words = full_text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 100:   # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks


def build_index(chunks: list[str]):
    """Embed chunks and build a FAISS flat-L2 index."""
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model, chunks


def search(query: str, index, model, chunks: list[str], top_k: int = 5) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    q_emb = model.encode([query], show_progress_bar=False).astype("float32")
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
