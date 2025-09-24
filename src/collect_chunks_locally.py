import chromadb
import pandas as pd
from pathlib import Path 
from sentence_transformers import SentenceTransformer
import uuid
from PyPDF2 import PdfReader

project_root = Path(__file__).resolve().parents[1]
persist_path = project_root / "chroma_persistent_storage"
data_dir = project_root / "src" / "data"
persist_path.mkdir(parents=True, exist_ok=True)
# Connect to your ChromaDB host
client = chromadb.PersistentClient(path=str(persist_path))





# Load your collection
collection = client.get_or_create_collection(name="ghg_collection")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
# Fetch all documents (adjust limit if needed)
results = collection.get(include=["documents", "metadatas", "embeddings"], limit=10)

def read_pdf(path: Path):
    texts = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # пробуем пустой пароль
            except:
                print(f"⚠️ Не удалось расшифровать {path.name}")
                return []
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = page.extract_text() or ""
                t = " ".join(t.split())
                if t:
                    texts.append((i, t))
            except Exception as e:
                print(f"⚠️ Ошибка чтения страницы {i} в {path.name}: {e}")
    return texts


def chunk_text(t: str, size=1000, overlap=200):
    res = []
    i = 0
    while i < len(t):
        res.append(t[i:i+size])
        i += max(1, size - overlap)
    return res

docs, ids, metas = [], [], []

for pdf in sorted(data_dir.glob("*.pdf")):
    pages = read_pdf(pdf)
    for page_num, page_text in pages:
        for ch_i, chunk in enumerate(chunk_text(page_text)):
            docs.append(chunk)
            ids.append(str(uuid.uuid4()))
            metas.append({
                "source": pdf.name,
                "page": page_num,
                "chunk": ch_i
            })

if docs:
    embs = model.encode(docs, batch_size=64, convert_to_numpy=True).tolist()

    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_embs = embs[i:i+batch_size]
        batch_ids  = ids[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"✅ add {len(batch_docs)} chunk (total {i+len(batch_docs)}/{len(docs)})")
else:
    print("⚠️ Err.")

collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
# Convert to DataFrame
df = pd.DataFrame(
    {
        "document": results["documents"],
        "metadata": results["metadatas"],
        "embedding": results["embeddings"],
    }
)

# Save to CSV
df.to_csv("ghg_chunks_export.csv", index=False)

print("✅ Exported to ghg_chunks_export.csv")
