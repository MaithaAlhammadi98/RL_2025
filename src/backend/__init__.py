# разово в отдельной ячейке/скрипте
import chromadb
from pathlib import Path
persist_path = Path(__file__).resolve().parents[2] / "chroma_persistent_storage"
client = chromadb.PersistentClient(path=str(persist_path))
try:
    client.delete_collection("ghg_collection")
    print("Deleted old 'ghg_collection'")
except Exception as e:
    print("Skip:", e)
