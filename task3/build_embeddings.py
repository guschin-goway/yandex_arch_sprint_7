from pathlib import Path
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DOCS_DIR = Path("task3/hp_docs") # Перенес из task2
INDEX_DIR = Path("task3/faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = []

start_time = time.time()

for file_path in DOCS_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    chunks = text_splitter.split_text(text)

    for idx, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_id": idx
                }
            )
        )

print(f"Создано чанков: {len(documents)}")

# Создание FAISS индекса
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# Сохранение индекса
INDEX_DIR.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(INDEX_DIR))

elapsed = time.time() - start_time

print(f"Индекс сохранён в: {INDEX_DIR}")
print(f"Время генерации индекса: {elapsed:.2f} сек")
