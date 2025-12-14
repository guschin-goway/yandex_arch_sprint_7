from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json


class SimpleRussianRAG:
    def __init__(self, faiss_index_path):
        print("Загружаю русскую RAG систему...")

        # Русские эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru"
        )

        # Загружаем FAISS
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ Система готова (только поиск, без генерации)")

    def smart_search(self, question):
        """Умный поиск с извлечением релевантных фрагментов"""
        docs = self.vector_store.similarity_search(question, k=3)

        if not docs:
            return "❌ Ничего не найдено"

        results = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content

            # Ищем предложения с ключевыми словами вопроса
            keywords = question.lower().split()
            sentences = content.split('. ')

            relevant_sentences = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords if len(keyword) > 3):
                    relevant_sentences.append(sentence.strip())

            if relevant_sentences:
                excerpt = '. '.join(relevant_sentences[:2]) + '.'
            else:
                excerpt = content[:200] + '...' if len(content) > 200 else content

            results.append(f"\n📄 Результат {i}:\n{excerpt}")

        return "\n".join(results)


# Быстрый запуск
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 РУССКИЙ ПОИСКОВЫЙ RAG (без генерации)")
    print("=" * 60)

    rag = SimpleRussianRAG("./task3/faiss_index")

    print("\nПримеры вопросов:")
    print("- Лира Вейл кто такая?")
    print("- Что такое инфернальный огонь?")
    print("- О чем документы?")
    print("=" * 60)

    while True:
        q = input("\n🔎 Ваш вопрос: ").strip()
        if q.lower() in ["выход", "exit"]:
            break

        result = rag.smart_search(q)
        print(f"\n{result}")