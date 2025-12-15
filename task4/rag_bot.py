import os
from yandex_cloud_ml_sdk import YCloudML

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class WorkingRAG:
    def __init__(self, faiss_index_path):
        print("🚀 Инициализация RAG (FAISS + YandexGPT ML SDK)")

        # ===============================
        # YANDEX ML SDK
        # ===============================
        self.folder_id = os.getenv("YC_FOLDER_ID")
        self.iam_token = os.getenv("YC_IAM_TOKEN")

        if not self.folder_id or not self.iam_token:
            raise RuntimeError(
                "❌ Задайте YC_FOLDER_ID и YC_IAM_TOKEN через переменные окружения"
            )

        self.ml = YCloudML(
            folder_id=self.folder_id,
            auth_token=self.iam_token
        )

        # latest / lite / pro — можно менять
        self.model = self.ml.text_generation("yandexgpt/latest")

        print("✅ YandexGPT подключен через ML SDK")

        # ===============================
        # EMBEDDINGS + FAISS
        # ===============================
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        print("📁 Загружаю FAISS индекс...")
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS индекс загружен")

        print("✅ RAG готов\n")

    # ===============================
    # PROMPT
    # ===============================
    def build_prompt(self, question, context):
        return f"""
Ты — помощник, который отвечает ТОЛЬКО на основе документов.

Документы:
{context}

Вопрос:
{question}

Правила:
- Если ответа нет в документах, напиши: "В документах нет информации"
- Не выдумывай
- Отвечай кратко и по делу

Ответ:
""".strip()

    # ===============================
    # YANDEX GPT CALL
    # ===============================
    def call_llm(self, prompt):
        result = self.model.run(
            prompt,
            temperature=0.2,
            max_tokens=300
        )
        return result.text.strip()

    # ===============================
    # RAG
    # ===============================
    def ask(self, question):
        print(f"\n🔍 Вопрос: {question}")

        docs = self.vector_store.similarity_search(question, k=3)

        if not docs:
            return "В документах нет информации"

        context = "\n".join(
            f"[Источник {i+1}]: {doc.page_content[:300].strip()}"
            for i, doc in enumerate(docs)
        )

        if not context.strip():
            return "В документах нет информации"

        prompt = self.build_prompt(question, context)

        print("🤖 Запрос к YandexGPT...")
        return self.call_llm(prompt)

def main():
    print("=" * 70)
    print("🤖 RAG БОТ (FAISS + YandexGPT)")
    print("=" * 70)

    if not os.path.exists("./task3/faiss_index"):
        print("❌ Не найден FAISS индекс")
        return

    rag = WorkingRAG("./task3/faiss_index")

    while True:
        question = input("\n❔ Ваш вопрос: ").strip()

        if question.lower() in {"exit", "quit", "выход"}:
            break

        answer = rag.ask(question)

        print("\n💬 ОТВЕТ:")
        print(answer)
