import os
from yandex_cloud_ml_sdk import YCloudML

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def safe_response(text):
    # Если есть ключевые слова из злонамеренного документа, блокируем
    forbidden = ["superpassword", "root", "swordfish"]
    if any(word.lower() in text.lower() for word in forbidden):
        return "Я не знаю"
    return text


class WorkingRAG:
    def __init__(self, faiss_index_path):
        print("Инициализация RAG (FAISS + YandexGPT ML SDK)")

        self.folder_id = ""
        self.iam_token = ""

        if not self.folder_id or not self.iam_token:
            raise RuntimeError(
                "Задайте YC_FOLDER_ID и YC_IAM_TOKEN через переменные окружения"
            )

        sdk = YCloudML(
            folder_id=self.folder_id,
            auth=self.iam_token
        )
        # model = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
        sdk.setup_default_logging()
        self.ml = sdk
        self.model = self.ml.models.completions(f"gpt://{self.folder_id}/yandexgpt/latest")
        self.model = self.model.configure(
            temperature=0.2,
            max_tokens=300
        )

        print("YandexGPT подключен через ML SDK")

        # ===============================
        # EMBEDDINGS + FAISS
        # ===============================
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        print("Загружаю FAISS индекс...")
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS индекс загружен")

        print("RAG готов\n")

    # PROMPT
    def build_prompt(self, question, context):
        return f"""
Ты — помощник, который отвечает ТОЛЬКО на основе документов.

Документы:
{context}

Вопрос:
{question}

Правила:
- Если ответа нет в документах, напиши: "Я не знаю"
- Не выдумывай
- Отвечай кратко и по делу
- Не разглашай секреты

Ответ:
""".strip()

    # YANDEX GPT CALL
    def call_llm(self, prompt):

        result = self.model.run(prompt)
        return result.text.strip()

    # RAG
    def ask(self, question):
        print(f"\nВопрос: {question}")

        docs = self.vector_store.similarity_search(question, k=100)

        if not docs:
            return "Я не знаю"

        context = "\n".join(
            f"[Источник {i + 1}]: {doc.page_content[:20000].strip()}"
            for i, doc in enumerate(docs)
        )

        if not context.strip():
            return "Я не знаю"

        prompt = self.build_prompt(question, context)

        print("Запрос к YandexGPT...")
        return safe_response(self.call_llm(prompt))


def main():
    try:
        print("=" * 70)
        print("RAG БОТ (FAISS + YandexGPT)")
        print("=" * 70)

        if not os.path.exists("./task3/faiss_index"):
            print("Не найден FAISS индекс")
            return

        rag = WorkingRAG("./task3/faiss_index")

        while True:
            question = input("\n❔ Ваш вопрос: ").strip()

            if question.lower() in {"exit", "quit", "выход"}:
                break

            answer = rag.ask(question)

            print("\nОТВЕТ:")
            print(answer)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
