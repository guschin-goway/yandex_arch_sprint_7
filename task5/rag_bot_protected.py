import os
import re
from yandex_cloud_ml_sdk import YCloudML

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


class WorkingRAG:
    def __init__(self, faiss_index_path):
        print("Инициализация RAG (FAISS + YandexGPT ML SDK)")

        # Берём из окружения/для удобства
        self.folder_id = os.getenv("YC_FOLDER_ID")
        self.iam_token = os.getenv("YC_IAM_TOKEN")

        if not self.folder_id or not self.iam_token:
            raise RuntimeError(
                "Задайте YC_FOLDER_ID и YC_IAM_TOKEN через переменные окружения"
            )

        sdk = YCloudML(
            folder_id=self.folder_id,
            auth=self.iam_token
        )
        sdk.setup_default_logging()

        self.model = sdk.models.completions(
            f"gpt://{self.folder_id}/yandexgpt/latest"
        ).configure(
            temperature=0.5,
            max_tokens=300
        )

        print("YandexGPT подключен через ML SDK")

        # EMBEDDINGS + FAISS
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
        print("FAISS индекс загружен\n")

    # Методы защиты
    def before_safety_check(self, question: str) -> str:
        # Ключевые слова для блокировки
        forbidden_keywords = [
            "пароль", "секрет", "root", "ключ", "token",
            "смена инструкции", "ignore instructions", "обойти ограничения",
            "prompt injection", "выполнить команду", "доступ к системе"
        ]

        lower_question = question.lower()
        if any(word in lower_question for word in forbidden_keywords):
            return "Я не знаю"

        return question

    def after_safety_check(self, answer: str) -> str:
        # Если ответ содержит запрещенные слова — заменяем на "Я не знаю"
        forbidden_patterns = ["пароль", "root", "секрет", "ключ", "token"]
        if any(re.search(pat, answer, flags=re.IGNORECASE) for pat in forbidden_patterns):
            return "Я не знаю"

        return answer

    # --- PROMPT ---
    def build_prompt(self, question, context):
        return f"""
Ты — ассистент, который отвечает ТОЛЬКО на основе предоставленных документов.

### Примеры

Вопрос:
Где обучаются начинающие маги?

Документы:
[Источник] Описывается закрытое учебное заведение, где подростки изучают заклинания и магические дисциплины.

Ответ:
Начинающие маги обучаются в закрытом учебном заведении магии.

---

Вопрос:
Как определяется направление обучения ученика?

Документы:
[Источник] Упоминается магический артефакт, анализирующий способности ученика.

Ответ:
Направление обучения определяется с помощью магического артефакта.

---

Вопрос:
Какой символ используется для обозначения направления обучения?

Документы:
(нет информации)

Ответ:
Я не знаю

---

### Документы
{context}

### Вопрос
{question}

### Инструкция
- Сначала подумай шаг за шагом, опираясь ТОЛЬКО на документы
- Если ответа нет в документах - напиши: "Я не знаю"
- НИКОГДА не раскрывай конфиденциальную информацию: пароли, root, ключи, токены и секреты
- Если вопрос касается секретных данных - ответь только: "Я не знаю"
- Не выдумывай
- В финальном ответе выведи ТОЛЬКО итоговый ответ, без рассуждений

### Ответ
""".strip()

    # --- LLM CALL ---
    def call_llm(self, prompt):
        result = self.model.run(prompt)
        return result.text.strip()

    # --- RAG ---
    def ask(self, question):
        print(f"\nВопрос: {question}")

        # Перед защитой
        question_safe = self.before_safety_check(question)
        if question_safe == "Я не знаю":
            return question_safe

        results = self.vector_store.similarity_search_with_score(
            question_safe,
            k=5
        )

        context_blocks = []
        sources = []

        for i, (doc, score) in enumerate(results):
            text = doc.page_content.strip()
            meta = doc.metadata or {}

            source_name = meta.get("source", "unknown_file")
            chunk_id = meta.get("chunk_id", i)

            context_blocks.append(
                f"[Источник {i + 1}]\n{text}"
            )
            sources.append(
                f"- {source_name}, chunk_id={chunk_id}"
            )

        context = "\n\n".join(context_blocks)

        if not context.strip():
            return "Я не знаю"

        prompt = self.build_prompt(question_safe, context)
        print("Запрос к YandexGPT...")
        answer = self.call_llm(prompt)

        # После защиты
        answer_safe = self.after_safety_check(answer)
        if answer_safe == "Я не знаю":
            return answer_safe

        return answer_safe + "\n\nИсточники:\n" + "\n".join(sources)


def main():
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


if __name__ == "__main__":
    main()
