import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class WorkingRAG:
    def __init__(self, faiss_index_path):
        print("🚀 Инициализация RAG системы...")

        # 1. Загружаем эмбеддинги и FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        print("📁 Загружаю векторную базу знаний...")
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Векторная база загружена")

        # 2. Загружаем модель (используем русскоязычную или хорошо обученную)
        print("🧠 Загружаю языковую модель...")

        # Попробуем разные модели в порядке надежности
        models_to_try = [
            "ai-forever/ruGPT-3.5-13B",  # Русскоязычная GPT-2
            # "ai-forever/rugpt3small_based_on_gpt2",  # Еще одна русская
            # "gpt2"  # Английская как запасной вариант
        ]

        self.model = None
        self.tokenizer = None

        for model_name in models_to_try:
            try:
                print(f"  Пробую {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Критически важные настройки для русских моделей
                if "ru" in model_name or "sber" in model_name:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                print(f"  ✅ {model_name} загружена успешно!")
                self.model_name = model_name
                break
            except Exception as e:
                print(f"  ❌ {model_name}: {str(e)[:100]}...")
                continue

        if self.model is None:
            print("❌ Не удалось загрузить ни одну модель!")
            raise Exception("Все модели не сработали")

        # 3. Создаем пайплайн с правильными параметрами
        print("⚙️ Настраиваю генератор...")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=80,  # Короткие ответы
            temperature=0.5,  # Более креативно
            top_p=0.85,  # Контроль разнообразия
            do_sample=False,  # Включить случайность
            repetition_penalty=1.2,  # Штраф за повторения
            pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 50256,
            truncation=True,  # Явно включаем усечение
            device=-1  # CPU для стабильности
        )

        print("✅ Система готова к работе!\n")

    def create_smart_prompt(self, question, context):
        """Создает умный промпт в зависимости от модели"""
        if "ru" in self.model_name or "sber" in self.model_name:
            # Промпт для русскоязычной модели
            return f"""Задание: Ответь на вопрос на основе информации из документов.

Контекстная информация:
{context}

Вопрос: {question}

Ответ:
- Если информация есть в документах, ответь кратко.
- Если информации нет, скажи "В документах нет информации".

Ответ:"""
        else:
            # Промпт для английской модели (более простой)
            return f"""Based on this information: {context}

Question: {question}

Answer in Russian:"""

    def ask(self, question):
        """Основной метод для вопросов"""
        print(f"\n🔍 Вопрос: '{question}'")

        # 1. Ищем релевантные документы (берем немного)
        print("   Ищу информацию в базе знаний...")
        try:
            docs = self.vector_store.similarity_search(question, k=2)
            print(f"   Найдено документов: {len(docs)}")
        except Exception as e:
            print(f"   ❌ Ошибка поиска: {e}")
            return "Ошибка при поиске информации"

        # 2. Формируем КОРОТКИЙ контекст
        context_parts = []
        for i, doc in enumerate(docs):
            # Берем только начало каждого документа
            content = doc.page_content.strip()
            if len(content) > 150:  # Ограничиваем длину
                content = content[:147] + "..."
            context_parts.append(f"[Источник {i + 1}]: {content}")

        context = "\n".join(context_parts)

        # 3. Создаем промпт
        prompt = self.create_smart_prompt(question, context)

        # 4. Проверяем длину промпта
        tokens = self.tokenizer.encode(prompt)
        print(f"   Длина промпта: {len(tokens)} токенов")

        if len(tokens) > 900:  # Слишком длинный
            print("   ⚠️  Слишком длинный промпт, сокращаю...")
            # Берем только первый документ
            if docs:
                content = docs[0].page_content.strip()
                if len(content) > 100:
                    content = content[:97] + "..."
                context = f"[Источник]: {content}"
                prompt = self.create_smart_prompt(question, context)

        # 5. Генерируем ответ
        print("   🤖 Генерирую ответ...")
        try:
            result = self.generator(
                prompt,
                return_full_text=False,
                num_return_sequences=1
            )

            if result and len(result) > 0:
                answer = result[0]["generated_text"].strip()

                # Очистка ответа
                if prompt in answer:
                    answer = answer.replace(prompt, "").strip()

                # Убираем повторения
                lines = answer.split('\n')
                if len(lines) > 1:
                    answer = lines[0].strip()

                print(f"   ✅ Ответ сгенерирован ({len(answer)} символов)")
                return answer
            else:
                return "Не удалось сгенерировать ответ"

        except Exception as e:
            print(f"   ❌ Ошибка генерации: {e}")
            # Возвращаем информацию из контекста как есть
            if context:
                return f"На основе найденной информации: {context[:200]}..."
            return "Не удалось обработать запрос"


def main():
    print("=" * 70)
    print("🤖 РАБОЧИЙ RAG БОТ С РУССКОЯЗЫЧНОЙ МОДЕЛЬЮ")
    print("=" * 70)

    # Проверяем наличие FAISS индекса
    import os
    if not os.path.exists("./task3/faiss_index"):
        print("❌ ОШИБКА: Не найден FAISS индекс!")
        print("Убедитесь, что путь './task3/faiss_index' существует")
        print("И содержит файлы: index.faiss и index.pkl")
        return

    try:
        rag = WorkingRAG("./task3/faiss_index")

        print("\n" + "=" * 70)
        print("💡 СОВЕТ: Задавайте конкретные вопросы по содержимому документов")
        print("   Пример: 'Что такое инфернальный огонь?'")
        print("   Пример: 'Кто главный герой?'")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            question = input("❔ Ваш вопрос: ").strip()

            if not question:
                continue

            if question.lower() in ["выход", "exit", "quit", "стоп"]:
                print("\n👋 До свидания!")
                break

            # Обрабатываем вопрос
            answer = rag.ask(question)

            print("\n" + "=" * 70)
            print("💬 ОТВЕТ:")
            print(answer)
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n👋 Завершение по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()