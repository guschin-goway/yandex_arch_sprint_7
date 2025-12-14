import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re


class RussianRAG:
    def __init__(self, faiss_index_path):
        print("🚀 Инициализация RAG системы с русской моделью...")

        # 1. Загружаем эмбеддинги и FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru"  # Русско-английские эмбеддинги
        )

        print("📁 Загружаю векторную базу знаний...")
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Векторная база загружена")

        # 2. Загружаем русскую языковую модель
        print("🧠 Загружаю русскую языковую модель...")

        # Используем маленькую русскую модель от Sberbank
        model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"✅ Модель {model_name} загружена успешно!")

        except Exception as e:
            print(f"❌ Не удалось загрузить {model_name}: {e}")
            print("Пробую альтернативную русскую модель...")

            # Альтернативная модель
            model_name = "ai-forever/rugpt3small_based_on_gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"✅ Загружена альтернативная модель: {model_name}")

        print("⚙️ Настраиваю генератор...")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=80,  # Короткие ответы
            temperature=0.7,  # Средняя температура
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,  # Штраф за повторения
            pad_token_id=self.tokenizer.pad_token_id,
            truncation=True,
            device=-1  # CPU
        )

        print("✅ Система готова к работе!\n")

    def extract_relevant_info(self, question, docs):
        """Извлекает самую релевантную информацию из документов"""
        relevant_texts = []

        for doc in docs:
            content = doc.page_content
            # Ищем упоминания вопроса в документе
            if question.lower() in content.lower():
                # Находим предложение с упоминанием
                sentences = re.split(r'[.!?]', content)
                for sentence in sentences:
                    if question.lower() in sentence.lower():
                        relevant_texts.append(sentence.strip())
                        break

            # Если не нашли прямое упоминание, берем начало документа
            if not relevant_texts:
                relevant_texts.append(content[:200].strip())

        return " ".join(relevant_texts[:3])  # Не более 3 фрагментов

    def create_strict_prompt(self, question, context):
        """Создает строгий промпт для русской модели"""
        return f"""Задание: Ответь на вопрос ТОЛЬКО на основе предоставленной информации.

Информация из документов:
{context}

Вопрос: {question}

Важные правила:
1. Отвечай ТОЛЬКО на основе информации выше
2. Если в информации нет ответа, скажи: "В предоставленных документах нет информации об этом"
3. Не добавляй свою информацию
4. Отвечай кратко и по делу
5. Будь точен

Ответ:"""

    def ask(self, question):
        """Основной метод для вопросов"""
        print(f"\n🔍 Вопрос: '{question}'")

        # 1. Ищем релевантные документы
        print("   Ищу информацию в базе знаний...")
        try:
            docs = self.vector_store.similarity_search(question, k=3)
            print(f"   Найдено документов: {len(docs)}")
        except Exception as e:
            print(f"   ❌ Ошибка поиска: {e}")
            return "Ошибка при поиске информации"

        if not docs:
            return "❌ В базе знаний нет информации"

        # 2. Извлекаем самую релевантную информацию
        context = self.extract_relevant_info(question, docs)

        # 3. Создаем строгий промпт
        prompt = self.create_strict_prompt(question, context)

        # 4. Проверяем длину промпта
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > 900:
            print("   ⚠️  Сокращаю контекст...")
            context = context[:400] + "..."
            prompt = self.create_strict_prompt(question, context)

        print(f"   Длина промпта: {len(tokens)} токенов")

        # 5. Генерируем ответ
        print("   🤖 Генерирую ответ...")
        try:
            result = self.generator(
                prompt,
                return_full_text=False,
                num_return_sequences=1,
                max_length=200  # Ограничиваем общую длину
            )

            if result and len(result) > 0:
                answer = result[0]["generated_text"].strip()

                # Очистка ответа от промпта
                if prompt in answer:
                    answer = answer.replace(prompt, "").strip()

                # Убираем все после первого законченного ответа
                answer = answer.split('\n')[0].strip()

                # Проверяем, что ответ не пустой и не бессмысленный
                if len(answer) < 5 or answer.lower() in ["да", "нет", "не знаю"]:
                    # Возвращаем информацию из контекста
                    return f"Согласно документам: {context[:150]}..."

                print(f"   ✅ Ответ сгенерирован")
                return answer

        except Exception as e:
            print(f"   ❌ Ошибка генерации: {e}")

        # Fallback: возвращаем информацию из контекста
        return f"На основе найденных документов: {context[:200]}..."

    def ask_simple(self, question):
        """Простой метод без генерации - только поиск"""
        print(f"\n🔍 Поиск информации: '{question}'")

        docs = self.vector_store.similarity_search(question, k=2)

        if not docs:
            return "❌ Информация не найдена"

        result = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if len(content) > 300:
                content = content[:297] + "..."

            result.append(f"\n📄 Документ {i}:\n{content}")

            # Показываем метаданные если есть
            if hasattr(doc, 'metadata') and doc.metadata:
                result.append(f"   📍 Источник: {doc.metadata}")

        return "\n".join(result)


def main():
    print("=" * 70)
    print("🤖 RUSSIAN RAG БОТ")
    print("=" * 70)

    # Проверяем наличие FAISS индекса
    import os
    faiss_path = "./task3/faiss_index"

    if not os.path.exists(faiss_path):
        print(f"❌ ОШИБКА: Не найден FAISS индекс по пути: {faiss_path}")
        print("Содержимое папки task3:")
        if os.path.exists("./task3"):
            print(os.listdir("./task3"))
        return

    print(f"📁 Путь к индексу: {faiss_path}")

    try:
        rag = RussianRAG(faiss_path)

        print("\n" + "=" * 70)
        print("💡 ДОСТУПНЫЕ КОМАНДЫ:")
        print("   'просто' - показать найденные документы без генерации")
        print("   'инфо'   - информация о системе")
        print("   'выход'  - завершить работу")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            question = input("❔ Ваш вопрос: ").strip()

            if not question:
                continue

            if question.lower() in ["выход", "exit", "quit", "стоп"]:
                print("\n👋 До свидания!")
                break

            if question.lower() == "инфо":
                print(f"\n📊 ИНФОРМАЦИЯ О СИСТЕМЕ:")
                print(f"   Модель: {rag.model.config._name_or_path}")
                print(f"   Эмбеддинги: cointegrated/LaBSE-en-ru")
                print(f"   Размер контекста: 1024 токенов")
                continue

            if question.lower() == "просто":
                q = input("Вопрос для поиска: ").strip()
                if q:
                    result = rag.ask_simple(q)
                    print(result)
                continue

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
    # Установите нужные пакеты если их нет:
    # pip install transformers[sentencepiece] transformers torch
    main()