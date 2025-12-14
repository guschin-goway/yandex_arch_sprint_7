import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class SmartRAG:
    def __init__(self, faiss_index_path, model_name="gpt2"):
        # Загружаем FAISS
        print("Загружаю векторную базу...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Векторная база загружена")

        # Загружаем модель
        print(f"Загружаю модель {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,  # Максимум новых токенов в ответе
            temperature=0.7
        )
        print("✅ Модель загружена")

    def _limit_context_length(self, text, max_tokens=500):
        """Ограничиваем длину текста по токенам"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            # Декодируем обратно в текст
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def ask(self, question):
        print("🔍 Ищу релевантные документы...")

        # Ищем документы (берем меньше для начала)
        docs = self.vector_store.similarity_search(question, k=2)

        # Формируем контекст, но ограничиваем длину каждого документа
        context_parts = []
        for i, doc in enumerate(docs):
            # Берем только первые 300 символов из каждого документа
            doc_text = doc.page_content[:300]
            if len(doc.page_content) > 300:
                doc_text += "..."
            context_parts.append(f"[Документ {i + 1}] {doc_text}")

        context = "\n".join(context_parts)

        print(f"📄 Найдено {len(docs)} документов, общий размер контекста: {len(context)} символов")

        # Формируем короткий промпт
        prompt = f"""Ответь на вопрос на основе информации.

Информация:
{context}

Вопрос: {question}

Ответ:"""

        # Проверяем длину промпта
        prompt_tokens = len(self.tokenizer.encode(prompt))
        print(f"📊 Длина промпта в токенах: {prompt_tokens}")

        if prompt_tokens > 900:  # Оставляем место для ответа
            print("⚠️  Промпт слишком длинный, сокращаю...")
            # Сокращаем контекст еще больше
            context = context[:200] + "..."
            prompt = f"""Ответь на вопрос.

Контекст: {context}

Вопрос: {question}

Ответ:"""

        print("🤖 Генерирую ответ...")
        try:
            result = self.generator(
                prompt,
                return_full_text=False,
                max_length=min(prompt_tokens + 100, 1024)  # Не превышаем лимит модели
            )

            if result and len(result) > 0:
                answer = result[0]["generated_text"].strip()
                # Очищаем ответ от возможных повторений промпта
                if prompt in answer:
                    answer = answer.replace(prompt, "").strip()
                return answer
            else:
                return "Не удалось сгенерировать ответ"

        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            # Пробуем совсем короткий промпт
            short_prompt = f"Вопрос: {question}\nОтвет:"
            try:
                result = self.generator(short_prompt, return_full_text=False, max_new_tokens=50)
                if result:
                    return result[0]["generated_text"].strip()
            except:
                return "Не могу ответить из-за технических ограничений"


def main():
    print("=" * 60)
    print("🤖 УМНЫЙ RAG БОТ (с ограничением длины контекста)")
    print("=" * 60)

    try:
        rag = SmartRAG("./task3/faiss_index", model_name="gpt2")

        print("\n" + "=" * 60)
        print("Готов к работе! Задавайте вопросы.")
        print("Команды: 'выход' - завершить, 'debug' - режим отладки")
        print("=" * 60)

        debug_mode = False

        while True:
            q = input("\n💭 Ваш вопрос: ").strip()

            if not q:
                continue

            if q.lower() in ["выход", "exit", "quit"]:
                print("До свидания!")
                break

            if q.lower() == "debug":
                debug_mode = not debug_mode
                status = "ВКЛЮЧЕН" if debug_mode else "ВЫКЛЮЧЕН"
                print(f"Режим отладки {status}")
                continue

            answer = rag.ask(q)

            if debug_mode:
                print("\n" + "=" * 60)
                print("🔧 РЕЖИМ ОТЛАДКИ")
                print(f"Вопрос: {q}")
                print(f"Ответ: {answer}")
                print("=" * 60)
            else:
                print(f"\n🤖 ОТВЕТ: {answer}")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print("Проверьте путь к FAISS индексу: ./task3/faiss_index")


if __name__ == "__main__":
    main()