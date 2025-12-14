import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalLLM:
    def __init__(self, model_name="gpt2", device="cpu"):  # Используем gpt2 как надежную модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Устанавливаем pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загружаем модель на CPU для стабильности
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Простой пайплайн на CPU
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=520,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            device=-1  # Всегда CPU для стабильности
        )

    def generate(self, prompt_text):
        """Генерация текста"""
        try:
            output = self.pipe(prompt_text, return_full_text=False)
            if output and isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], dict):
                    return output[0].get("generated_text", "")
                else:
                    return str(output[0])
            return ""
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return ""


class SimpleRAGBot:
    def __init__(self, faiss_index_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2", llm_model="gpt2"):
        # Эмбеддинги для поиска
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Загрузка FAISS индекса
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Создаем LLM
        self.llm = LocalLLM(llm_model)

        # Промпт
        self.template = """Вопрос: {question}

Контекст из документов:
{context}

На основе предоставленного контекста, дай подробный ответ на вопрос.
Если в контексте нет нужной информации, скажи: "В предоставленных документах нет информации по этому вопросу."

Ответ:"""

        self.prompt_template = PromptTemplate(
            template=self.template,
            input_variables=["question", "context"]
        )

    def ask(self, query, k=30):
        """Основной метод для вопросов"""
        try:
            # 1. Поиск релевантных документов
            docs = self.vector_store.similarity_search(query, k=k)

            # 2. Формирование контекста
            context = "\n---\n".join([
                f"Документ {i + 1}: {doc.page_content}"
                for i, doc in enumerate(docs)
            ])

            # 3. Формирование финального промпта
            final_prompt = self.prompt_template.format(
                question=query,
                context=context
            )

            print(f"\n{'=' * 50}")
            print("Ищу информацию...")
            print(f"Найдено документов: {len(docs)}")
            print(f"{'=' * 50}\n")

            # 4. Генерация ответа
            answer = self.llm.generate(final_prompt)

            # 5. Возврат ответа и источников
            return {
                "answer": answer,
                "sources": docs,
                "context_preview": context[:500] + "..." if len(context) > 500 else context
            }

        except Exception as e:
            print(f"Ошибка в ask: {e}")
            return {
                "answer": f"Произошла ошибка: {str(e)}",
                "sources": [],
                "context_preview": ""
            }


def main():
    faiss_index_path = "./task3/faiss_index"

    print("=" * 60)
    print("🤖 Простой RAG Bot")
    print("=" * 60)

    # Выбор модели
    print("\nДоступные модели:")
    print("1. gpt2 (быстрая, маленькая)")
    print("2. microsoft/DialoGPT-medium (диалоговая)")
    print("3. tiiuae/falcon-7b-instruct (мощная, но может не работать)")

    choice = input("\nВыберите модель (1-3, по умолчанию 1): ").strip()

    model_map = {
        "1": "gpt2",
        "2": "microsoft/DialoGPT-medium",
        "3": "tiiuae/falcon-7b-instruct"
    }

    llm_model = model_map.get(choice, "gpt2")

    print(f"\nЗагружаю модель: {llm_model}")
    print("Использую CPU для стабильности...")

    try:
        bot = SimpleRAGBot(
            faiss_index_path=faiss_index_path,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model=llm_model
        )
        print("✅ Бот успешно инициализирован!")

    except Exception as e:
        print(f"❌ Ошибка при загрузке модели {llm_model}: {e}")
        print("Пробую загрузить gpt2...")
        bot = SimpleRAGBot(
            faiss_index_path=faiss_index_path,
            llm_model="gpt2"
        )
        print("✅ Бот с GPT2 успешно инициализирован!")

    print("\n" + "=" * 60)
    print("Введите 'exit' или 'выход' для завершения")
    print("=" * 60)

    while True:
        try:
            query = input("\n❔ Ваш вопрос: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "выход"]:
                print("Завершение работы...")
                break

            # Получаем ответ
            result = bot.ask(query)

            # Выводим ответ
            print("\n" + "=" * 60)
            print("💬 ОТВЕТ:")
            print(result["answer"])
            print("\n📚 ИСТОЧНИКИ:")

            if result["sources"]:
                for i, doc in enumerate(result["sources"], 1):
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"{i}. {preview}")
            else:
                print("Источники не найдены")

            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()