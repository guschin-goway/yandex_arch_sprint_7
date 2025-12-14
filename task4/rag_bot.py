import torch
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalLLM:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct", device="cpu"):
        # Проверяем CUDA
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Устанавливаем pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(self.device)

        # Создаем пайплайн
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            device=0 if self.device.type == "cuda" else -1
        )

    def __call__(self, prompt_text):
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


# Класс RAG-бота
class RAGBot:
    def __init__(self, faiss_index_path, embedding_model="all-MiniLM-L6-v2", llm_model="tiiuae/falcon-7b-instruct"):
        # Эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Загрузка FAISS индекса
        self.vector_store = FAISS.load_local(
            faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Локальная LLM
        self.llm = HuggingFacePipeline(pipeline=LocalLLM(llm_model))

        # Промпт
        template = """Ты полезный ассистент. Ответь на вопрос на основе предоставленного контекста.
Если в контексте нет информации для ответа, скажи "Я не нашел информацию по этому вопросу в документах".

Контекст: {context}

Вопрос: {input}

Ответ:"""

        self.prompt = PromptTemplate(template=template, input_variables=["context", "input"])

        # Создаем цепочку для работы с документами
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )

        # Создаем полную RAG цепочку
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )

    def ask(self, query):
        try:
            # Используем invoke вместо прямого вызова метода
            result = self.qa_chain.invoke({"input": query})
            return result.get("answer", "Не удалось получить ответ.")
        except Exception as e:
            print(f"Ошибка в ask: {e}")
            # Альтернативный простой способ
            return self.simple_ask(query)

    def simple_ask(self, query):
        """Простая реализация без сложных цепочек"""
        try:
            # Получаем релевантные документы
            docs = self.retriever.invoke(query)

            # Собираем контекст
            context = "\n".join([doc.page_content for doc in docs])

            # Формируем промпт
            prompt_text = self.prompt.format(context=context, input=query)

            # Генерируем ответ
            answer = self.llm(prompt_text)
            return answer
        except Exception as e:
            print(f"Ошибка в simple_ask: {e}")
            return "Произошла ошибка при обработке запроса."


# Интерактивный интерфейс
def main():
    faiss_index_path = "./task3/faiss_index"

    # Проверяем CUDA
    if torch.cuda.is_available():
        print("✅ Используется GPU")
        device = "cuda"
    else:
        print("⚠️  Используется CPU")
        device = "cpu"

    # Пробуем разные модели, если Falcon не работает
    models_to_try = [
        "tiiuae/falcon-7b-instruct",
        "microsoft/DialoGPT-medium",
        "gpt2"
    ]

    for model_name in models_to_try:
        print(f"\nПопытка загрузить модель: {model_name}")
        try:
            bot = RAGBot(
                faiss_index_path=faiss_index_path,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                llm_model=model_name
            )
            print(f"✅ Модель {model_name} загружена успешно")
            break
        except Exception as e:
            print(f"❌ Ошибка с моделью {model_name}: {e}")
            if model_name == models_to_try[-1]:
                print("\n❌ Все модели не сработали. Проверьте подключение к интернету.")
                return

    print("\n" + "=" * 60)
    print("🤖 RAG Bot запущен. Введите 'exit' или 'выход' для завершения.")
    print("=" * 60)

    while True:
        try:
            query = input("\n❔ Ваш вопрос: ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "выход"]:
                print("Завершение работы...")
                break

            print("🤔 Ищу информацию...")
            answer = bot.ask(query)

            print("\n" + "=" * 60)
            print("💬 Ответ:")
            print(answer)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()