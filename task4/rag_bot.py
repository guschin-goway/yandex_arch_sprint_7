import torch
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalLLM:


    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        if device == "cuda" and torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16  # Используем половинную точность для экономии памяти
        else:
            device_map = "cpu"
            torch_dtype = torch.float32

            # Загружаем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True  # Для некоторых моделей Falcon
        )

        # Устанавливаем pad_token, если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Создаем pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,  # Количество новых токенов для генерации
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,  # Штраф за повторения
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def __call__(self, prompt_text):
        output = self.pipe(prompt_text, return_full_text=False)
        if isinstance(output[0], dict):
            return output[0]["generated_text"]
        return output[0]


# Класс RAG-бота
class RAGBot:
    def __init__(self, faiss_index_path, embedding_model="all-MiniLM-L6-v2", llm_model="TheBloke/WizardLM-7B-uncensored-HF-4bit"):
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

        # Few-shot + CoT промпт
        template = """Система: Ты помощник, который сначала размышляет, а потом отвечает. 
Используй следующие фрагменты контекста, чтобы ответить на вопрос. 
Если не знаешь ответа, скажи, что не знаешь. Отвечай подробно.

Контекст: {context}

Вопрос: {input}

Подумай шаг за шагом и дай подробный ответ:"""

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
        result = self.qa_chain.invoke({"input": query})
        return result["answer"]


# Интерактивный интерфейс
def main():
    faiss_index_path = "./task3/faiss_index"
    bot = RAGBot(faiss_index_path)
    while True:
        query = input("\nВведите вопрос: ")
        if query.lower() in ["exit", "выход"]:
            break
        answer = bot.ask(query)
        print("\nОтвет:\n", answer)


if __name__ == "__main__":
    main()