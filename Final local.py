import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import time


# Limpiar respuestas
def clean_response(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def split_large_paragraph(paragraph, max_chars=2000):
    chunks = []
    while len(paragraph) > max_chars:
        # Buscar un salto de línea cercano antes del límite
        split_idx = paragraph.rfind('\n', 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars
        chunks.append(paragraph[:split_idx].strip())
        paragraph = paragraph[split_idx:].strip()
    if paragraph:
        chunks.append(paragraph)
    return chunks


# Configuración inicial del modelo RAG
def setup_rag():
    print("Inicializando modelo RAG...")
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Cargar documento .txt
    txt_path = "trami4.txt"
    with open(txt_path, "r", encoding="utf-8") as file:
        full_text = file.read()

    # Dividir por párrafos usando doble salto de línea
    raw_paragraphs = full_text.split("\n\n")

    MAX_CHARS = 2000
    docs = []
    discarded = 0
    for i, p in enumerate(raw_paragraphs):
        p = p.strip()
        if not p:
            continue
        if len(p) <= MAX_CHARS:
            docs.append(Document(page_content=p))
        else:
            split_parts = split_large_paragraph(p, MAX_CHARS)
            if split_parts:
                for part in split_parts:
                    docs.append(Document(page_content=part))
            else:
                discarded += 1
                print(f"❌ Párrafo {i} descartado completamente.")

    print(f"✅ Total fragmentos usados: {len(docs)}")
    print(f"❌ Párrafos descartados completamente: {discarded}")

    if not docs:
        raise ValueError("❌ No se generaron fragmentos válidos para el vectorstore.")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = SKLearnVectorStore.from_documents(documents=docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(k=10)

    prompt = PromptTemplate(
        template="""Eres un asistente virtual inteligente y empático. Responde apropiadamente según el tipo de entrada del usuario:

        1. Si el mensaje es una **pregunta** relacionada con el documento, responde de forma **profesional, clara y útil**, usando viñetas para organizar la información.
        2. Si el mensaje es un **saludo, despedida, peticion o cumplido**, responde de forma **creativa, amable y correspondiente**. Pero no te desvies de tu funcion original.
        3. Si no hay información suficiente en el documento para responder a una pregunta, indica educadamente que no cuentas con esa información.
        4. **Nunca inventes información**. Responde únicamente en base al contenido del documento proporcionado.

        Documento: {documentos}

        Mensaje del usuario: {pregunta}
        """,
        input_variables=["pregunta", "documentos"],
    )

    llm = ChatOllama(model="gemma2:2b", temperature=0.1, device=device)
    rag_chain = prompt | llm | StrOutputParser()

    print(f"Modelo cargado en {time.time() - start:.2f} segundos")
    return RAGApplication(retriever, rag_chain)


# Clase RAG
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        start_time = time.time()
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"pregunta": question, "documentos": doc_texts})
        return clean_response(answer), time.time() - start_time


# Ejecutar si es main
if __name__ == "__main__":
    rag_application = setup_rag()
    print("Sistema RAG listo. Escribe 'chau' para salir.")
    while True:
        pregunta = input("\nTu pregunta: ")
        if pregunta.lower() == "chau":
            print("Hasta luego!")
            break
        respuesta, tiempo = rag_application.run(pregunta)
        print(f"\nRespuesta ({tiempo:.2f}s):\n{respuesta}")
