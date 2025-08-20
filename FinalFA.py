import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import time

# ------------------------- FastAPI Setup -------------------------
app = FastAPI(title="RAG API TXT", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    pregunta: str

# ------------------------- Helper Functions -------------------------
def clean_response(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

def split_large_paragraph(paragraph, max_chars=2000):
    chunks = []
    while len(paragraph) > max_chars:
        split_idx = paragraph.rfind('\n', 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars
        chunks.append(paragraph[:split_idx].strip())
        paragraph = paragraph[split_idx:].strip()
    if paragraph:
        chunks.append(paragraph)
    return chunks

# ------------------------- RAG Setup -------------------------
def setup_rag():
    print("Inicializando modelo RAG...")
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Leer documento txt
    txt_path = "trami4.txt"
    with open(txt_path, "r", encoding="utf-8") as file:
        full_text = file.read()

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
2. Si el mensaje es un **saludo**, responde cordialmente.
3. Si el mensaje es una **despedida**, responde de forma cálida.
4. Si es un **cumplido o favor fuera de tema**, agradécelo y recuerda amablemente que eres un asistente enfocado en el contenido del documento.
5. Si no hay información suficiente, di educadamente que no puedes responder.
6. Nunca inventes datos. Usa solo el contenido del documento.

Documento: {documentos}

Mensaje del usuario: {pregunta}
""",
        input_variables=["pregunta", "documentos"],
    )

    llm = ChatOllama(model="gemma2:2b", temperature=0.1, device=device)
    rag_chain = prompt | llm | StrOutputParser()

    print(f"Modelo cargado en {time.time() - start:.2f} segundos")
    return RAGApplication(retriever, rag_chain)

# ------------------------- RAG Wrapper -------------------------
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

# ------------------------- Inicialización -------------------------
rag_application = setup_rag()

@app.get("/")
def health():
    return {"status": "OK", "modelo": "gemma2:2b", "embedding": "nomic-embed-text"}

@app.post("/consulta/")
async def consulta(query: Query):
    if query.pregunta.lower().strip() == "chau":
        return {"respuesta": "Hasta luego!", "tiempo_respuesta": "0.00 segundos"}

    respuesta, tiempo = rag_application.run(query.pregunta)
    return {"respuesta": respuesta, "tiempo_respuesta": f"{tiempo:.2f} segundos"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
