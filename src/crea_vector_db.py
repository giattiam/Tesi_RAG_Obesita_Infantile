"""
crea_vector_db.py
Crea o carica un vector store ChromaDB a partire dai chunks del document_loader.
Espone anche get_retriever() pronto per la pipeline RAG.
"""

import os
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

# ── Configurazione ────────────────────────────────────────────────────────────
CARTELLA_DB   = "./chroma_db"
MODELLO_EMBED = "models/gemini-embedding-001"   # unico stabile sul tuo account
API_VERSION   = "v1beta"                         # unica versione che lista i modelli
RETRIEVER_K   = 5
# ─────────────────────────────────────────────────────────────────────────────


def _get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY non trovata. Controlla il file .env.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version=API_VERSION),
    )


class GeminiEmbeddings(Embeddings):
    """
    Wrapper LangChain-compatibile per gli embedding Gemini.
    Usa models/gemini-embedding-001 tramite endpoint v1beta.
    """

    def __init__(self, model: str = MODELLO_EMBED):
        self.model  = model
        self.client = _get_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedding di una lista di testi (indicizzazione chunks)."""
        risultati = []
        for testo in texts:
            risposta = self.client.models.embed_content(
                model=self.model,
                contents=testo,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            risultati.append(risposta.embeddings[0].values)
        return risultati

    def embed_query(self, text: str) -> List[float]:
        """Embedding di una singola query (retrieval)."""
        risposta = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return risposta.embeddings[0].values


def _get_embeddings() -> GeminiEmbeddings:
    return GeminiEmbeddings(model=MODELLO_EMBED)


def crea_vector_db(chunks: list) -> Chroma:
    if not chunks:
        raise ValueError("La lista di chunks e vuota: non c e nulla da indicizzare.")
    print(f"\n Creazione vector store con {len(chunks)} chunks...")
    print(f"   Modello embedding : {MODELLO_EMBED}")
    print(f"   Destinazione      : {os.path.abspath(CARTELLA_DB)}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CARTELLA_DB,
    )
    print(" Vector store creato e salvato con successo.")
    return vector_store


def carica_vector_db() -> Chroma:
    if not os.path.isdir(CARTELLA_DB):
        raise FileNotFoundError(
            f"Il database {CARTELLA_DB} non esiste. Esegui prima crea_vector_db()."
        )
    print(f"\n Caricamento vector store da: {os.path.abspath(CARTELLA_DB)}")
    vector_store = Chroma(
        persist_directory=CARTELLA_DB,
        embedding_function=_get_embeddings(),
    )
    print(" Vector store caricato e pronto.")
    return vector_store


def get_o_crea_vector_db(chunks: list | None = None) -> Chroma:
    if os.path.isdir(CARTELLA_DB):
        print("Il database esiste gia - caricamento senza ri-indicizzazione.")
        return carica_vector_db()
    if not chunks:
        raise ValueError("Il database non esiste e nessun chunk e stato fornito.")
    return crea_vector_db(chunks)


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    from document_loader import carica_e_taglia_cartella
    cartella_src = os.path.dirname(os.path.abspath(__file__))
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))
    print("=" * 55)
    print("  TEST: creazione/caricamento vector store")
    print("=" * 55)
    if os.path.isdir(CARTELLA_DB):
        print("DB gia presente - caricamento diretto.\n")
        db = carica_vector_db()
    else:
        print("DB non trovato - avvio pipeline completa.\n")
        chunks = carica_e_taglia_cartella(cartella_raw)
        db     = crea_vector_db(chunks)
    retriever  = get_retriever(db, k=3)
    query_test = "Di cosa parla il documento?"
    print(f"\n Query di test: {query_test!r}")
    risultati = retriever.invoke(query_test)
    print(f"   Chunks restituiti: {len(risultati)}")
    for i, r in enumerate(risultati, 1):
        fonte = r.metadata.get("titolo_documento", "?")
        print(f"   [{i}] {fonte} - {r.page_content[:80]}...")
    print("\n Script terminato correttamente.")