"""
crea_vector_db.py
Crea o carica un vector store ChromaDB a partire dai chunks del document_loader.
Espone anche get_retriever() pronto per la pipeline RAG.

Nota: usa google-generativeai (SDK classico) per gli embedding invece di
langchain-google-genai, per evitare il bug dell'endpoint v1beta che non
supporta text-embedding-004 nelle versioni recenti del nuovo SDK google-genai.
"""

import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

# ── Configurazione ────────────────────────────────────────────────────────────
CARTELLA_DB   = "./chroma_db"
MODELLO_EMBED = "models/text-embedding-004"
RETRIEVER_K   = 5
# ─────────────────────────────────────────────────────────────────────────────


class GeminiEmbeddings(Embeddings):
    """
    Wrapper LangChain-compatibile per gli embedding di Gemini.
    Usa il vecchio SDK google-generativeai che punta direttamente
    all'endpoint v1 corretto, aggirando il bug v1beta del nuovo SDK.
    """

    def __init__(self, model: str = MODELLO_EMBED):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY non trovata. Controlla il file .env."
            )
        genai.configure(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedding di una lista di testi (usato per indicizzare i chunks)."""
        risultati = []
        for testo in texts:
            risposta = genai.embed_content(
                model=self.model,
                content=testo,
                task_type="retrieval_document",
            )
            risultati.append(risposta["embedding"])
        return risultati

    def embed_query(self, text: str) -> List[float]:
        """Embedding di una singola query (usato durante il retrieval)."""
        risposta = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )
        return risposta["embedding"]


def _get_embeddings() -> GeminiEmbeddings:
    return GeminiEmbeddings(model=MODELLO_EMBED)


def crea_vector_db(chunks: list) -> Chroma:
    """
    Crea un nuovo ChromaDB a partire dalla lista di chunks e lo persiste su disco.

    Args:
        chunks: lista di Document LangChain (output di carica_e_taglia_cartella).

    Returns:
        Istanza Chroma pronta all'uso.

    Raises:
        ValueError: se la lista di chunks è vuota.
    """
    if not chunks:
        raise ValueError("La lista di chunks è vuota: non c'è nulla da indicizzare.")

    print(f"\n🔨 Creazione vector store con {len(chunks)} chunks...")
    print(f"   Modello embedding : {MODELLO_EMBED}")
    print(f"   Destinazione      : {os.path.abspath(CARTELLA_DB)}")

    embeddings   = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CARTELLA_DB,
    )

    print("✅ Vector store creato e salvato con successo.")
    return vector_store


def carica_vector_db() -> Chroma:
    """
    Carica un ChromaDB già esistente dal disco.

    Returns:
        Istanza Chroma pronta all'uso.

    Raises:
        FileNotFoundError: se la cartella del DB non esiste.
    """
    if not os.path.isdir(CARTELLA_DB):
        raise FileNotFoundError(
            f"Il database '{CARTELLA_DB}' non esiste. "
            "Esegui prima crea_vector_db() per crearlo."
        )

    print(f"\n📂 Caricamento vector store da: {os.path.abspath(CARTELLA_DB)}")
    embeddings   = _get_embeddings()
    vector_store = Chroma(
        persist_directory=CARTELLA_DB,
        embedding_function=embeddings,
    )
    print("✅ Vector store caricato e pronto.")
    return vector_store


def get_o_crea_vector_db(chunks: list | None = None) -> Chroma:
    """
    Restituisce il vector store esistente oppure lo crea al volo.

    Se il DB esiste già su disco lo carica (ignorando i chunks in input).
    Se non esiste, usa i chunks forniti per crearlo.
    """
    if os.path.isdir(CARTELLA_DB):
        print("ℹ️  Il database esiste già → caricamento senza ri-indicizzazione.")
        return carica_vector_db()
    else:
        if not chunks:
            raise ValueError(
                "Il database non esiste e nessun chunk è stato fornito per crearlo."
            )
        return crea_vector_db(chunks)


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K):
    """
    Restituisce un retriever LangChain configurato per la similarity search.

    Args:
        vector_store: istanza Chroma.
        k: numero di chunks da restituire per ogni query.

    Returns:
        Retriever compatibile con le LangChain chains (es. RetrievalQA).
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ── Esecuzione diretta (test end-to-end) ──────────────────────────────────────
if __name__ == "__main__":
    from document_loader import carica_e_taglia_cartella

    cartella_src = os.path.dirname(os.path.abspath(__file__))
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))

    print("=" * 55)
    print("  TEST: creazione/caricamento vector store")
    print("=" * 55)

    if os.path.isdir(CARTELLA_DB):
        print("ℹ️  DB già presente → caricamento diretto.\n")
        db = carica_vector_db()
    else:
        print("ℹ️  DB non trovato → avvio pipeline completa.\n")
        chunks = carica_e_taglia_cartella(cartella_raw)
        db     = crea_vector_db(chunks)

    # Test rapido del retriever
    retriever = get_retriever(db, k=3)
    query_test = "Di cosa parla il documento?"
    print(f"\n🔎 Query di test: '{query_test}'")
    risultati = retriever.invoke(query_test)
    print(f"   Chunks restituiti: {len(risultati)}")
    for i, r in enumerate(risultati, 1):
        fonte = r.metadata.get("titolo_documento", "?")
        print(f"   [{i}] {fonte} — {r.page_content[:80]}…")

    print("\n✅ Script terminato correttamente.")