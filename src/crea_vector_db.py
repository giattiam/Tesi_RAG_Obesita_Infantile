"""
crea_vector_db.py

Crea o carica un vector store ChromaDB e un indice BM25 parallelo.
Espone un retriever ibrido che combina similarity search vettoriale e BM25
keyword-based per migliorare la precisione su termini tecnici specifici.
"""

import os
import pickle
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------

CARTELLA_DB    = "./chroma_db"
PERCORSO_BM25  = "./chroma_db/bm25_index.pkl"
MODELLO_EMBED  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RETRIEVER_K       = 5
PESO_VETTORIALE   = 0.6    # peso della similarity search nel punteggio ibrido
PESO_BM25         = 0.4    # peso del BM25 nel punteggio ibrido
SOGLIA_PERTINENZA = 0.25   # chunk con punteggio ibrido inferiore vengono scartati
RERANKER_K        = 15     # candidati recuperati prima del reranking


# ---------------------------------------------------------------------------
# Client e modello di embedding
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Istanzia il modello di embedding HuggingFace in locale.
    Il modello paraphrase-multilingual-mpnet-base-v2 supporta l'italiano
    e non richiede API key ne connessione internet dopo il primo download.
    """
    return HuggingFaceEmbeddings(
        model_name=MODELLO_EMBED,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Indice BM25
# ---------------------------------------------------------------------------

def _tokenizza(testo: str) -> List[str]:
    """Tokenizzazione minima: lowercase e split su spazi/punteggiatura."""
    return re.sub(r"[^\w\s]", " ", testo.lower()).split()


def _costruisci_bm25(chunks: list) -> BM25Okapi:
    """Costruisce l'indice BM25 a partire dalla lista di chunks."""
    corpus = [_tokenizza(doc.page_content) for doc in chunks]
    return BM25Okapi(corpus)


def _salva_bm25(indice: BM25Okapi, chunks: list) -> None:
    with open(PERCORSO_BM25, "wb") as f:
        pickle.dump({"indice": indice, "chunks": chunks}, f)


def _carica_bm25() -> tuple:
    """
    Carica l'indice BM25 e i chunks associati dal disco.
    Returns:
        Tuple (BM25Okapi, list[Document])
    """
    with open(PERCORSO_BM25, "rb") as f:
        dati = pickle.load(f)
    return dati["indice"], dati["chunks"]


# ---------------------------------------------------------------------------
# Vector store ChromaDB
# ---------------------------------------------------------------------------

def crea_vector_db(chunks: list) -> Chroma:
    """
    Crea ChromaDB e indice BM25 a partire dalla lista di chunks.
    Entrambi vengono persistiti su disco.

    Raises:
        ValueError: se la lista di chunks e vuota.
    """
    if not chunks:
        raise ValueError("La lista di chunks e vuota.")

    print(f"\nCreazione vector store con {len(chunks)} chunks...")
    print(f"   Modello embedding : {MODELLO_EMBED}")
    print(f"   Destinazione      : {os.path.abspath(CARTELLA_DB)}")

    os.makedirs(CARTELLA_DB, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CARTELLA_DB,
    )

    print("   Costruzione indice BM25...")
    indice_bm25 = _costruisci_bm25(chunks)
    _salva_bm25(indice_bm25, chunks)

    print("Vector store e indice BM25 creati.")
    return vector_store


def carica_vector_db() -> Chroma:
    """
    Carica ChromaDB dal disco.

    Raises:
        FileNotFoundError: se la cartella del DB non esiste.
    """
    if not os.path.isdir(CARTELLA_DB):
        raise FileNotFoundError(
            f"Il database '{CARTELLA_DB}' non esiste. Esegui prima crea_vector_db()."
        )
    print(f"\nCaricamento vector store da: {os.path.abspath(CARTELLA_DB)}")
    vector_store = Chroma(
        persist_directory=CARTELLA_DB,
        embedding_function=_get_embeddings(),
    )
    print("Vector store caricato.")
    return vector_store


def get_o_crea_vector_db(chunks: list | None = None) -> Chroma:
    if os.path.isdir(CARTELLA_DB):
        print("Il database esiste gia - caricamento senza ri-indicizzazione.")
        return carica_vector_db()
    if not chunks:
        raise ValueError("Il database non esiste e nessun chunk e stato fornito.")
    return crea_vector_db(chunks)


# ---------------------------------------------------------------------------
# Retriever ibrido
# ---------------------------------------------------------------------------

class RetrieverIbrido:
    """
    Combina similarity search vettoriale (ChromaDB) e ricerca keyword (BM25).

    Per ogni query:
    1. Recupera i top-k*2 candidati da ciascuna sorgente
    2. Normalizza i punteggi di entrambe in [0, 1]
    3. Calcola il punteggio ibrido: peso_vettoriale * score_vett + peso_bm25 * score_bm25
    4. Restituisce i top-k per punteggio ibrido, deduplicati

    Il parametro breakpoint_threshold nel SemanticChunker e i pesi qui
    sono i principali parametri da ottimizzare per la valutazione RAGAS.
    """

    def __init__(self, vector_store: Chroma, k: int = RETRIEVER_K):
        self.vector_store = vector_store
        self.k            = k
        self.embeddings   = _get_embeddings()

        if os.path.exists(PERCORSO_BM25):
            self.indice_bm25, self.chunks_bm25 = _carica_bm25()
        else:
            print("Avviso: indice BM25 non trovato. Solo similarity search attiva.")
            self.indice_bm25  = None
            self.chunks_bm25  = []

    def _normalizza(self, punteggi: list) -> list:
        if not punteggi:
            return []
        minp, maxp = min(punteggi), max(punteggi)
        if maxp == minp:
            return [1.0] * len(punteggi)
        return [(p - minp) / (maxp - minp) for p in punteggi]

    def invoke(self, query: str) -> List[Document]:
        candidati = {}   # doc_id -> {"doc": Document, "score": float}
        n_candidati = self.k * 2

        # --- Similarity search vettoriale ---
        risultati_vett = self.vector_store.similarity_search_with_score(
            query, k=n_candidati
        )
        punteggi_vett = [score for _, score in risultati_vett]
        # Chroma restituisce distanze (minore = migliore): invertiamo
        punteggi_vett_inv = [1 - s for s in punteggi_vett]
        norm_vett = self._normalizza(punteggi_vett_inv)

        for (doc, _), score in zip(risultati_vett, norm_vett):
            doc_id = doc.page_content[:80]
            candidati[doc_id] = {
                "doc":        doc,
                "score_vett": score,
                "score_bm25": 0.0,
            }

        # --- BM25 ---
        if self.indice_bm25 is not None:
            token_query  = _tokenizza(query)
            punteggi_bm25 = self.indice_bm25.get_scores(token_query)
            top_idx       = sorted(
                range(len(punteggi_bm25)),
                key=lambda i: punteggi_bm25[i],
                reverse=True,
            )[:n_candidati]
            top_scores    = [punteggi_bm25[i] for i in top_idx]
            norm_bm25     = self._normalizza(top_scores)

            for idx, score in zip(top_idx, norm_bm25):
                doc    = self.chunks_bm25[idx]
                doc_id = doc.page_content[:80]
                if doc_id in candidati:
                    candidati[doc_id]["score_bm25"] = score
                else:
                    candidati[doc_id] = {
                        "doc":        doc,
                        "score_vett": 0.0,
                        "score_bm25": score,
                    }

        # --- Punteggio ibrido e ranking finale ---
        for doc_id in candidati:
            candidati[doc_id]["score_ibrido"] = (
                    PESO_VETTORIALE * candidati[doc_id]["score_vett"]
                    + PESO_BM25     * candidati[doc_id]["score_bm25"]
            )

        ordinati = sorted(
            candidati.values(),
            key=lambda x: x["score_ibrido"],
            reverse=True,
        )

        # Filtro soglia pertinenza
        filtrati = [e for e in ordinati if e["score_ibrido"] >= SOGLIA_PERTINENZA]
        if not filtrati:
            filtrati = ordinati  # fallback senza filtro

        return [entry["doc"] for entry in filtrati[: self.k]]


# ---------------------------------------------------------------------------
# Prompt per il reranking dei chunk
# ---------------------------------------------------------------------------

PROMPT_RERANKING = """Sei un sistema di information retrieval per documenti scientifici
italiani su nutrizione pediatrica e obesita' infantile.

Valuta la pertinenza del seguente chunk rispetto alla query dell'utente.
Rispondi SOLO con un numero intero da 0 a 10 (0 = irrilevante, 10 = perfettamente pertinente).
Nessuna spiegazione, solo il numero.

Query: {query}

Chunk:
{chunk}

Pertinenza (0-10):"""


def rerank_chunks(chunks: list, query: str, llm, k: int = RETRIEVER_K) -> list:
    """
    Riordina i chunk per pertinenza usando il LLM come cross-encoder leggero.
    Recupera RERANKER_K candidati, assegna un punteggio 0-10 a ciascuno,
    e restituisce i migliori k dopo il filtraggio per soglia.

    Args:
        chunks   : lista di Document candidati dal retriever ibrido.
        query    : query originale o riscritta.
        llm      : istanza ChatGroq per il reranking.
        k        : numero di chunk finali da restituire.

    Returns:
        Lista ordinata dei migliori chunk dopo reranking e filtro soglia.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    parser = StrOutputParser()
    punteggi = []

    for chunk in chunks:
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_RERANKING.format(
                query=query,
                chunk=chunk.page_content[:500],
            ))
        ])
        try:
            chain     = prompt | llm | parser
            risultato = chain.invoke({}).strip()
            # Estrae il primo numero intero dalla risposta
            numeri = [int(x) for x in risultato.split() if x.isdigit()]
            punteggio = numeri[0] if numeri else 0
        except Exception:
            punteggio = 0
        punteggi.append(punteggio)

    # Associa punteggi e ordina
    chunk_con_score = sorted(
        zip(chunks, punteggi),
        key=lambda x: x[1],
        reverse=True,
    )

    # Filtra per soglia (scala 0-10 -> soglia minima 3)
    chunk_filtrati = [c for c, s in chunk_con_score if s >= 3]

    if not chunk_filtrati:
        # Fallback: restituisce i migliori k senza filtro
        return [c for c, _ in chunk_con_score[:k]]

    return chunk_filtrati[:k]


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K) -> RetrieverIbrido:
    """
    Restituisce il retriever ibrido (vettoriale + BM25).
    Firma identica al vecchio get_retriever per compatibilita con chatbot.py.
    Il reranking viene applicato in chatbot.py dopo il retrieval.
    """
    return RetrieverIbrido(vector_store=vector_store, k=RERANKER_K)


# ---------------------------------------------------------------------------
# Esecuzione diretta
# ---------------------------------------------------------------------------

import re

if __name__ == "__main__":
    from document_loader import carica_e_taglia_cartella

    cartella_src = os.path.dirname(os.path.abspath(__file__))
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))

    print("=" * 55)
    print("  TEST: creazione/caricamento vector store ibrido")
    print("=" * 55)

    if os.path.isdir(CARTELLA_DB):
        print("DB gia presente - caricamento diretto.\n")
        db = carica_vector_db()
    else:
        print("DB non trovato - avvio pipeline completa.\n")
        chunks = carica_e_taglia_cartella(cartella_raw)
        db     = crea_vector_db(chunks)

    retriever  = get_retriever(db, k=3)
    query_test = "alimenti consigliati merenda bambini"
    print(f"\nQuery di test: {query_test!r}")
    risultati = retriever.invoke(query_test)
    print(f"Chunks restituiti: {len(risultati)}")
    for i, r in enumerate(risultati, 1):
        fonte = r.metadata.get("titolo_documento", "?")
        print(f"  [{i}] {fonte} - {r.page_content[:100]}...")

    print("\nScript terminato.")