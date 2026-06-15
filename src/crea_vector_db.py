"""
crea_vector_db.py
Crea o carica un vector store ChromaDB e un indice BM25 parallelo.
Espone un retriever ibrido che combina similarity search vettoriale e BM25
keyword-based per migliorare la precisione su termini tecnici specifici.
"""
import os
import pickle
import re
import hashlib
import threading
from functools import lru_cache
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
load_dotenv()
CARTELLA_DB    = "./chroma_db"
PERCORSO_BM25  = "./chroma_db/bm25_index.pkl"
MODELLO_EMBED  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RETRIEVER_K       = 5
PESO_VETTORIALE   = 0.6
PESO_BM25         = 0.4
SOGLIA_PERTINENZA = 0.25
RERANKER_K        = 15

@lru_cache(maxsize=1)

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Istanzia il modello di embedding HuggingFace in locale (singleton di processo).
    Cachato con lru_cache: la prima chiamata carica il modello (~500 MB), le
    successive riusano la stessa istanza. Evita doppio caricamento da
    carica_vector_db + RetrieverIbrido.__init__.
    """
    return HuggingFaceEmbeddings(
        model_name=MODELLO_EMBED,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _tokenizza(testo: str) -> List[str]:
    """Tokenizzazione minima: lowercase e split su spazi/punteggiatura."""
    return re.sub(r"[^\w\s]", " ", testo.lower()).split()

def _doc_id(doc: Document) -> str:
    """
    Id stabile per la deduplica cross-sorgente (vettoriale + BM25).
    Hash dell'INTERO page_content: lo stesso chunk trovato da entrambe le
    sorgenti collide e viene fuso, mentre chunk diversi che condividono il
    prefisso (es. la stessa intestazione "[Documento: ...]" lunga > 80 char)
    restano distinti. Sostituisce il vecchio page_content[:80], che faceva
    collassare in un unico candidato tutti i chunk dei documenti con titolo
    lungo (linee guida ministeriali).
    """
    if os.getenv("RAG_DOCID_LEGACY", "0") == "1":
        # Comportamento pre-fix (per ablation/confronto): collide sui titoli lunghi.
        return doc.page_content[:80]
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

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

def _aggiungi_intestazione(chunks: list) -> list:
    """
    Antepone il titolo del documento al testo di ogni chunk prima
    dell'indicizzazione (contextual chunk header). Rende ogni chunk
    raggiungibile per nome documento sia via BM25 (keyword match sul
    titolo) che via similarity search (il titolo entra nell'embedding).
    Mitiga la sotto-rappresentazione dei documenti con pochi chunk:
    una query che nomina il documento (es. "OKkio alla salute") matcha
    tutti i suoi chunk, non solo quelli che ne citano il nome nel testo.
    """
    for doc in chunks:
        titolo = doc.metadata.get("titolo_documento", "")
        titolo = os.path.splitext(titolo)[0].strip()
        if titolo and not doc.page_content.startswith("[Documento:"):
            doc.page_content = f"[Documento: {titolo}]\n{doc.page_content}"
    return chunks

def crea_vector_db(chunks: list) -> Chroma:
    """
    Crea ChromaDB e indice BM25 a partire dalla lista di chunks.
    Entrambi vengono persistiti su disco.
    Raises:
        ValueError: se la lista di chunks e vuota.
    """
    if not chunks:
        raise ValueError("La lista di chunks e vuota.")
    chunks = _aggiungi_intestazione(chunks)
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
        candidati = {}
        n_candidati = self.k * 2
        risultati_vett = self.vector_store.similarity_search_with_score(
            query, k=n_candidati
        )
        punteggi_vett = [score for _, score in risultati_vett]
        punteggi_vett_inv = [1 - s for s in punteggi_vett]
        norm_vett = self._normalizza(punteggi_vett_inv)
        for (doc, _), score in zip(risultati_vett, norm_vett):
            doc_id = _doc_id(doc)
            candidati[doc_id] = {
                "doc":        doc,
                "score_vett": score,
                "score_bm25": 0.0,
            }
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
                doc_id = _doc_id(doc)
                if doc_id in candidati:
                    candidati[doc_id]["score_bm25"] = score
                else:
                    candidati[doc_id] = {
                        "doc":        doc,
                        "score_vett": 0.0,
                        "score_bm25": score,
                    }
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
        filtrati = [e for e in ordinati if e["score_ibrido"] >= SOGLIA_PERTINENZA]
        if not filtrati:
            filtrati = ordinati
        return [entry["doc"] for entry in filtrati[: self.k]]
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
        llm      : istanza LLM (ChatOpenAI/OpenRouter) per il reranking.
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
            numeri = [int(x) for x in risultato.split() if x.isdigit()]
            punteggio = numeri[0] if numeri else 0
        except Exception:
            punteggio = 0
        punteggi.append(punteggio)
    chunk_con_score = sorted(
        zip(chunks, punteggi),
        key=lambda x: x[1],
        reverse=True,
    )
    chunk_filtrati = [c for c, s in chunk_con_score if s >= 3]
    if not chunk_filtrati:
        return [c for c, _ in chunk_con_score[:k]]
    return chunk_filtrati[:k]

def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K) -> RetrieverIbrido:
    """
    Restituisce il retriever ibrido (vettoriale + BM25).
    Firma identica al vecchio get_retriever per compatibilita con chatbot.py.
    Il reranking viene applicato in chatbot.py dopo il retrieval.
    """
    return RetrieverIbrido(vector_store=vector_store, k=RERANKER_K)
_RETRIEVER_SINGLETON: "RetrieverIbrido | None" = None
_RETRIEVER_LOCK = threading.Lock()

def get_o_crea_retriever() -> RetrieverIbrido:
    """Restituisce il retriever globale, inizializzandolo al primo accesso."""
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is not None:
        return _RETRIEVER_SINGLETON
    with _RETRIEVER_LOCK:
        if _RETRIEVER_SINGLETON is None:
            db = get_o_crea_vector_db()
            _RETRIEVER_SINGLETON = get_retriever(db)
    return _RETRIEVER_SINGLETON
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
