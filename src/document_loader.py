"""
document_loader.py
Carica documenti da una cartella (PDF, TXT, MD), li pulisce e li divide in chunks.
Utilizza il chunking semantico come strategia principale: le divisioni avvengono
dove cambia il contenuto, non a intervalli fissi di caratteri.
In caso di fallimento del chunking semantico su un documento, viene applicato
automaticamente il RecursiveCharacterTextSplitter come fallback.
"""
import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crea_vector_db import _get_embeddings
CHUNK_SIZE_FALLBACK    = 1000
CHUNK_OVERLAP_FALLBACK = 200
MIN_CHUNK_LEN          = 50
ENCODING_FALLBACKS     = ["utf-8", "latin-1", "cp1252"]
ESTENSIONI_SUPPORTATE = {
    ".pdf": "PDF",
    ".txt": "TXT",
    ".md":  "Markdown",
}

def pulisci_testo(testo: str) -> str:
    testo = re.sub(r"\n{3,}", "\n\n", testo)
    testo = re.sub(r" {2,}", " ", testo)
    testo = re.sub(r"\t+", " ", testo)
    return testo.strip()

def _carica_txt(percorso: str) -> list:
    for enc in ENCODING_FALLBACKS:
        try:
            loader = TextLoader(percorso, encoding=enc)
            return loader.load()
        except (UnicodeDecodeError, Exception):
            continue
    print(f"   Impossibile decodificare: {os.path.basename(percorso)} - saltato.")
    return []

def _chunking_semantico(documenti: list) -> list:
    """
    Divide i documenti usando SemanticChunker.
    Il punto di taglio viene determinato dalla distanza semantica tra frasi
    consecutive, calcolata tramite il modello di embedding configurato.
    """
    print("\nChunking semantico in corso...")
    embeddings = _get_embeddings()
    splitter   = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks_totali = []
    for doc in documenti:
        try:
            chunks = splitter.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata],
            )
            chunks_totali.extend(chunks)
        except Exception as e:
            print(f"   Chunking semantico fallito su "
                  f"{doc.metadata.get('titolo_documento', '?')}: {e}")
            print(f"   Applicazione fallback RecursiveCharacterTextSplitter.")
            splitter_fallback = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_FALLBACK,
                chunk_overlap=CHUNK_OVERLAP_FALLBACK,
            )
            chunks_totali.extend(splitter_fallback.split_documents([doc]))
    return chunks_totali

def carica_e_taglia_cartella(cartella_dati: str) -> list:
    """
    Scansiona la cartella, carica i documenti supportati, li pulisce
    e li divide in chunks tramite chunking semantico.
    Returns:
        Lista di Document LangChain con metadati arricchiti.
    """
    cartella_dati = os.path.abspath(cartella_dati)
    print(f"\n Cartella sorgente: {cartella_dati}")
    if not os.path.isdir(cartella_dati):
        print(f"Errore: la cartella non esiste - {cartella_dati}")
        return []
    documenti_totali = []
    contatori        = {ext: 0 for ext in ESTENSIONI_SUPPORTATE}
    ignorati         = []
    for nome_file in sorted(os.listdir(cartella_dati)):
        percorso_file = os.path.join(cartella_dati, nome_file)
        if not os.path.isfile(percorso_file):
            continue
        estensione = os.path.splitext(nome_file)[1].lower()
        if estensione == ".pdf":
            print(f"  PDF     -> {nome_file}")
            try:
                loader = PyPDFLoader(percorso_file)
                documenti_totali.extend(loader.load())
                contatori[".pdf"] += 1
            except Exception as e:
                print(f"   Errore nel caricare {nome_file}: {e}")
        elif estensione in (".txt", ".md"):
            etichetta = ESTENSIONI_SUPPORTATE[estensione]
            print(f"  {etichetta:<9} -> {nome_file}")
            docs = _carica_txt(percorso_file)
            documenti_totali.extend(docs)
            if docs:
                contatori[estensione] += 1
        else:
            ignorati.append(nome_file)
    if ignorati:
        print(f"\n  File ignorati ({len(ignorati)}): {', '.join(ignorati)}")
    print(f"\nPagine/sezioni caricate: {len(documenti_totali)}")
    print(f"   PDF: {contatori['.pdf']} file | "
          f"TXT: {contatori['.txt']} file | "
          f"MD: {contatori['.md']} file")
    if not documenti_totali:
        print("Nessun documento caricato.")
        return []
    for doc in documenti_totali:
        doc.page_content = pulisci_testo(doc.page_content)
        source = doc.metadata.get("source", "")
        doc.metadata["titolo_documento"] = os.path.basename(source)
    chunks = _chunking_semantico(documenti_totali)
    chunks_validi = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    scartati      = len(chunks) - len(chunks_validi)
    print(f"Chunks creati: {len(chunks_validi)}"
          + (f" ({scartati} troppo corti scartati)" if scartati else ""))
    return chunks_validi
if __name__ == "__main__":
    cartella_src = os.path.dirname(os.path.abspath(__file__))
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))
    print(f"Percorso assoluto: {cartella_raw}")
    chunks = carica_e_taglia_cartella(cartella_raw)
    if chunks:
        print("\n--- PRIMO CHUNK ---")
        print(f"Fonte : {chunks[0].metadata.get('titolo_documento', 'Sconosciuto')}")
        print(f"Pagina: {chunks[0].metadata.get('page', 'n/d')}")
        print(f"Testo :\n{chunks[0].page_content}")
        print("-------------------\n")
    else:
        print("Nessun chunk prodotto.")
