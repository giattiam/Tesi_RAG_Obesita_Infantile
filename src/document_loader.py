"""
document_loader.py
Carica documenti da una cartella (PDF, TXT, MD), li pulisce e li divide in chunks.
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configurazione ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 50          # chunk più corti di così vengono scartati
ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252"]

ESTENSIONI_SUPPORTATE = {
    ".pdf": "PDF",
    ".txt": "TXT",
    ".md":  "Markdown",
}
# ─────────────────────────────────────────────────────────────────────────────


def pulisci_testo(testo: str) -> str:
    """Normalizza spazi e ritorni a capo multipli."""
    testo = re.sub(r"\n{3,}", "\n\n", testo)   # max 2 newline consecutivi
    testo = re.sub(r" {2,}", " ", testo)         # spazi multipli → uno solo
    testo = re.sub(r"\t+", " ", testo)           # tab → spazio
    return testo.strip()


def _carica_txt(percorso: str) -> list:
    """Tenta di caricare un file di testo provando più encoding."""
    for enc in ENCODING_FALLBACKS:
        try:
            loader = TextLoader(percorso, encoding=enc)
            docs = loader.load()
            return docs
        except (UnicodeDecodeError, Exception):
            continue
    print(f"     Impossibile decodificare: {os.path.basename(percorso)} — saltato.")
    return []


def carica_e_taglia_cartella(cartella_dati: str) -> list:
    """
    Scansiona `cartella_dati`, carica i documenti supportati,
    li pulisce e li divide in chunks pronti per il vector store.

    Returns:
        Lista di Document (LangChain) con metadati arricchiti.
    """
    cartella_dati = os.path.abspath(cartella_dati)
    print(f"\n Cartella sorgente: {cartella_dati}")

    if not os.path.isdir(cartella_dati):
        print(f" ERRORE: la cartella non esiste → {cartella_dati}")
        return []

    documenti_totali = []
    contatori = {ext: 0 for ext in ESTENSIONI_SUPPORTATE}
    ignorati  = []

    for nome_file in sorted(os.listdir(cartella_dati)):
        percorso_file = os.path.join(cartella_dati, nome_file)
        if not os.path.isfile(percorso_file):
            continue

        estensione = os.path.splitext(nome_file)[1].lower()

        if estensione == ".pdf":
            print(f"   PDF     → {nome_file}")
            try:
                loader = PyPDFLoader(percorso_file)
                documenti_totali.extend(loader.load())
                contatori[".pdf"] += 1
            except Exception as e:
                print(f"     Errore nel caricare {nome_file}: {e}")

        elif estensione in (".txt", ".md"):
            etichetta = ESTENSIONI_SUPPORTATE[estensione]
            print(f"   {etichetta:<9}→ {nome_file}")
            docs = _carica_txt(percorso_file)
            documenti_totali.extend(docs)
            if docs:
                contatori[estensione] += 1

        else:
            ignorati.append(nome_file)

    if ignorati:
        print(f"\n    File ignorati ({len(ignorati)}): {', '.join(ignorati)}")

    print(f"\n Pagine/sezioni caricate: {len(documenti_totali)}")
    print(f"   PDF: {contatori['.pdf']} file | "
          f"TXT: {contatori['.txt']} file | "
          f"MD: {contatori['.md']} file")

    if not documenti_totali:
        print("  Nessun documento caricato. Controlla la cartella.")
        return []

    # ── Pulizia testo e metadati ─────────────────────────────────────────────
    for doc in documenti_totali:
        doc.page_content = pulisci_testo(doc.page_content)
        source = doc.metadata.get("source", "")
        doc.metadata["titolo_documento"] = os.path.basename(source)

    # ── Chunking ─────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documenti_totali)

    # Scarta chunks troppo corti (intestazioni orfane, pagine vuote, ecc.)
    chunks_validi = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    scartati = len(chunks) - len(chunks_validi)

    print(f"  Chunks creati: {len(chunks_validi)}"
          + (f" ({scartati} troppo corti scartati)" if scartati else ""))

    return chunks_validi


# ── Esecuzione diretta (test) ─────────────────────────────────────────────────
if __name__ == "__main__":
    cartella_src = os.path.dirname(os.path.abspath(__file__))
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))

    print(f" Percorso assoluto: {cartella_raw}")

    chunks = carica_e_taglia_cartella(cartella_raw)

    if chunks:
        print("\n--- ESEMPIO: PRIMO CHUNK ---")
        print(f"FONTE : {chunks[0].metadata.get('titolo_documento', 'Sconosciuto')}")
        print(f"PAGINA: {chunks[0].metadata.get('page', 'n/d')}")
        print(f"TESTO :\n{chunks[0].page_content}")
        print("----------------------------\n")
    else:
        print("\n  Nessun chunk prodotto.")