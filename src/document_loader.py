import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pulisci_testo(testo):
    """Pulisce il testo da spazi e a capo in eccesso."""
    testo_pulito = re.sub(r'\n+', '\n', testo)
    testo_pulito = re.sub(r' +', ' ', testo_pulito)
    return testo_pulito.strip()

def carica_e_taglia_cartella(cartella_dati):
    print(f"Sto analizzando la cartella: {cartella_dati}")
    documenti_totali = []

    try:
        # 1. Scansioniamo i file uno per uno all'interno della cartella
        for nome_file in os.listdir(cartella_dati):
            percorso_file = os.path.join(cartella_dati, nome_file)

            # Ignoriamo eventuali sottocartelle
            if not os.path.isfile(percorso_file):
                continue

            # 2. Scegliamo lo strumento giusto in base all'estensione del file
            if nome_file.endswith('.pdf'):
                print(f"📄 Trovato PDF: {nome_file}")
                loader = PyPDFLoader(percorso_file)
                documenti_totali.extend(loader.load())

            elif nome_file.endswith('.txt'):
                print(f"📝 Trovato TXT: {nome_file}")
                # utf-8 serve per leggere correttamente le lettere accentate italiane
                loader = TextLoader(percorso_file, encoding='utf-8')
                documenti_totali.extend(loader.load())

            else:
                # Se ci metti per sbaglio un'immagine o un file non supportato, lo salta
                print(f"⚠️ File ignorato (formato non supportato): {nome_file}")

        print(f"\nTrovate {len(documenti_totali)} pagine/documenti totali.")

        # 3. Pulizia e Metadati
        for doc in documenti_totali:
            doc.page_content = pulisci_testo(doc.page_content)
            # Estraiamo il nome del file per le future citazioni del chatbot
            if 'source' in doc.metadata:
                doc.metadata['titolo_documento'] = os.path.basename(doc.metadata['source'])

        # 4. Taglio il testo (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documenti_totali)
        print(f"I documenti sono stati ripuliti e divisi in {len(chunks)} chunks totali.")
        return chunks

    except Exception as e:
        print(f"ERRORE GRAVE durante il caricamento: {e}")
        return []

# --- TEST DELLO SCRIPT ---
if __name__ == "__main__":
    # Trova la cartella 'src' in cui si trova questo script
    cartella_src = os.path.dirname(os.path.abspath(__file__))

    # Risale di una cartella (..) e poi entra in 'data' e 'raw'
    cartella_raw = os.path.abspath(os.path.join(cartella_src, "..", "data", "raw"))

    print(f"🔍 PERCORSO ASSOLUTO CALCOLATO: {cartella_raw}")

    if os.path.exists(cartella_raw):
        miei_chunks = carica_e_taglia_cartella(cartella_raw)

        if len(miei_chunks) > 0:
            print("\n--- ESEMPIO DEL PRIMO CHUNK ---")
            print(f"FONTE: {miei_chunks[0].metadata.get('titolo_documento', 'Sconosciuto')}")
            print("TESTO:")
            print(miei_chunks[0].page_content)
            print("-------------------------------\n")
    else:
        print(f"❌ ERRORE: La cartella non esiste in questo percorso: {cartella_raw}")