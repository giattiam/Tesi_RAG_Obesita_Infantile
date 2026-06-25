"""
test_smoke.py
Smoke test del sistema RAG: verifica che i moduli core importino senza errori
e che le funzioni pure e le strutture dati di configurazione siano coerenti.
Non richiede la GOOGLE_API_KEY ne il caricamento del modello di embedding:
i test sono veloci, deterministici e offline.
"""
import json
from pathlib import Path

import pytest

import crea_vector_db
import chatbot

DATASET = Path(__file__).parent.parent / "data" / "eval" / "dataset_valutazione.json"


def test_moduli_core_importabili():
    """chatbot e crea_vector_db devono importare senza errori di sintassi/dipendenze."""
    assert hasattr(chatbot, "AssistenteRAG")
    assert hasattr(chatbot, "get_llm")
    assert hasattr(crea_vector_db, "RetrieverIbrido")
    assert hasattr(crea_vector_db, "get_o_crea_retriever")


def test_costanti_retriever_coerenti():
    assert crea_vector_db.RETRIEVER_K > 0
    assert crea_vector_db.RERANKER_K >= crea_vector_db.RETRIEVER_K
    assert 0.0 <= crea_vector_db.SOGLIA_PERTINENZA <= 1.0
    somma_pesi = crea_vector_db.PESO_VETTORIALE + crea_vector_db.PESO_BM25
    assert somma_pesi == pytest.approx(1.0)


def test_tokenizza():
    """Tokenizzazione: lowercase + split su spazi e punteggiatura."""
    tokens = crea_vector_db._tokenizza("Ciao, MONDO! frutta-verdura")
    assert tokens == ["ciao", "mondo", "frutta", "verdura"]
    assert crea_vector_db._tokenizza("") == []


def test_normalizza():
    """La normalizzazione min-max porta i punteggi in [0, 1]."""
    norm = crea_vector_db.RetrieverIbrido._normalizza(None, [0.0, 5.0, 10.0])
    assert norm == [0.0, 0.5, 1.0]
    # lista vuota -> lista vuota
    assert crea_vector_db.RetrieverIbrido._normalizza(None, []) == []
    # tutti uguali -> tutti 1.0 (nessuna divisione per zero)
    assert crea_vector_db.RetrieverIbrido._normalizza(None, [3.0, 3.0]) == [1.0, 1.0]


def test_profili_struttura():
    """Ogni profilo deve avere i campi attesi e un prompt con il placeholder {context}."""
    assert set(chatbot.PROFILI) == {
        "genitore", "insegnante", "pediatra", "nutrizionista", "ricercatore",
    }
    for chiave, dati in chatbot.PROFILI.items():
        for campo in ("etichetta", "descrizione", "temperature", "prompt"):
            assert campo in dati, f"Profilo '{chiave}' senza campo '{campo}'"
        assert 0.0 <= dati["temperature"] <= 1.0
        assert "{context}" in dati["prompt"], f"Profilo '{chiave}': prompt senza {{context}}"


def test_dataset_valutazione_valido():
    """Il dataset RAGAS deve essere ben formato e usare solo profili noti."""
    assert DATASET.exists(), f"Dataset non trovato: {DATASET}"
    dati = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(dati, list) and len(dati) > 0
    profili_validi = set(chatbot.PROFILI)
    for i, campione in enumerate(dati):
        for campo in ("profilo", "domanda", "ground_truth"):
            assert campo in campione, f"Campione {i} senza campo '{campo}'"
            assert campione[campo].strip(), f"Campione {i}: campo '{campo}' vuoto"
        assert campione["profilo"] in profili_validi, (
            f"Campione {i}: profilo sconosciuto '{campione['profilo']}'"
        )
