"""
conftest.py
Configurazione condivisa dei test.
Disabilita il logging di audit dei guardrail durante l'esecuzione dei test:
i test esercitano filtra_input/filtra_output molte volte e non devono
generare file logs/guardrail_*.jsonl che inquinerebbero la dashboard.
"""
import os

os.environ.setdefault("GUARDRAIL_LOG", "0")
