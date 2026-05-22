#!/usr/bin/env python3
"""Run RAG evaluation from backend/ (loads .env). Usage: python run_eval.py --mode retrieval"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = Path(__file__).resolve().parent
for p in (_BACKEND, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.run import main

if __name__ == "__main__":
    raise SystemExit(main())
