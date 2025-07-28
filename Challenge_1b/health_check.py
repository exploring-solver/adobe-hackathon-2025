# health_check.py

import sys
from pathlib import Path

try:
    # Test imports
    import spacy
    import sentence_transformers
    import networkx
    import sklearn

    # Check cache directories
    Path("/app/cache").mkdir(exist_ok=True)
    Path("/app/model_cache").mkdir(exist_ok=True)

    # Test spaCy model
    nlp = spacy.load("en_core_web_sm")

    print("✅ All enhanced features healthy")
    sys.exit(0)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)
