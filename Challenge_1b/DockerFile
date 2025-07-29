# Official lightweight Python image
FROM python:3.10-slim

# Set environment variables for caching and performance
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/models/huggingface \
    HF_HOME=/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers \
    SPACY_CACHE_DIR=/models/spacy \
    TOKENIZERS_PARALLELISM=false

# Create working directory
WORKDIR /app

# Install system dependencies and Python requirements
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download and cache small models for offline use
RUN python -m spacy download en_core_web_sm \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project source code into the container
COPY . .

# Default command to run your main app
CMD ["python", "main.py"]
