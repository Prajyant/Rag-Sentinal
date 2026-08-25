FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy remaining files
COPY configs/ configs/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "rag_sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
