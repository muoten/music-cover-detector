FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    flask \
    librosa \
    numpy \
    pandas \
    pyyaml \
    scikit-learn \
    scipy \
    tqdm \
    umap-learn \
    transformers \
    einops \
    timm

WORKDIR /app

# Copy API code and model
COPY discogs-coverhunter-itunes/api.py .
COPY discogs-coverhunter-itunes/pipeline.py .
COPY discogs-coverhunter-itunes/livi_model.py .
COPY entrypoint.sh .
COPY crawl_songs.py .
COPY update_data.py .
COPY discogs-coverhunter-itunes/model/ ./model/

# Pre-download Whisper model so first startup isn't slow
RUN python -c "from transformers import WhisperModel, WhisperFeatureExtractor; \
    WhisperModel.from_pretrained('openai/whisper-large-v3-turbo'); \
    WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3-turbo')"

# Copy static files (web UI) — symlink docs->static so update_data.py writes to the right place
COPY docs/ ./static/
RUN ln -s /app/static /app/docs

# Expose port
EXPOSE 8080

# Run entrypoint
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
