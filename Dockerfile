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
    umap-learn

WORKDIR /app

# Copy API code and model
COPY discogs-coverhunter-itunes/api.py .
COPY discogs-coverhunter-itunes/pipeline.py .
COPY entrypoint.sh .
COPY crawl_songs.py .
COPY update_data.py .
COPY discogs-coverhunter-itunes/model/ ./model/

# Copy static files (web UI) — symlink docs->static so update_data.py writes to the right place
COPY docs/ ./static/
RUN ln -s /app/static /app/docs

# Copy data files (vectors.csv lives on persistent volume /app/data)
COPY videos_to_test.csv .
COPY discogs-coverhunter-itunes/cover_map.json ./

# Expose port
EXPOSE 8080

# Run entrypoint
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
