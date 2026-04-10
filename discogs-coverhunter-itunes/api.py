#!/usr/bin/env python3
"""
Flask API for CoverHunter cover detection.

Serves the Music Cover Detector web app and API.
Accepts a YouTube URL, computes CoverHunter embedding via iTunes preview,
and returns top similar songs from the database.

Usage:
    python api.py [--port 5000] [--host 0.0.0.0]
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

# Add parent dir to path so model imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model.utils import load_model

# Import functions from pipeline
from pipeline import (
    get_youtube_title,
    get_youtube_info,
    get_youtube_metadata,
    parse_artist_track,
    search_itunes,
    search_itunes_detailed,
    lookup_itunes_by_id,
    smart_search_itunes,
    ITunesRateLimitError,
    download_preview,
    convert_to_wav,
    compute_cqt_features,
    compute_embedding,
    load_audio,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Static files directory
STATIC_DIR = os.path.join(SCRIPT_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR)

# ============================================================================
# Global state (loaded on startup)
# ============================================================================

model = None
device = None
embeddings_matrix = None  # (N, 128) normalized
video_ids = []  # list of YouTube IDs in same order as embeddings
video_metadata = {}  # youtube_id -> {title, artist, track}
clique_map = {}  # youtube_id -> clique_id
cover_map = {}  # youtube_id -> [list of cover youtube_ids]
embedding_hashes = {}  # youtube_id -> hash of embedding
track_ids = {}  # youtube_id -> iTunes trackId
_dup_cache = {}  # cached duplicate track_id stats
_dup_cache_dirty = True  # flag to recompute dup stats
db_stats = {}  # database statistics
db_lock = threading.Lock()  # protects all database mutations + stats
last_regen_count = 0  # song count at last data.json regeneration threshold
regen_running = False  # True while update_data.py is running in background

# LIVI model (score fusion)
livi_model = None
whisper_encoder = None
livi_embeddings_matrix = None  # (M, 768) normalized, M <= N
livi_video_ids = []             # parallel to livi_embeddings_matrix
livi_vid_to_idx = {}            # youtube_id -> index in livi_embeddings_matrix
FUSION_WEIGHT_CH = 0.60
FUSION_WEIGHT_LIVI = 0.40


def compute_embedding_hash(embedding):
    """Compute a short hash of an embedding vector."""
    emb_bytes = embedding.astype(np.float32).tobytes()
    return hashlib.md5(emb_bytes).hexdigest()[:8]


def build_cover_map():
    """Build bidirectional cover map from clique_map.

    Groups videos by clique and maps each video to all other videos in the
    same clique. This replaces the static cover_map.json file.
    """
    global cover_map
    cover_map = {}

    # Group video IDs by clique
    clique_to_vids = {}
    for vid, clique_id in clique_map.items():
        clique_to_vids.setdefault(clique_id, []).append(vid)

    # Build bidirectional map
    for clique_id, vids in clique_to_vids.items():
        if len(vids) < 2:
            continue
        for vid in vids:
            cover_map[vid] = [other for other in vids if other != vid]

    db_stats['cover_map_songs'] = len(cover_map)
    db_stats['cliques'] = len(clique_to_vids)
    logging.info(f"Built cover_map: {len(cover_map)} songs with covers ({len(clique_to_vids)} cliques)")


def load_clique_map():
    """Load clique IDs from videos_to_test.csv."""
    global clique_map

    csv_paths = [
        '/app/data/videos_to_test.csv',
        os.path.join(SCRIPT_DIR, '..', 'videos_to_test.csv'),
        os.path.join(SCRIPT_DIR, 'videos_to_test.csv'),
        '/code/videos_to_test.csv',
    ]

    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            logging.info(f"Loading clique map from {csv_path}...")
            with open(csv_path, 'r') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        clique_id = parts[0]
                        url = parts[1]
                        if 'v=' in url:
                            vid = url.split('v=')[1].split('&')[0]
                            clique_map[vid] = clique_id
            logging.info(f"Loaded {len(clique_map)} clique mappings")
            discogs_songs = sum(1 for vid in video_ids if vid in clique_map and clique_map[vid].startswith('C'))
            db_stats['discogs_songs'] = discogs_songs
            logging.info(f"Songs from Discogs: {discogs_songs}/{len(video_ids)}")
            build_cover_map()
            return

    logging.info("No clique map found")


def _log_user_search(youtube_id, title):
    """Log a non-Discogs user search to user_searches.csv."""
    path = '/app/data/user_searches.csv' if os.path.isdir('/app/data') else 'user_searches.csv'
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a') as f:
            if write_header:
                f.write('video_id,title,clique,timestamp\n')
            safe_title = title.replace(',', ' ')
            f.write(f'{youtube_id},{safe_title},,{time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
    except Exception as e:
        logging.error(f"Failed to log user search: {e}")


REGEN_INTERVAL = 500  # regenerate data.json every N new songs


def check_data_regen():
    """Check if song count crossed a REGEN_INTERVAL boundary and trigger data.json regeneration."""
    global last_regen_count, regen_running

    current_count = len(video_ids)
    if current_count // REGEN_INTERVAL > last_regen_count // REGEN_INTERVAL:
        logging.info(
            f"Song count crossed {REGEN_INTERVAL}-boundary: {last_regen_count} -> {current_count}, "
            f"triggering data.json regeneration"
        )
        last_regen_count = current_count
        _run_data_regen()
    else:
        last_regen_count = current_count


def _run_data_regen():
    """Run update_data.py in a background thread."""
    global regen_running

    if regen_running:
        logging.info("data.json regeneration already running, skipping")
        return

    def _regen():
        global regen_running
        regen_running = True
        try:
            # In Docker, update_data.py is at /app/, static files at /app/static/
            # Locally, it's at the repo root with docs/ for output
            if os.path.exists('/app/update_data.py'):
                regen_dir = '/app'
            else:
                regen_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
            persistent_path = '/app/data/vectors.csv'
            if os.path.exists(persistent_path):
                vectors_path = persistent_path
            else:
                vectors_path = os.path.join(SCRIPT_DIR, '..', 'vectors.csv')
            cmd = [
                sys.executable, 'update_data.py',
                '--vectors', os.path.abspath(vectors_path),
                '--skip-training',
            ]
            logging.info(f"Running: {' '.join(cmd)} in {regen_dir}")
            proc = subprocess.run(
                cmd,
                cwd=regen_dir,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                logging.info("data.json regeneration completed successfully")
            else:
                logging.error(f"data.json regeneration failed (exit {proc.returncode})")
                if proc.stderr:
                    logging.error(f"stderr: {proc.stderr[:500]}")
        except Exception as e:
            logging.error(f"data.json regeneration error: {e}")
        finally:
            regen_running = False
            # Re-check in case we crossed another boundary while running
            check_data_regen()

    thread = threading.Thread(target=_regen, daemon=True)
    thread.start()


def load_livi_vectors():
    """Load pre-computed LIVI embeddings from vectors_livi.csv."""
    global livi_embeddings_matrix, livi_video_ids, livi_vid_to_idx

    livi_paths = [
        '/app/data/vectors_livi.csv',
        os.path.join(SCRIPT_DIR, '..', 'vectors_livi.csv'),
        os.path.join(SCRIPT_DIR, 'vectors_livi.csv'),
    ]
    livi_path = None
    for p in livi_paths:
        if os.path.exists(p):
            livi_path = p
            break

    if not livi_path:
        logging.info("No LIVI vectors file found, fusion will use CoverHunter-only")
        livi_embeddings_matrix = None
        livi_video_ids.clear()
        livi_vid_to_idx.clear()
        return

    logging.info(f"Loading LIVI embeddings from {livi_path}...")
    vids = []
    embs = []
    with open(livi_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid = parts[0]
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == 768:
                    vids.append(vid)
                    embs.append(vec)
            except Exception:
                continue

    if not embs:
        logging.info("LIVI vectors file was empty")
        livi_embeddings_matrix = None
        livi_video_ids.clear()
        livi_vid_to_idx.clear()
        return

    livi_video_ids.clear()
    livi_video_ids.extend(vids)

    mat = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    livi_embeddings_matrix = mat / norms

    livi_vid_to_idx = {v: i for i, v in enumerate(livi_video_ids)}
    logging.info(f"Loaded {len(livi_video_ids)} LIVI embeddings (768-dim)")


def load_database():
    """Load the CoverHunter embeddings database."""
    global embeddings_matrix, video_ids, video_metadata, embedding_hashes

    # Look for vectors file
    persistent_path = '/app/data/vectors.csv'
    if os.path.exists(persistent_path):
        vectors_path = persistent_path
    else:
        vectors_path = os.path.join(SCRIPT_DIR, '..', 'vectors.csv')
        if not os.path.exists(vectors_path):
            vectors_path = os.path.join(SCRIPT_DIR, 'vectors.csv')
            if not os.path.exists(vectors_path):
                vectors_path = os.path.join(SCRIPT_DIR, 'vectors_coverhunter.csv')

    embeddings = []
    video_ids_local = []

    # Load removed IDs to exclude from loading
    removed_path = '/app/data/removed.txt' if os.path.isdir('/app/data') else 'removed.txt'
    removed_ids = set()
    if os.path.exists(removed_path):
        with open(removed_path, 'r') as f:
            removed_ids = {line.strip() for line in f if line.strip()}
        if removed_ids:
            logging.info(f"Excluding {len(removed_ids)} removed videos")

    logging.info(f"Loading embeddings from {vectors_path}...")
    with open(vectors_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid_or_path = parts[0]
            # Extract YouTube ID from path like /tmp/.../ABC123.wav
            if '/' in vid_or_path:
                vid = os.path.basename(vid_or_path).replace('.wav', '')
            else:
                vid = vid_or_path
            if vid in removed_ids:
                continue
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == 128:  # CoverHunter is 128-dim
                    video_ids_local.append(vid)
                    embeddings.append(vec)
            except:
                continue

    video_ids.clear()
    video_ids.extend(video_ids_local)

    embeddings_matrix_local = np.array(embeddings, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings_matrix_local, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_matrix_local = embeddings_matrix_local / norms

    global embeddings_matrix
    embeddings_matrix = embeddings_matrix_local

    # Compute hashes for each embedding
    logging.info("Computing embedding hashes...")
    embedding_hashes.clear()
    for i, vid in enumerate(video_ids):
        embedding_hashes[vid] = compute_embedding_hash(embeddings_matrix[i])

    # Compute collision stats
    hash_to_vids = {}
    for vid, h in embedding_hashes.items():
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    songs_with_collisions = sum(len(vids) for vids in hash_to_vids.values() if len(vids) > 1)
    unique_hashes = len(hash_to_vids)
    total_songs = len(video_ids)

    db_stats['total_songs'] = total_songs
    db_stats['unique_hashes'] = unique_hashes
    db_stats['songs_with_collisions'] = songs_with_collisions
    db_stats['collision_ratio'] = round(songs_with_collisions / total_songs, 4) if total_songs > 0 else 0

    logging.info(f"Loaded {len(video_ids)} embeddings")
    logging.info(f"Hash collisions: {songs_with_collisions}/{total_songs} ({db_stats['collision_ratio']*100:.1f}%)")

    # Load trackId mappings
    global track_ids
    persistent_track_ids = '/app/data/track_ids.json'
    if os.path.exists(persistent_track_ids):
        track_ids_path = persistent_track_ids
    else:
        track_ids_path = os.path.join(SCRIPT_DIR, 'track_ids.json')
    progress_path = os.path.join(SCRIPT_DIR, 'progress.json')

    bak_path = track_ids_path + '.bak'
    if os.path.exists(track_ids_path):
        try:
            with open(track_ids_path, 'r') as f:
                track_ids.update(json.load(f))
            logging.info(f"Loaded {len(track_ids)} trackId mappings from track_ids.json")
            shutil.copy2(track_ids_path, bak_path)
            logging.info(f"Backed up track_ids.json ({len(track_ids)} entries)")
        except json.JSONDecodeError:
            logging.error("track_ids.json is corrupted, attempting recovery from backup")
            if os.path.exists(bak_path):
                with open(bak_path, 'r') as f:
                    track_ids.update(json.load(f))
                # Restore the backup as the main file
                shutil.copy2(bak_path, track_ids_path)
                logging.info(f"Recovered {len(track_ids)} trackId mappings from backup")
            else:
                logging.error("No backup available for track_ids.json")
    elif os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        track_ids.update(progress.get('track_ids', {}))
        logging.info(f"Loaded {len(track_ids)} trackId mappings from progress.json")

    # One-time migration: merge track_ids_new.json into track_ids.json
    new_tid_path = '/app/data/track_ids_new.json' if os.path.exists('/app/data') else os.path.join(SCRIPT_DIR, 'track_ids_new.json')
    if os.path.exists(new_tid_path):
        with open(new_tid_path, 'r') as f:
            new_tids = json.load(f)
        if new_tids:
            track_ids.update(new_tids)
            with open(track_ids_path, 'w') as f:
                json.dump(dict(track_ids), f)
            logging.info(f"Merged {len(new_tids)} entries from track_ids_new.json into track_ids.json (total: {len(track_ids)})")
        os.remove(new_tid_path)
        logging.info("Removed track_ids_new.json")

    # Compute initial stats (including dup_track_ids)
    _recompute_stats()

    # Load LIVI vectors for score fusion
    load_livi_vectors()


def _compute_fused_similarities(query_idx):
    """Compute fused similarity scores for a query (by CH index).

    Returns (N,) array of fused scores for all songs in the database.
    """
    ch_sims = embeddings_matrix[query_idx] @ embeddings_matrix.T  # (N,)

    if livi_embeddings_matrix is None or len(livi_video_ids) == 0:
        return ch_sims

    query_vid = video_ids[query_idx]
    query_livi_idx = livi_vid_to_idx.get(query_vid)

    if query_livi_idx is None:
        # Query has no LIVI embedding — fall back to CH-only
        return ch_sims

    query_livi = livi_embeddings_matrix[query_livi_idx]  # (768,)
    fused = np.full_like(ch_sims, 0.0)

    for i, vid in enumerate(video_ids):
        li = livi_vid_to_idx.get(vid)
        if li is not None:
            livi_sim = float(query_livi @ livi_embeddings_matrix[li])
            fused[i] = FUSION_WEIGHT_CH * ch_sims[i] + FUSION_WEIGHT_LIVI * livi_sim
        else:
            # No LIVI embedding — scale CH-only to same range as fused scores
            fused[i] = FUSION_WEIGHT_CH * ch_sims[i]

    return fused


_P1_CACHE_PATH = '/app/data/p1_cache.json' if os.path.isdir('/app/data') else os.path.join(SCRIPT_DIR, 'p1_cache.json')


def _load_p1_cache():
    """Load cached P@1 stats from disk into db_stats."""
    try:
        with open(_P1_CACHE_PATH) as f:
            cached = json.load(f)
        for key in ('precision_at_1', 'precision_at_1_se', 'precision_at_1_hits',
                     'precision_at_1_time', 'evaluable_songs', 'model_name', 'livi_coverage'):
            if key in cached:
                db_stats[key] = cached[key]
        logging.info(f"Loaded cached P@1: {cached.get('precision_at_1', '?')} ({cached.get('model_name', '?')})")
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _save_p1_cache():
    """Save current P@1 stats to disk."""
    try:
        cached = {k: db_stats[k] for k in ('precision_at_1', 'precision_at_1_se', 'precision_at_1_hits',
                  'precision_at_1_time', 'evaluable_songs', 'model_name', 'livi_coverage') if k in db_stats}
        with open(_P1_CACHE_PATH, 'w') as f:
            json.dump(cached, f)
    except Exception as e:
        logging.error(f"Failed to save P@1 cache: {e}")


def compute_precision_at_1():
    """Compute Precision@1 using the in-memory cover_map (with fusion if available)."""
    global db_stats

    start_time = time.time()

    if not cover_map:
        logging.info("No cover_map available, skipping P@1 computation")
        return

    vid_to_idx = {vid: i for i, vid in enumerate(video_ids)}

    eval_songs = []
    for vid in video_ids:
        if vid not in cover_map:
            continue
        covers_in_set = [c for c in cover_map[vid] if c in vid_to_idx]
        if covers_in_set:
            eval_songs.append((vid, covers_in_set))

    if not eval_songs:
        logging.info("No evaluable songs for P@1")
        return

    use_fusion = livi_embeddings_matrix is not None and len(livi_video_ids) > 0

    hits = 0
    for vid, covers in eval_songs:
        query_idx = vid_to_idx[vid]
        cover_idxs = set(vid_to_idx[c] for c in covers)

        if use_fusion:
            sims = _compute_fused_similarities(query_idx)
        else:
            sims = embeddings_matrix[query_idx] @ embeddings_matrix.T

        sims[query_idx] = -2
        nn_idx = int(np.argmax(sims))
        if nn_idx in cover_idxs:
            hits += 1

    elapsed = time.time() - start_time
    n = len(eval_songs)
    p_at_1 = hits / n
    se = np.sqrt(p_at_1 * (1 - p_at_1) / n)
    model_name = 'CoverHunter+LIVI' if use_fusion else 'CoverHunter'
    db_stats['precision_at_1'] = round(p_at_1, 4)
    db_stats['precision_at_1_se'] = round(se, 4)
    db_stats['precision_at_1_hits'] = hits
    db_stats['precision_at_1_time'] = round(elapsed, 2)
    db_stats['evaluable_songs'] = n
    db_stats['model_name'] = model_name
    main_vids = set(video_ids)
    livi_coverage = sum(1 for v in livi_video_ids if v in main_vids)
    db_stats['livi_coverage'] = livi_coverage
    logging.info(f"Precision@1 ({model_name}): {hits}/{n} = {p_at_1:.2%} ± {se:.2%} ({elapsed:.2f}s, LIVI coverage: {livi_coverage})")
    _save_p1_cache()


def init_model():
    """Initialize the CoverHunter model and (optionally) the LIVI model."""
    global model, device, livi_model, whisper_encoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Check persistent volume first, then local
    persistent_checkpoint = '/app/data/coverhunter_checkpoint'
    local_checkpoint = os.path.join(SCRIPT_DIR, 'checkpoint')

    if os.path.exists(os.path.join(persistent_checkpoint, 'hparams.yaml')):
        checkpoint_dir = persistent_checkpoint
    else:
        checkpoint_dir = local_checkpoint

    logging.info(f"Loading CoverHunter model from {checkpoint_dir}")
    model = load_model(checkpoint_dir, device=str(device))
    logging.info("CoverHunter model loaded")

    # LIVI inference model (optional — needs ~1.5GB extra RAM for Whisper + LIVI)
    # Set LIVI_INFERENCE=1 to enable computing LIVI embeddings for new songs.
    # Without it, pre-computed LIVI vectors are still used for fusion at search time.
    if os.environ.get('LIVI_INFERENCE', '').strip() == '1':
        livi_checkpoint_paths = [
            '/app/data/livi_checkpoint/livi.pth',
            os.path.join(SCRIPT_DIR, '..', 'livi_checkpoint', 'livi.pth'),
            '/tmp/LIVI/src/livi/apps/audio_encoder/checkpoints/LIVI/livi.pth',
        ]
        livi_checkpoint = None
        for p in livi_checkpoint_paths:
            if os.path.exists(p):
                livi_checkpoint = p
                break

        if livi_checkpoint:
            try:
                from livi_model import init_livi_model
                livi_model, whisper_encoder = init_livi_model(livi_checkpoint, device)
                logging.info("LIVI model loaded for score fusion + new song inference")
            except Exception as e:
                logging.warning(f"Failed to load LIVI model: {e}")
                livi_model = None
                whisper_encoder = None
        else:
            logging.info("LIVI checkpoint not found, running CoverHunter-only mode")
    else:
        logging.info("LIVI inference disabled (set LIVI_INFERENCE=1 to enable). Pre-computed vectors will be used for fusion.")


def compute_embedding_for_video(youtube_id, title=None, channel=None, yt_metadata=None):
    """
    Compute CoverHunter (and optionally LIVI) embedding for a YouTube video via iTunes preview.

    Returns: (ch_embedding, livi_embedding_or_None, track_id, error_msg)
    """
    if title is None:
        title = get_youtube_title(youtube_id)

    if not title:
        return None, None, None, 'Failed to get YouTube title'

    # Search iTunes
    try:
        track_info, track_id = smart_search_itunes(youtube_id, title, channel=channel, yt_metadata=yt_metadata)
    except ITunesRateLimitError:
        return None, None, None, 'iTunes rate limited'
    if not track_info or not track_info.get('previewUrl'):
        return None, None, None, 'No iTunes preview found'

    preview_url = track_info['previewUrl']

    # Download and process
    temp_dir = tempfile.gettempdir()
    m4a_path = os.path.join(temp_dir, f'{youtube_id}.m4a')
    wav_path = os.path.join(temp_dir, f'{youtube_id}.wav')

    try:
        if not download_preview(preview_url, m4a_path):
            return None, None, track_id, 'Failed to download preview'

        if not convert_to_wav(m4a_path, wav_path, sample_rate=16000):
            return None, None, track_id, 'Failed to convert to WAV'

        signal = load_audio(wav_path, sample_rate=16000)
        cqt = compute_cqt_features(signal, sample_rate=16000, hop_size=0.04, mean_size=3)
        ch_embedding = compute_embedding(model, cqt, device=str(device))

        # Compute LIVI embedding from the same WAV (if model loaded)
        livi_emb = None
        if livi_model is not None and whisper_encoder is not None:
            try:
                from livi_model import compute_livi_embedding
                livi_emb = compute_livi_embedding(livi_model, whisper_encoder, wav_path, device)
            except Exception as e:
                logging.warning(f"LIVI embedding failed for {youtube_id}: {e}")

        return ch_embedding, livi_emb, track_id, None

    except Exception as e:
        return None, None, track_id, str(e)

    finally:
        for path in [m4a_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


STATS_INTERVAL = 20  # recompute stats every N seconds


def _recompute_stats():
    """Recompute all db_stats from current state. Called periodically, not on every mutation."""
    total = len(video_ids)
    discogs = sum(1 for vid in video_ids if vid in clique_map and clique_map[vid].startswith('C'))
    hash_to_vids = {}
    for vid, h in embedding_hashes.items():
        hash_to_vids.setdefault(h, []).append(vid)
    unique = len(hash_to_vids)
    collisions = sum(len(vids) for vids in hash_to_vids.values() if len(vids) > 1)
    ratio = round(collisions / total, 4) if total > 0 else 0

    # Count videos with duplicate track_ids (cached, recomputed when dirty)
    global _dup_cache_dirty
    if _dup_cache_dirty or not _dup_cache:
        video_ids_set = set(video_ids)
        tid_to_vids = {}
        for vid, tid in track_ids.items():
            if vid in video_ids_set:
                tid_to_vids.setdefault(tid, []).append(vid)
        dup_vids = set()
        for vids in tid_to_vids.values():
            if len(vids) > 1:
                dup_vids.update(vids)

        resolved_set = set()
        resolved_path = '/app/data/duplicates_resolved.txt' if os.path.isdir('/app/data') else 'duplicates_resolved.txt'
        if os.path.exists(resolved_path):
            try:
                with open(resolved_path, 'r') as f:
                    resolved_set = {line.strip() for line in f if line.strip()}
            except Exception:
                pass

        no_track_id = len(video_ids_set - set(track_ids.keys()))

        _dup_cache.update({
            'dup_track_ids': len(dup_vids),
            'dup_track_ids_resolved': len(dup_vids & resolved_set),
            'no_track_id': no_track_id,
        })
        _dup_cache_dirty = False

    dup_track_ids = _dup_cache['dup_track_ids']
    dup_resolved = _dup_cache['dup_track_ids_resolved']
    no_track_id = _dup_cache.get('no_track_id', 0)

    # Cheaply count evaluable songs (have at least one cover in the database)
    evaluable = db_stats.get('evaluable_songs', 0)
    if cover_map:
        vid_set = set(video_ids)
        evaluable = sum(
            1 for vid in video_ids
            if vid in cover_map and any(c in vid_set for c in cover_map[vid])
        )

    # Count pipeline-level stats from data files
    source_path = '/app/data/videos_to_test.csv' if os.path.isdir('/app/data') else 'videos_to_test.csv'
    errors_path = '/app/data/errors.csv' if os.path.isdir('/app/data') else 'errors.csv'
    removed_path2 = '/app/data/removed.txt' if os.path.isdir('/app/data') else 'removed.txt'
    tried = 0
    no_itunes = 0
    removed_count = 0
    try:
        with open(source_path, 'r') as f:
            tried = sum(1 for _ in f) - 1  # minus header
    except Exception:
        pass
    try:
        with open(errors_path, 'r') as f:
            for line in f:
                reason = line.split(',', 1)[1].lower() if ',' in line else ''
                if 'no itunes' in reason or 'no preview' in reason:
                    no_itunes += 1
    except Exception:
        pass
    try:
        with open(removed_path2, 'r') as f:
            removed_count = sum(1 for line in f if line.strip())
    except Exception:
        pass

    db_stats.update({
        'total_songs': total,
        'discogs_songs': discogs,
        'user_searches': total - discogs,
        'unique_hashes': unique,
        'songs_with_collisions': collisions,
        'collision_ratio': ratio,
        'dup_track_ids': dup_track_ids,
        'dup_track_ids_resolved': dup_resolved,
        'no_track_id': no_track_id,
        'evaluable_songs': evaluable,
        'tried': tried,
        'no_itunes_match': no_itunes,
        'removed_duplicates': removed_count,
        'livi_coverage': sum(1 for v in livi_video_ids if v in set(video_ids)),
    })

    # Remove stale crawl progress (no update for 5 minutes)
    cp = db_stats.get('crawl_progress')
    if cp and time.time() - cp.get('updated_at', 0) > 300:
        del db_stats['crawl_progress']


def _stats_timer():
    """Background thread that recomputes stats every STATS_INTERVAL seconds."""
    while True:
        time.sleep(STATS_INTERVAL)
        try:
            _recompute_stats()
        except Exception as e:
            logging.error(f"Stats recompute error: {e}")


def remove_from_database(youtube_id):
    """Remove a song from the in-memory database (e.g., when re-crawl finds artist mismatch)."""
    global embeddings_matrix, livi_embeddings_matrix

    with db_lock:
        if youtube_id not in video_ids:
            return

        idx = video_ids.index(youtube_id)
        video_ids.pop(idx)
        embeddings_matrix = np.delete(embeddings_matrix, idx, axis=0)
        embedding_hashes.pop(youtube_id, None)
        if youtube_id in track_ids:
            track_ids.pop(youtube_id)
            global _dup_cache_dirty
            _dup_cache_dirty = True

        # Remove from LIVI index
        li = livi_vid_to_idx.pop(youtube_id, None)
        if li is not None and livi_embeddings_matrix is not None:
            livi_video_ids.pop(li)
            livi_embeddings_matrix = np.delete(livi_embeddings_matrix, li, axis=0)
            # Rebuild index after deletion
            for i, v in enumerate(livi_video_ids):
                livi_vid_to_idx[v] = i

    # Persist removal to disk so it survives restarts
    removed_path = '/app/data/removed.txt' if os.path.isdir('/app/data') else 'removed.txt'
    try:
        with open(removed_path, 'a') as f:
            f.write(youtube_id + '\n')
    except Exception as e:
        logging.error(f"Failed to persist removal of {youtube_id}: {e}")

    logging.info(f"Removed {youtube_id} from database ({len(video_ids)} songs remaining)")


def batch_remove_from_database(ids_to_remove):
    """Remove multiple songs from the in-memory database in one batch."""
    global embeddings_matrix, _dup_cache_dirty, livi_embeddings_matrix

    removed = []
    with db_lock:
        indices_to_delete = []
        for vid in ids_to_remove:
            if vid in video_ids:
                indices_to_delete.append(video_ids.index(vid))
                removed.append(vid)

        if not indices_to_delete:
            return removed

        # Delete from numpy array in one call
        embeddings_matrix = np.delete(embeddings_matrix, indices_to_delete, axis=0)

        # Remove from lists/dicts in reverse index order to preserve indices
        for idx in sorted(indices_to_delete, reverse=True):
            video_ids.pop(idx)

        for vid in removed:
            embedding_hashes.pop(vid, None)
            if vid in track_ids:
                track_ids.pop(vid)
                _dup_cache_dirty = True

        # Remove from LIVI index (batch)
        livi_indices_to_delete = []
        for vid in removed:
            li = livi_vid_to_idx.pop(vid, None)
            if li is not None:
                livi_indices_to_delete.append(li)
        if livi_indices_to_delete and livi_embeddings_matrix is not None:
            for li in sorted(livi_indices_to_delete, reverse=True):
                livi_video_ids.pop(li)
            livi_embeddings_matrix = np.delete(livi_embeddings_matrix, livi_indices_to_delete, axis=0)
            # Rebuild index
            livi_vid_to_idx.clear()
            for i, v in enumerate(livi_video_ids):
                livi_vid_to_idx[v] = i

    # Persist removals to disk
    removed_path = '/app/data/removed.txt' if os.path.isdir('/app/data') else 'removed.txt'
    try:
        with open(removed_path, 'a') as f:
            for vid in removed:
                f.write(vid + '\n')
    except Exception as e:
        logging.error(f"Failed to persist batch removal: {e}")

    logging.info(f"Batch removed {len(removed)} songs from database ({len(video_ids)} remaining)")
    return removed


def add_embedding_to_database(youtube_id, embedding, track_id=None, force=False, livi_embedding=None):
    """
    Add a new embedding to the in-memory database and persistent storage.
    Called asynchronously after returning search results.
    """
    global embeddings_matrix, _dup_cache_dirty, livi_embeddings_matrix

    # Normalize embedding
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    embedding = embedding.astype(np.float32)

    new_hash = compute_embedding_hash(embedding)

    with db_lock:
        # Check if this hash already exists (collision)
        colliding_with = [v for v, h in embedding_hashes.items() if h == new_hash and v != youtube_id]
        if colliding_with:
            db_stats['last_collision_event'] = (
                f"{youtube_id} (track_id={track_id}) collides with "
                f"{colliding_with[:3]} (track_id={track_ids.get(colliding_with[0])})"
            )
            logging.warning(f"Collision: {youtube_id} (tid={track_id}) == {colliding_with[:3]}")

        # Update in-place if already in database
        is_new = youtube_id not in video_ids
        if not is_new:
            idx = video_ids.index(youtube_id)
            old_track_id = track_ids.get(youtube_id)
            if track_id and track_id == old_track_id and not force:
                return  # same iTunes match and not forced, nothing to update
            embeddings_matrix[idx] = embedding
            embedding_hashes[youtube_id] = new_hash
            if track_id:
                track_ids[youtube_id] = track_id
                _dup_cache_dirty = True
            logging.info(f"Updated embedding for {youtube_id}: track_id {old_track_id} -> {track_id}")
        else:
            logging.info(f"Adding new embedding to database: {youtube_id}")
            video_ids.append(youtube_id)
            embeddings_matrix = np.vstack([embeddings_matrix, embedding.reshape(1, -1)])
            embedding_hashes[youtube_id] = new_hash
            if track_id:
                track_ids[youtube_id] = track_id
                _dup_cache_dirty = True

    # Persist to disk
    persistent_vectors = '/app/data/vectors.csv'
    local_vectors = os.path.join(SCRIPT_DIR, '..', 'vectors.csv')
    vectors_path = persistent_vectors if os.path.exists('/app/data') else local_vectors

    if is_new:
        emb_str = ' '.join(f'{x:.6f}' for x in embedding)
        with open(vectors_path, 'a') as f:
            f.write(f'{youtube_id},"[ {emb_str} ]"\n')
    elif force:
        # Rewrite vectors.csv to update the embedding on disk
        import tempfile
        emb_str = ' '.join(f'{x:.6f}' for x in embedding)
        tmp_path = vectors_path + '.tmp'
        with open(vectors_path, 'r') as fin, open(tmp_path, 'w') as fout:
            for line in fin:
                vid_in_line = line.split(',', 1)[0]
                if vid_in_line.endswith('.wav'):
                    vid_in_line = vid_in_line.rsplit('/', 1)[-1].replace('.wav', '')
                if vid_in_line == youtube_id:
                    fout.write(f'{youtube_id},"[ {emb_str} ]"\n')
                else:
                    fout.write(line)
        os.replace(tmp_path, vectors_path)

    # Save track_id mapping (atomic write to prevent corruption on restart)
    if track_id:
        track_ids_path = '/app/data/track_ids.json' if os.path.exists('/app/data') else os.path.join(SCRIPT_DIR, 'track_ids.json')
        existing = {}
        if os.path.exists(track_ids_path):
            with open(track_ids_path, 'r') as f:
                existing = json.load(f)
        existing[youtube_id] = track_id
        tmp_path = track_ids_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(existing, f)
        os.replace(tmp_path, track_ids_path)

    # Persist LIVI embedding (if provided)
    if livi_embedding is not None:
        livi_emb_norm = livi_embedding / (np.linalg.norm(livi_embedding) + 1e-8)
        livi_emb_norm = livi_emb_norm.astype(np.float32)

        with db_lock:
            existing_livi_idx = livi_vid_to_idx.get(youtube_id)
            if existing_livi_idx is not None:
                livi_embeddings_matrix[existing_livi_idx] = livi_emb_norm
            else:
                livi_video_ids.append(youtube_id)
                livi_vid_to_idx[youtube_id] = len(livi_video_ids) - 1
                if livi_embeddings_matrix is not None:
                    livi_embeddings_matrix = np.vstack([livi_embeddings_matrix, livi_emb_norm.reshape(1, -1)])
                else:
                    livi_embeddings_matrix = livi_emb_norm.reshape(1, -1)

        # Append to vectors_livi.csv
        persistent_livi = '/app/data/vectors_livi.csv'
        local_livi = os.path.join(SCRIPT_DIR, '..', 'vectors_livi.csv')
        livi_path = persistent_livi if os.path.exists('/app/data') else local_livi
        write_header = not os.path.exists(livi_path) or os.path.getsize(livi_path) == 0
        livi_str = ' '.join(f'{x:.8f}' for x in livi_emb_norm)
        with open(livi_path, 'a') as f:
            if write_header:
                f.write('youtube_id,embeddings\n')
            f.write(f'{youtube_id},"[ {livi_str} ]"\n')

    # Recompute Precision@1 every 100 songs (takes ~15s per computation)
    if db_stats['total_songs'] % 100 == 0:
        compute_precision_at_1()

    logging.info(f"Database updated: {db_stats['total_songs']} songs, P@1={db_stats.get('precision_at_1', '?')}")

    # Check if we crossed a 10-boundary for data.json regeneration
    check_data_regen()


def async_add_embedding(youtube_id, embedding, track_id=None, force=False, livi_embedding=None):
    """Trigger async database update in background thread."""
    thread = threading.Thread(
        target=add_embedding_to_database,
        args=(youtube_id, embedding, track_id, force, livi_embedding),
        daemon=True
    )
    thread.start()


def find_similar(query_embedding, top_k=5, query_livi_embedding=None):
    """Find top-k most similar songs to query embedding (with optional LIVI fusion)."""
    query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    query = query.astype(np.float32)

    ch_sims = embeddings_matrix @ query  # (N,)

    # Score fusion with LIVI
    if (query_livi_embedding is not None
            and livi_embeddings_matrix is not None
            and len(livi_video_ids) > 0):
        q_livi = query_livi_embedding / (np.linalg.norm(query_livi_embedding) + 1e-8)
        q_livi = q_livi.astype(np.float32)
        similarities = np.empty_like(ch_sims)
        # Batch compute LIVI sims for songs that have LIVI embeddings
        livi_sims = livi_embeddings_matrix @ q_livi  # (M,)
        for i, vid in enumerate(video_ids):
            li = livi_vid_to_idx.get(vid)
            if li is not None:
                similarities[i] = FUSION_WEIGHT_CH * ch_sims[i] + FUSION_WEIGHT_LIVI * livi_sims[li]
            else:
                similarities[i] = FUSION_WEIGHT_CH * ch_sims[i]
    else:
        similarities = ch_sims

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        vid = video_ids[idx]
        sim = float(similarities[idx])

        info = get_youtube_info(vid)
        title = info.get('title', '') if info else ''
        channel = info.get('channel', '') if info else ''

        yt_artist = channel
        if yt_artist.endswith(' - Topic'):
            yt_artist = yt_artist[:-8]

        itunes_artist, itunes_track, preview_url = '', '', ''
        stored_track_id = track_ids.get(vid)
        if stored_track_id:
            itunes_data = lookup_itunes_by_id(stored_track_id)
            if itunes_data:
                itunes_artist = itunes_data.get('artistName', '')
                itunes_track = itunes_data.get('trackName', '')
                preview_url = itunes_data.get('previewUrl', '')
        if not itunes_artist and title:
            parsed_artist, parsed_track = parse_artist_track(title)
            search_term = f"{parsed_artist} {parsed_track}" if parsed_artist else title
            itunes_results = search_itunes_detailed(search_term, limit=1)
            if itunes_results:
                itunes_artist = itunes_results[0].get('artistName', '')
                itunes_track = itunes_results[0].get('trackName', '')
                preview_url = itunes_results[0].get('previewUrl', '')

        _, parsed_track = parse_artist_track(title) if title else (None, None)

        result = {
            'youtube_id': vid,
            'similarity': round(sim, 4),
            'youtube_url': f'https://www.youtube.com/watch?v={vid}',
            'title': title or vid,
            'yt_channel': yt_artist,
            'artist': itunes_artist or yt_artist or '',
            'itunes_artist': itunes_artist,
            'itunes_track': itunes_track,
            'track': itunes_track or parsed_track or title or vid,
            'clique_id': clique_map.get(vid, ''),
            'hash': embedding_hashes.get(vid, ''),
            'track_id': track_ids.get(vid),
            'preview_url': preview_url,
        }
        results.append(result)

    track_id_counts = {}
    for r in results:
        tid = r.get('track_id')
        if tid:
            track_id_counts[tid] = track_id_counts.get(tid, 0) + 1

    for r in results:
        tid = r.get('track_id')
        if tid and track_id_counts.get(tid, 0) > 1:
            r['same_preview'] = True

    return results


def extract_youtube_id(url_or_id):
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return None


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for similar songs given a YouTube URL."""
    data = request.get_json() or {}
    url = data.get('url', '').strip()
    top_k = min(int(data.get('top_k', 20)), 100)

    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400

    youtube_id = extract_youtube_id(url)
    if not youtube_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    yt_metadata = get_youtube_metadata(youtube_id)
    if not yt_metadata:
        yt_metadata = get_youtube_info(youtube_id)  # oEmbed fallback
    title = yt_metadata.get('title', '') if yt_metadata else ''
    yt_channel = yt_metadata.get('channel', '') if yt_metadata else ''
    if yt_channel.endswith(' - Topic'):
        yt_channel = yt_channel[:-8]

    force = data.get('force', False)

    # Check if video is already in database
    query_livi_emb = None  # LIVI embedding for fusion at search time
    if youtube_id in video_ids and not force:
        idx = video_ids.index(youtube_id)
        embedding = embeddings_matrix[idx]
        query_track_id = track_ids.get(youtube_id)
        query_hash = embedding_hashes.get(youtube_id, '')
        in_database = True
        # Look up cached LIVI embedding
        li = livi_vid_to_idx.get(youtube_id)
        if li is not None and livi_embeddings_matrix is not None:
            query_livi_emb = livi_embeddings_matrix[li]
    else:
        # Compute embedding on-the-fly via iTunes preview
        raw_channel = yt_metadata.get('channel', '') if yt_metadata else ''
        embedding, livi_emb, query_track_id, error_msg = compute_embedding_for_video(youtube_id, title, channel=raw_channel, yt_metadata=yt_metadata)
        if embedding is None:
            # If force re-crawl failed, remove the old (bad) entry
            if force and youtube_id in video_ids:
                remove_from_database(youtube_id)
            status_code = 429 if error_msg == 'iTunes rate limited' else 404
            return jsonify({
                'error': f'Failed to compute embedding: {error_msg}',
                'title': title,
                'youtube_id': youtube_id,
                'database_size': len(video_ids),
            }), status_code
        query_hash = compute_embedding_hash(embedding)
        query_livi_emb = livi_emb
        in_database = False

        # Async add to database for future searches
        async_add_embedding(youtube_id, embedding.copy(), query_track_id, force=force,
                            livi_embedding=livi_emb.copy() if livi_emb is not None else None)

        # Track non-Discogs searches
        if youtube_id not in clique_map:
            _log_user_search(youtube_id, title)

    # Get iTunes info for display — use stored track_id first, fall back to search
    itunes_artist, itunes_track, preview_url = '', '', ''
    if query_track_id:
        itunes_data = lookup_itunes_by_id(query_track_id)
        if itunes_data:
            itunes_artist = itunes_data.get('artistName', '')
            itunes_track = itunes_data.get('trackName', '')
            preview_url = itunes_data.get('previewUrl', '')
    if not itunes_artist and title:
        parsed_artist, parsed_track = parse_artist_track(title)
        search_term = f"{parsed_artist} {parsed_track}" if parsed_artist else title
        itunes_results = search_itunes_detailed(search_term, limit=1)
        if itunes_results:
            itunes_artist = itunes_results[0].get('artistName', '')
            itunes_track = itunes_results[0].get('trackName', '')
            preview_url = itunes_results[0].get('previewUrl', '')

    # Find similar songs
    similar = find_similar(embedding, top_k + 5, query_livi_embedding=query_livi_emb)

    # Filter out query video
    similar = [r for r in similar if r['youtube_id'] != youtube_id][:top_k]

    # Check for same preview warning
    query_matches_result = any(
        r.get('track_id') == query_track_id
        for r in similar
        if query_track_id and r.get('track_id')
    )

    return jsonify({
        'query': {
            'youtube_id': youtube_id,
            'youtube_url': f'https://www.youtube.com/watch?v={youtube_id}',
            'title': title,
            'yt_channel': yt_channel,
            'artist': itunes_artist,
            'track': itunes_track,
            'track_id': query_track_id,
            'hash': query_hash,
            'preview_url': preview_url,
            'clique_id': clique_map.get(youtube_id, ''),
            'same_preview_in_db': query_matches_result,
            'in_database': in_database,
        },
        'results': similar,
        'stats': db_stats,
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'embeddings_count': len(video_ids),
        'device': str(device),
        'stats': db_stats,
    })


@app.route('/api/crawl-progress', methods=['POST'])
def crawl_progress():
    """Receive crawl progress from crawl_songs.py."""
    data = request.get_json(force=True)
    data['updated_at'] = time.time()
    db_stats['crawl_progress'] = data
    return jsonify({'status': 'ok'})


@app.route('/api/reload', methods=['POST'])
def reload_database():
    """Reload the embeddings database."""
    try:
        load_database()
        load_livi_vectors()
        load_clique_map()
        check_data_regen()
        # Recompute P@1 in background so reload returns immediately
        threading.Thread(target=compute_precision_at_1, daemon=True).start()
        return jsonify({
            'status': 'ok',
            'embeddings_count': len(video_ids),
            'stats': db_stats,
            'regen_running': regen_running,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/resolve-trackid', methods=['POST'])
def resolve_trackid():
    """Look up iTunes track_id for a video already in the database, without recomputing embedding."""
    data = request.get_json() or {}
    youtube_id = data.get('youtube_id', '').strip()

    if not youtube_id:
        return jsonify({'error': 'Missing youtube_id'}), 400

    if youtube_id not in video_ids:
        return jsonify({'error': 'Not in database'}), 404

    # Already has a track_id
    if youtube_id in track_ids:
        return jsonify({'status': 'ok', 'youtube_id': youtube_id, 'track_id': track_ids[youtube_id], 'cached': True})

    # Look up YouTube metadata
    yt_metadata = get_youtube_metadata(youtube_id)
    if not yt_metadata:
        yt_metadata = get_youtube_info(youtube_id)
    title = yt_metadata.get('title', '') if yt_metadata else ''
    channel = yt_metadata.get('channel', '') if yt_metadata else ''

    if not title:
        return jsonify({'error': 'Failed to get YouTube title', 'youtube_id': youtube_id}), 404

    # Search iTunes (no download, no embedding)
    try:
        track_info, track_id = smart_search_itunes(youtube_id, title, channel=channel, yt_metadata=yt_metadata)
    except ITunesRateLimitError:
        return jsonify({'error': 'iTunes rate limited'}), 429

    if not track_id:
        return jsonify({'error': 'No iTunes match found', 'youtube_id': youtube_id}), 404

    # Save the track_id
    global _dup_cache_dirty
    track_ids[youtube_id] = track_id
    _dup_cache_dirty = True

    # Persist atomically
    track_ids_path = '/app/data/track_ids.json' if os.path.exists('/app/data') else os.path.join(SCRIPT_DIR, 'track_ids.json')
    existing = {}
    if os.path.exists(track_ids_path):
        with open(track_ids_path, 'r') as f:
            existing = json.load(f)
    existing[youtube_id] = track_id
    tmp_path = track_ids_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(existing, f)
    os.replace(tmp_path, track_ids_path)

    return jsonify({'status': 'ok', 'youtube_id': youtube_id, 'track_id': track_id})


@app.route('/api/cleanup-unverified', methods=['POST'])
def cleanup_unverified():
    """Remove all indexed tracks that have no iTunes match (no track_id)."""
    with db_lock:
        to_remove = [vid for vid in video_ids if vid not in track_ids]

    if not to_remove:
        return jsonify({'status': 'ok', 'removed': 0, 'total_remaining': len(video_ids),
                        'message': 'no unverified tracks to remove'})

    removed = batch_remove_from_database(to_remove)
    _recompute_stats()
    _run_data_regen()

    logging.info(f"Cleanup: removed {len(removed)} unverified tracks")
    return jsonify({
        'status': 'ok',
        'removed': len(removed),
        'total_remaining': len(video_ids),
    })


@app.route('/api/dedup', methods=['POST'])
def dedup_database():
    """Remove duplicate entries: by track_id and by embedding hash."""
    from collections import defaultdict

    # Snapshot the data under lock to avoid race conditions
    with db_lock:
        vids_snapshot = list(video_ids)
        tids_snapshot = dict(track_ids)
        hashes_snapshot = dict(embedding_hashes)

    to_remove_set = set()

    # Pass 1: dedup by track_id
    tid_groups = 0
    tid_to_vids = defaultdict(list)
    for vid in vids_snapshot:
        tid = tids_snapshot.get(vid)
        if tid:
            tid_to_vids[tid].append(vid)
    for tid, vids in tid_to_vids.items():
        if len(vids) > 1:
            tid_groups += 1
            to_remove_set.update(vids[1:])

    # Pass 2: dedup by embedding hash (catches identical audio with different track_ids)
    hash_groups = 0
    hash_to_vids = defaultdict(list)
    for vid in vids_snapshot:
        if vid in to_remove_set:
            continue  # already marked for removal
        h = hashes_snapshot.get(vid)
        if h:
            hash_to_vids[h].append(vid)
    for h, vids in hash_to_vids.items():
        if len(vids) > 1:
            hash_groups += 1
            to_remove_set.update(vids[1:])

    if not to_remove_set:
        return jsonify({'status': 'ok', 'removed': 0, 'total_remaining': len(video_ids),
                        'message': 'no duplicates to remove'})

    removed = batch_remove_from_database(list(to_remove_set))
    _recompute_stats()
    _run_data_regen()

    logging.info(f"Dedup: removed {len(removed)} entries ({tid_groups} track_id groups, {hash_groups} embedding hash groups)")
    return jsonify({
        'status': 'ok',
        'removed': len(removed),
        'tid_duplicate_groups': tid_groups,
        'hash_duplicate_groups': hash_groups,
        'total_remaining': len(video_ids),
    })


@app.route('/api/regen-status')
def regen_status():
    """Check if data.json regeneration is running."""
    return jsonify({
        'regen_running': regen_running,
        'last_regen_count': last_regen_count,
        'current_count': len(video_ids),
    })


# Search history
HISTORY_FILE = '/app/data/search_history.json' if os.path.isdir('/app/data') else os.path.join(SCRIPT_DIR, 'search_history.json')

def load_search_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_search_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[:10], f)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(load_search_history())

@app.route('/api/history', methods=['POST'])
def add_history():
    data = request.get_json() or {}
    history = load_search_history()
    history = [h for h in history if h.get('youtube_id') != data.get('youtube_id')]
    history.insert(0, {
        'youtube_id': data.get('youtube_id'),
        'artist': data.get('artist'),
        'track': data.get('track')
    })
    save_search_history(history[:10])
    return jsonify({'status': 'ok'})


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoverHunter Cover Detection API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    args = parser.parse_args()

    logging.info("Initializing CoverHunter Cover Detection API...")
    init_model()
    load_database()
    load_clique_map()
    last_regen_count = len(video_ids)
    logging.info(f"Initial song count for regen tracking: {last_regen_count} (next threshold: {(last_regen_count // REGEN_INTERVAL + 1) * REGEN_INTERVAL})")

    # Regenerate data.json at startup (in background, non-blocking)
    logging.info("Triggering data.json regeneration at startup...")
    _run_data_regen()

    # Load cached P@1 so stats are available immediately, then recompute in background
    _load_p1_cache()
    logging.info("Starting P@1 computation in background...")
    threading.Thread(target=compute_precision_at_1, daemon=True).start()

    # Start periodic stats recomputation
    threading.Thread(target=_stats_timer, daemon=True).start()
    logging.info(f"Stats recomputation timer started (every {STATS_INTERVAL}s)")

    logging.info(f"\nStarting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
