#!/usr/bin/env python3
"""
Compute Discogs-VINet embeddings for songs via iTunes previews.

Resumable pipeline:
  1. YouTube title lookup (oEmbed API)
  2. iTunes search for preview URL
  3. Download preview (m4a), convert to WAV
  4. Compute CQT, run model inference
  5. Append 512-dim embedding to output CSV
  6. Clean up temp files

Usage:
    python pipeline.py [--ids youtube_ids.txt] [--output vectors_vinet.csv]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import torch
import yaml

# Add parent dir to path so model/utilities imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import librosa
from model.utils import load_model


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 22050
HOP_SIZE = 512
CQT_BINS = 84
BINS_PER_OCTAVE = 12
CONTEXT_LENGTH = 7600
DOWNSAMPLE_FACTOR = 20

ITUNES_DELAY = 0.2  # seconds between iTunes API calls
OEMBED_TIMEOUT = 10  # seconds
ITUNES_TIMEOUT = 10
DOWNLOAD_TIMEOUT = 30


# ============================================================================
# YouTube title lookup
# ============================================================================

def get_youtube_title(youtube_id):
    """Get video title via YouTube oEmbed API (no auth needed)."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=OEMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('title', '')
    except Exception:
        return None


def get_youtube_info(youtube_id):
    """Get video title AND channel name via YouTube oEmbed API."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=OEMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return {
                'title': data.get('title', ''),
                'channel': data.get('author_name', ''),
            }
    except Exception:
        return None


# ============================================================================
# Parse artist/track from title
# ============================================================================

# Common suffixes to strip from titles
STRIP_PATTERNS = [
    r'\s*\(official\s*(music\s*)?video\)',
    r'\s*\[official\s*(music\s*)?video\]',
    r'\s*\(official\s*audio\)',
    r'\s*\[official\s*audio\]',
    r'\s*\(lyric\s*video\)',
    r'\s*\[lyric\s*video\]',
    r'\s*\(lyrics?\)',
    r'\s*\[lyrics?\]',
    r'\s*\(audio\)',
    r'\s*\[audio\]',
    r'\s*\(visualizer\)',
    r'\s*\[visualizer\]',
    r'\s*\(live\)',
    r'\s*\[live\]',
    r'\s*\(hd\)',
    r'\s*\[hd\]',
    r'\s*\(hq\)',
    r'\s*\[hq\]',
    r'\s*ft\.?\s+.*$',
    r'\s*feat\.?\s+.*$',
    r'\s*featuring\s+.*$',
]


def parse_artist_track(title):
    """Parse 'Artist - Track' from YouTube title. Returns (artist, track) or (None, cleaned_title)."""
    if not title:
        return None, None

    cleaned = title
    for pattern in STRIP_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Try splitting on " - "
    if ' - ' in cleaned:
        parts = cleaned.split(' - ', 1)
        artist = parts[0].strip()
        track = parts[1].strip()
        return artist, track

    # No clear separator - return whole title
    return None, cleaned


# ============================================================================
# iTunes search
# ============================================================================

def search_itunes(artist, track):
    """Search iTunes for a song preview URL. Returns (preview_url, matched_artist, matched_track) or (None, None, None)."""
    if artist and track:
        term = f"{artist} {track}"
    elif track:
        term = track
    else:
        return None, None, None

    params = urllib.parse.urlencode({
        'term': term,
        'media': 'music',
        'entity': 'song',
        'limit': '5',
    })
    url = f"https://itunes.apple.com/search?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=ITUNES_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        results = data.get('results', [])
        if not results:
            return None, None, None

        # Return first result with a previewUrl
        for result in results:
            preview_url = result.get('previewUrl')
            if preview_url:
                return (
                    preview_url,
                    result.get('artistName', ''),
                    result.get('trackName', ''),
                )

        return None, None, None

    except Exception:
        return None, None, None


def search_itunes_detailed(query, limit=5):
    """Search iTunes and return full results."""
    params = urllib.parse.urlencode({
        'term': query,
        'media': 'music',
        'entity': 'song',
        'limit': str(limit),
    })
    url = f"https://itunes.apple.com/search?{params}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=ITUNES_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('results', [])
    except:
        return []


def lookup_itunes_by_id(track_id):
    """Look up iTunes track by ID and return full metadata."""
    url = f"https://itunes.apple.com/lookup?id={track_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=ITUNES_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = data.get('results', [])
        return results[0] if results else None
    except:
        return None


def clean_channel_name(channel):
    """Clean YouTube channel name for use as artist."""
    if not channel:
        return None
    # Remove common suffixes
    cleaned = re.sub(r'\s*-\s*Topic$', '', channel, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*VEVO$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*Official$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip() if cleaned.strip() else None


def score_itunes_match(result, youtube_title, youtube_channel):
    """Score how well an iTunes result matches the YouTube video."""
    itunes_artist = result.get('artistName', '').lower()
    itunes_track = result.get('trackName', '').lower()
    yt_title = youtube_title.lower() if youtube_title else ''
    yt_channel = youtube_channel.lower() if youtube_channel else ''

    score = 0

    # Channel name matches iTunes artist (strong signal)
    if yt_channel and itunes_artist:
        if itunes_artist in yt_channel or yt_channel in itunes_artist:
            score += 50

    # iTunes track appears in YouTube title
    if itunes_track and itunes_track in yt_title:
        score += 30

    # iTunes artist appears in YouTube title
    if itunes_artist and itunes_artist in yt_title:
        score += 20

    return score


def smart_search_itunes(youtube_id):
    """
    Smart iTunes search using multiple strategies.
    Returns (preview_url, artist, track, track_id) or (None, None, None, None).
    """
    info = get_youtube_info(youtube_id)
    if not info:
        return None, None, None, None

    title = info['title']
    channel = info['channel']
    clean_channel = clean_channel_name(channel)

    # Parse artist/track from title
    artist, track = parse_artist_track(title)

    # Build search strategies (ordered by preference)
    strategies = []

    # Strategy 1: Channel name as artist + parsed track
    if clean_channel and track:
        strategies.append(f'{clean_channel} {track}')

    # Strategy 2: Parsed artist + track from title
    if artist and track:
        strategies.append(f'{artist} {track}')

    # Strategy 3: Channel name + cleaned title
    if clean_channel and title:
        cleaned_title = title
        for pattern in STRIP_PATTERNS:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)
        strategies.append(f'{clean_channel} {cleaned_title.strip()}')

    # Strategy 4: Just the track (fallback)
    if track:
        strategies.append(track)

    # Try each strategy and pick best match
    best_result = None
    best_score = -1

    for query in strategies:
        time.sleep(ITUNES_DELAY)
        results = search_itunes_detailed(query, limit=5)

        for result in results:
            if not result.get('previewUrl'):
                continue
            score = score_itunes_match(result, title, channel)
            if score > best_score:
                best_score = score
                best_result = result

    if best_result:
        return (
            best_result.get('previewUrl'),
            best_result.get('artistName', ''),
            best_result.get('trackName', ''),
            best_result.get('trackId'),
        )

    return None, None, None, None


# ============================================================================
# Download & convert
# ============================================================================

def download_preview(preview_url, output_path):
    """Download iTunes preview to a file."""
    try:
        req = urllib.request.Request(preview_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            with open(output_path, 'wb') as f:
                f.write(resp.read())
        return True
    except Exception:
        return False


def convert_to_wav(input_path, output_path):
    """Convert audio file to mono WAV at 22050 Hz using ffmpeg."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-f', 'wav', output_path],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================================
# CQT preprocessing (matches Discogs-VINet inference pipeline)
# ============================================================================

def mean_downsample_cqt(cqt, factor):
    """Downsample CQT by averaging groups of `factor` frames."""
    T, F = cqt.shape
    new_T = T // factor
    new_cqt = np.zeros((new_T, F), dtype=cqt.dtype)
    for i in range(new_T):
        new_cqt[i, :] = cqt[i * factor:(i + 1) * factor, :].mean(axis=0)
    return new_cqt


def compute_cqt_features(wav_path):
    """Load audio and compute CQT features matching Discogs-VINet preprocessing."""
    # Load audio using librosa (replacement for essentia.MonoLoader)
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) == 0:
        return None

    # Compute CQT
    cqt = librosa.cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_bins=CQT_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )  # (F, T) complex

    if cqt.size == 0:
        return None

    # Transpose to (T, F) to match Discogs-VINet convention
    cqt = cqt.T

    # Magnitude
    cqt = np.abs(cqt)

    # float16 -> float32 round-trip for consistency with training extraction
    cqt = cqt.astype(np.float16).astype(np.float32)

    if np.isnan(cqt).any() or np.isinf(cqt).any():
        return None

    # Pad if too short
    if cqt.shape[0] < CONTEXT_LENGTH:
        cqt = np.pad(
            cqt,
            ((0, CONTEXT_LENGTH - cqt.shape[0]), (0, 0)),
            'constant',
            constant_values=0,
        )

    # Downsample
    if DOWNSAMPLE_FACTOR > 1:
        cqt = mean_downsample_cqt(cqt, DOWNSAMPLE_FACTOR)

    # Clip below zero
    cqt = np.where(cqt < 0, 0, cqt)

    # Scale to [0, 1]
    max_val = np.max(cqt)
    if max_val > 0:
        cqt = cqt / (max_val + 1e-6)

    # Back to (F, T) for the model
    cqt = cqt.T

    return cqt


# ============================================================================
# Model inference
# ============================================================================

@torch.inference_mode()
def compute_embedding(model, cqt_features, device):
    """Run model inference on CQT features. Returns 512-dim numpy array."""
    # cqt_features is (F, T), model expects (batch, 1, F, T)
    cqt_tensor = torch.as_tensor(cqt_features, dtype=torch.float32)
    cqt_tensor = cqt_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
    cqt_tensor = cqt_tensor.to(device)

    embedding = model(cqt_tensor)  # (1, 512)
    return embedding.cpu().numpy()[0]


# ============================================================================
# Progress tracking
# ============================================================================

def load_progress(progress_path):
    """Load progress from JSON file."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {
        'processed': [],
        'failed': {},
        'stats': {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'oembed_fail': 0,
            'itunes_fail': 0,
            'download_fail': 0,
            'inference_fail': 0,
        }
    }


def save_progress(progress, progress_path):
    """Save progress to JSON file."""
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Precision@1 evaluation
# ============================================================================

def _compute_p_at_1(embeddings, id_to_idx, idx_to_id, eval_vids, cover_map, same_preview_pairs=None):
    """Compute P@1 given an embedding matrix and evaluable songs.

    If same_preview_pairs is provided, cover pairs where both songs used the
    same iTunes preview are excluded (they'd have identical embeddings).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    if same_preview_pairs is None:
        same_preview_pairs = set()

    hits = 0
    total = 0
    for vid in eval_vids:
        query_idx = id_to_idx[vid]
        # Filter out covers that share the same preview URL
        cover_idxs = set()
        for c in cover_map[vid]:
            if c not in id_to_idx:
                continue
            pair_key = (min(vid, c), max(vid, c))
            if pair_key in same_preview_pairs:
                continue
            cover_idxs.add(id_to_idx[c])
        if not cover_idxs:
            continue
        total += 1
        sims = normed[query_idx] @ normed.T
        sims[query_idx] = -2.0
        nn_idx = int(np.argmax(sims))
        if nn_idx in cover_idxs:
            hits += 1
    return hits, total


def evaluate_precision_at_1(output_csv, cover_map_path, baseline_path=None, preview_urls=None):
    """Compute Precision@1 for Discogs-VINet AND CoverHunter baseline on the
    exact same set of evaluable songs, so the comparison is fair.

    Excludes cover pairs where both songs used the same iTunes preview URL
    (which would produce identical embeddings and inflate the metric).
    """
    if not os.path.exists(cover_map_path):
        print('  [P@1] cover_map.json not found, skipping evaluation')
        return None, None

    with open(cover_map_path) as f:
        cover_map = json.load(f)

    # If we have preview URLs, build a "tainted" set of cover pairs
    # where both songs got the same preview (identical audio)
    same_preview_pairs = set()
    if preview_urls:
        for vid, covers in cover_map.items():
            if vid not in preview_urls:
                continue
            for c in covers:
                if c not in preview_urls:
                    continue
                if preview_urls[vid] == preview_urls[c]:
                    same_preview_pairs.add((min(vid, c), max(vid, c)))
        if same_preview_pairs:
            print(f'  [P@1] Excluding {len(same_preview_pairs)} cover pairs with same iTunes preview')

    # Load Discogs-VINet embeddings from CSV
    vinet_ids = []
    vinet_vecs = []
    with open(output_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            vid = row[0].strip()
            emb_str = row[1].strip().strip('"').strip('[]')
            try:
                vec = [float(x) for x in emb_str.split()]
            except ValueError:
                continue
            vinet_ids.append(vid)
            vinet_vecs.append(vec)

    if len(vinet_ids) < 2:
        return None, None

    vinet_id_set = set(vinet_ids)

    # Load CoverHunter baseline
    ch_id_to_vec = {}
    if baseline_path and os.path.exists(baseline_path):
        data = np.load(baseline_path, allow_pickle=True)
        bl_ids = data['ids']
        bl_vecs = data['vecs']
        for bid, bvec in zip(bl_ids, bl_vecs):
            ch_id_to_vec[str(bid)] = bvec

    # Build the SAME song set for both: songs present in VINet embeddings
    # that also have a CoverHunter embedding (if baseline available)
    if ch_id_to_vec:
        common_ids = vinet_id_set & set(ch_id_to_vec.keys())
    else:
        common_ids = vinet_id_set

    # Build evaluation set: songs in common_ids that have covers also in common_ids
    # (excluding pairs with same preview URL)
    eval_vids = []
    for vid in common_ids:
        if vid not in cover_map:
            continue
        has_valid_cover = False
        for c in cover_map[vid]:
            if c not in common_ids:
                continue
            pair_key = (min(vid, c), max(vid, c))
            if pair_key not in same_preview_pairs:
                has_valid_cover = True
                break
        if has_valid_cover:
            eval_vids.append(vid)

    if not eval_vids:
        print(f'  [P@1] No evaluable pairs yet (0 songs with non-duplicate covers)')
        return None, None

    # --- Discogs-VINet P@1 (on the common set only) ---
    common_vinet_ids = [vid for vid in vinet_ids if vid in common_ids]
    common_vinet_vecs = np.array(
        [vinet_vecs[i] for i, vid in enumerate(vinet_ids) if vid in common_ids],
        dtype=np.float32
    )
    vinet_id_to_idx = {vid: i for i, vid in enumerate(common_vinet_ids)}
    vinet_idx_to_id = {i: vid for vid, i in vinet_id_to_idx.items()}

    vinet_hits, vinet_total = _compute_p_at_1(
        common_vinet_vecs, vinet_id_to_idx, vinet_idx_to_id,
        eval_vids, cover_map, same_preview_pairs)
    vinet_p1 = vinet_hits / vinet_total if vinet_total else 0

    # --- CoverHunter baseline P@1 (same songs) ---
    ch_p1 = None
    if ch_id_to_vec:
        ch_ordered_ids = [vid for vid in common_vinet_ids]
        ch_ordered_vecs = np.array(
            [ch_id_to_vec[vid] for vid in ch_ordered_ids], dtype=np.float32)
        ch_id_to_idx = {vid: i for i, vid in enumerate(ch_ordered_ids)}
        ch_idx_to_id = {i: vid for vid, i in ch_id_to_idx.items()}

        ch_hits, ch_total = _compute_p_at_1(
            ch_ordered_vecs, ch_id_to_idx, ch_idx_to_id,
            eval_vids, cover_map, same_preview_pairs)
        ch_p1 = ch_hits / ch_total if ch_total else 0

        print(f'  [P@1] Discogs-VINet: {vinet_hits}/{vinet_total} = {vinet_p1:.3f}  |  '
              f'CoverHunter: {ch_hits}/{ch_total} = {ch_p1:.3f}  '
              f'({len(common_vinet_ids)} common songs, {vinet_total} evaluable, '
              f'{len(same_preview_pairs)} same-preview pairs excluded)')
    else:
        print(f'  [P@1] Discogs-VINet: {vinet_hits}/{vinet_total} = {vinet_p1:.3f} '
              f'({len(common_vinet_ids)} songs, {vinet_total} evaluable, no baseline)')

    return vinet_p1, ch_p1


# ============================================================================
# Format embedding for CSV output
# ============================================================================

def format_embedding(embedding):
    """Format embedding as space-separated string in brackets, matching CoverHunter format."""
    values = ' '.join(f'{v: .8f}' for v in embedding)
    return f'"[ {values} ]"'


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute Discogs-VINet embeddings via iTunes previews')
    parser.add_argument('--ids', default=os.path.join(SCRIPT_DIR, 'youtube_ids.txt'),
                        help='Path to file with YouTube IDs, one per line')
    parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'vectors_vinet.csv'),
                        help='Output CSV path')
    parser.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'checkpoint', 'config.yaml'),
                        help='Model config YAML path')
    parser.add_argument('--progress', default=os.path.join(SCRIPT_DIR, 'progress.json'),
                        help='Progress tracking JSON path')
    parser.add_argument('--tmp-dir', default=os.path.join(SCRIPT_DIR, 'tmp'),
                        help='Temporary directory for audio files')
    parser.add_argument('--cover-map', default=os.path.join(SCRIPT_DIR, 'cover_map.json'),
                        help='Path to cover_map.json for P@1 evaluation')
    parser.add_argument('--baseline', default=os.path.join(SCRIPT_DIR, 'coverhunter_baseline.npz'),
                        help='Path to CoverHunter baseline embeddings (.npz) for comparison')
    parser.add_argument('--eval-interval', type=int, default=2000,
                        help='Evaluate P@1 every N songs')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fix checkpoint path to be relative to script dir
    if not os.path.isabs(config['MODEL']['CHECKPOINT_PATH']):
        config['MODEL']['CHECKPOINT_PATH'] = os.path.join(
            SCRIPT_DIR, config['MODEL']['CHECKPOINT_PATH']
        )

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # Load model
    print('Loading model...')
    model = load_model(config, device)
    print('Model loaded.')

    # Load YouTube IDs
    with open(args.ids) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    print(f'Total YouTube IDs: {len(all_ids)}')

    # Load progress
    progress = load_progress(args.progress)
    processed_set = set(progress['processed'])
    failed_set = set(progress['failed'].keys())

    # Filter to unprocessed IDs
    remaining_ids = [vid for vid in all_ids if vid not in processed_set and vid not in failed_set]
    print(f'Already processed: {len(processed_set)}')
    print(f'Previously failed: {len(failed_set)}')
    print(f'Remaining: {len(remaining_ids)}')

    if not remaining_ids:
        print('Nothing to process. Done!')
        return

    progress['stats']['total'] = len(all_ids)

    # Track when we last ran P@1 evaluation
    last_eval_count = (len(processed_set) // args.eval_interval) * args.eval_interval

    # Run P@1 on resume if we already have embeddings
    if len(processed_set) >= args.eval_interval:
        print(f'Running P@1 evaluation on {len(processed_set)} existing embeddings...')
        vinet_p1, ch_p1 = evaluate_precision_at_1(
            args.output, args.cover_map, args.baseline,
            preview_urls=progress.get('preview_urls'))
        if vinet_p1 is not None:
            progress['stats'].setdefault('precision_at_1_history', [])
            entry = {
                'n_embeddings': len(processed_set),
                'vinet_p_at_1': round(vinet_p1, 4),
            }
            if ch_p1 is not None:
                entry['coverhunter_p_at_1'] = round(ch_p1, 4)
            progress['stats']['precision_at_1_history'].append(entry)
            save_progress(progress, args.progress)

    # Create temp dir
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Ensure output CSV exists with header
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    if write_header:
        with open(args.output, 'w') as f:
            f.write('youtube_id,embeddings\n')

    # Process each song
    for i, youtube_id in enumerate(remaining_ids):
        m4a_path = os.path.join(args.tmp_dir, f'{youtube_id}.m4a')
        wav_path = os.path.join(args.tmp_dir, f'{youtube_id}.wav')

        try:
            # Step 1-3: Smart iTunes search (uses channel name as artist, multiple strategies)
            preview_url, matched_artist, matched_track, track_id = smart_search_itunes(youtube_id)
            if not preview_url:
                # Check if it was an oEmbed failure vs iTunes failure
                info = get_youtube_info(youtube_id)
                if not info:
                    progress['failed'][youtube_id] = 'oembed_fail'
                    progress['stats']['oembed_fail'] += 1
                else:
                    progress['failed'][youtube_id] = 'itunes_fail'
                    progress['stats']['itunes_fail'] += 1
                progress['stats']['failed'] += 1
                if (i + 1) % 10 == 0:
                    save_progress(progress, args.progress)
                continue

            # Track the iTunes track ID for dedup analysis
            progress.setdefault('track_ids', {})[youtube_id] = track_id

            # Step 4: Download preview
            if not download_preview(preview_url, m4a_path):
                progress['failed'][youtube_id] = 'download_fail'
                progress['stats']['download_fail'] += 1
                progress['stats']['failed'] += 1
                if (i + 1) % 10 == 0:
                    save_progress(progress, args.progress)
                continue

            # Step 5: Convert to WAV
            if not convert_to_wav(m4a_path, wav_path):
                progress['failed'][youtube_id] = 'download_fail'
                progress['stats']['download_fail'] += 1
                progress['stats']['failed'] += 1
                if (i + 1) % 10 == 0:
                    save_progress(progress, args.progress)
                continue

            # Step 6: Compute CQT
            cqt = compute_cqt_features(wav_path)
            if cqt is None:
                progress['failed'][youtube_id] = 'inference_fail'
                progress['stats']['inference_fail'] += 1
                progress['stats']['failed'] += 1
                if (i + 1) % 10 == 0:
                    save_progress(progress, args.progress)
                continue

            # Step 7: Model inference
            embedding = compute_embedding(model, cqt, device)

            # Step 8: Append to output CSV
            emb_str = format_embedding(embedding)
            with open(args.output, 'a') as f:
                f.write(f'{youtube_id},{emb_str}\n')

            # Log preview URL for dedup analysis
            progress.setdefault('preview_urls', {})[youtube_id] = preview_url

            # Step 9: Update progress
            progress['processed'].append(youtube_id)
            progress['stats']['processed'] += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(remaining_ids):
                pct = (len(progress['processed']) / len(all_ids)) * 100
                print(f'[{i+1}/{len(remaining_ids)}] Processed: {len(progress["processed"])}, '
                      f'Failed: {progress["stats"]["failed"]}, '
                      f'Progress: {pct:.1f}%')

            # Periodic P@1 evaluation
            current_count = len(progress['processed'])
            if current_count >= last_eval_count + args.eval_interval:
                print(f'\n--- P@1 evaluation at {current_count} embeddings ---')
                vinet_p1, ch_p1 = evaluate_precision_at_1(
                    args.output, args.cover_map, args.baseline,
                    preview_urls=progress.get('preview_urls'))
                if vinet_p1 is not None:
                    progress['stats'].setdefault('precision_at_1_history', [])
                    entry = {
                        'n_embeddings': current_count,
                        'vinet_p_at_1': round(vinet_p1, 4),
                    }
                    if ch_p1 is not None:
                        entry['coverhunter_p_at_1'] = round(ch_p1, 4)
                    progress['stats']['precision_at_1_history'].append(entry)
                    save_progress(progress, args.progress)
                last_eval_count = (current_count // args.eval_interval) * args.eval_interval
                print('--- resuming pipeline ---\n')

        except Exception as e:
            progress['failed'][youtube_id] = f'error: {str(e)[:200]}'
            progress['stats']['failed'] += 1

        finally:
            # Step 10: Clean up temp files
            for path in [m4a_path, wav_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            # Save progress periodically
            if (i + 1) % 10 == 0 or (i + 1) == len(remaining_ids):
                save_progress(progress, args.progress)

    # Final save
    save_progress(progress, args.progress)
    print(f'\nDone! Processed: {progress["stats"]["processed"]}, '
          f'Failed: {progress["stats"]["failed"]}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
