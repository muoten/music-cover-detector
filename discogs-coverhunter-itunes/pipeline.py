#!/usr/bin/env python3
"""
Pipeline for computing CoverHunter embeddings from iTunes 30-second previews.

This module provides functions for:
1. YouTube title lookup via oEmbed API
2. iTunes search and preview URL retrieval
3. Audio download and conversion
4. CQT feature extraction (matching CoverHunter training)
5. CoverHunter embedding computation

Usage:
    python pipeline.py [--checkpoint checkpoint/] [--output vectors_coverhunter.csv]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

import numpy as np

# Add parent dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ============================================================================
# YouTube API helpers
# ============================================================================

def get_youtube_info(video_id):
    """Get YouTube video title and channel via oEmbed API."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {
                'title': data.get('title', ''),
                'channel': data.get('author_name', ''),
            }
    except Exception as e:
        return None


def get_youtube_metadata(video_id):
    """Get rich YouTube metadata via yt-dlp without downloading."""
    try:
        import yt_dlp
    except ImportError:
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'skip_download': True, 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', ''),
                'channel': info.get('channel', '') or info.get('uploader', ''),
                'artist': info.get('artist', ''),
                'track': info.get('track', ''),
            }
    except Exception:
        return None


def get_youtube_title(video_id):
    """Get YouTube video title via oEmbed API."""
    info = get_youtube_info(video_id)
    return info.get('title', '') if info else ''


def parse_artist_track(title):
    """Parse artist and track from YouTube title."""
    if not title:
        return None, None

    # Common separators
    for sep in [' - ', ' – ', ' — ', ' | ']:
        if sep in title:
            parts = title.split(sep, 1)
            artist = parts[0].strip()
            track = parts[1].strip()
            # Remove common suffixes
            for suffix in ['(Official Video)', '(Official Audio)', '(Lyrics)',
                          '[Official Video]', '[Official Audio]', '[Lyrics]',
                          '(Official Music Video)', '[Official Music Video]',
                          '(Audio)', '[Audio]', '(Video)', '[Video]',
                          '(Lyric Video)', '[Lyric Video]']:
                track = track.replace(suffix, '').strip()
            return artist, track

    return None, title


# ============================================================================
# iTunes API helpers
# ============================================================================

def search_itunes(term, limit=3):
    """Search iTunes for tracks."""
    encoded = urllib.parse.quote(term)
    url = f"https://itunes.apple.com/search?term={encoded}&media=music&entity=song&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])
            # Return list of dicts with previewUrl, trackId, etc.
            return results
    except Exception as e:
        return []


def search_itunes_detailed(term, limit=3):
    """Search iTunes and return full result details."""
    return search_itunes(term, limit)


def lookup_itunes_by_id(track_id):
    """Lookup iTunes track by ID."""
    url = f"https://itunes.apple.com/lookup?id={track_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])
            return results[0] if results else None
    except Exception as e:
        return None


def smart_search_itunes(youtube_id, title=None, channel=None, yt_metadata=None):
    """
    Smart iTunes search with fallback strategies.

    Returns: (track_info, track_id) or (None, None)
    """
    if not title:
        title = get_youtube_title(youtube_id)
    if not title:
        return None, None

    artist, track = parse_artist_track(title)

    # Strategy YT: Use structured artist+track from yt-dlp metadata (highest confidence)
    if yt_metadata:
        yt_artist = yt_metadata.get('artist', '')
        yt_track = yt_metadata.get('track', '')
        if yt_artist and yt_track:
            results = search_itunes(f"{yt_artist} {yt_track}", limit=3)
            if results:
                for r in results:
                    r_artist = r.get('artistName', '').lower()
                    if (yt_artist.lower() in r_artist or r_artist in yt_artist.lower()) and r.get('previewUrl'):
                        return r, r.get('trackId')

    # Extract artist from YouTube channel name
    channel_artist = None
    if channel:
        channel_artist = channel
        if channel_artist.endswith(' - Topic'):
            channel_artist = channel_artist[:-8]

    # Strategy 0: Use channel artist + track for artist-specific match
    if channel_artist:
        search_term = f"{channel_artist} {track}" if track else f"{channel_artist} {title}"
        results = search_itunes(search_term, limit=3)
        if results:
            # Require artist match between channel and iTunes result
            for r in results:
                r_artist = r.get('artistName', '').lower()
                if (channel_artist.lower() in r_artist or r_artist in channel_artist.lower()) and r.get('previewUrl'):
                    return r, r.get('trackId')

    # Best known artist (from yt-dlp or channel) for final mismatch rejection
    known_artist = (yt_metadata.get('artist', '') if yt_metadata else '') or channel_artist

    def _artist_matches(known, itunes_artist):
        """Check if known artist matches iTunes artist (substring in either direction)."""
        if not known:
            return True  # no known artist, can't reject
        k = known.lower()
        i = itunes_artist.lower()
        return k in i or i in k

    # Strategy 1: Search with artist + track
    if artist and track:
        results = search_itunes(f"{artist} {track}", limit=3)
        if results:
            # Try to find exact match
            for r in results:
                r_artist = r.get('artistName', '').lower()
                r_track = r.get('trackName', '').lower()
                if (artist.lower() in r_artist or r_artist in artist.lower()) and \
                   (track.lower() in r_track or r_track in track.lower()):
                    if r.get('previewUrl') and _artist_matches(known_artist, r.get('artistName', '')):
                        return r, r.get('trackId')

            # Fall back to first result with preview
            for r in results:
                if r.get('previewUrl') and _artist_matches(known_artist, r.get('artistName', '')):
                    return r, r.get('trackId')

    # Strategy 2: Search with just title
    results = search_itunes(title, limit=3)
    for r in results:
        if r.get('previewUrl') and _artist_matches(known_artist, r.get('artistName', '')):
            return r, r.get('trackId')

    # Strategy 3: Search with cleaned title
    clean_title = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', title).strip()
    if clean_title != title:
        results = search_itunes(clean_title, limit=3)
        for r in results:
            if r.get('previewUrl') and _artist_matches(known_artist, r.get('artistName', '')):
                return r, r.get('trackId')

    return None, None


# ============================================================================
# Audio processing helpers
# ============================================================================

def download_preview(url, output_path):
    """Download iTunes preview to file."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        return False


def convert_to_wav(input_path, output_path, sample_rate=16000):
    """Convert audio file to WAV with specified sample rate."""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', '1',
            '-f', 'wav',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        return False


def load_audio(wav_path, sample_rate=16000):
    """Load audio file using librosa."""
    import librosa
    signal, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    return signal


# ============================================================================
# CQT feature extraction (matching CoverHunter training)
# ============================================================================

def compute_cqt_features(signal, sample_rate=16000, hop_size=0.04, mean_size=3):
    """
    Compute CQT features matching CoverHunter training pipeline.

    Args:
        signal: Audio signal as numpy array
        sample_rate: Sample rate (16000 Hz for CoverHunter)
        hop_size: Hop size in seconds (0.04 = 25 frames/sec)
        mean_size: Downsampling factor (3 = ~8.3 frames/sec)

    Returns:
        cqt: CQT features as numpy array (time, freq=96)
    """
    from model.cqt import PyCqt, shorter

    # Create CQT extractor
    cqt_extractor = PyCqt(
        sample_rate=sample_rate,
        hop_size=hop_size,
        octave_resolution=12,
        min_freq=32,
        max_freq=sample_rate // 2,
    )

    # Compute CQT spectrogram
    cqt = cqt_extractor.compute_cqt(signal_float=signal, feat_dim_first=False)

    # Downsample with mean pooling
    if mean_size > 1:
        cqt = shorter(cqt, mean_size)

    return cqt.astype(np.float32)


# ============================================================================
# Embedding computation
# ============================================================================

def compute_embedding(model, cqt_features, device='cpu'):
    """
    Compute CoverHunter embedding from CQT features.

    Args:
        model: Loaded CoverHunter model
        cqt_features: CQT spectrogram (time, freq)
        device: Device to compute on

    Returns:
        embedding: 128-dim L2-normalized embedding
    """
    import torch

    # Prepare input tensor: (batch=1, time, freq)
    feat = torch.from_numpy(cqt_features).float().unsqueeze(0)
    feat = feat.to(device)

    # Run inference
    with torch.no_grad():
        embed, _ = model.inference(feat)

    # Convert to numpy and L2 normalize
    embed = embed.cpu().numpy().flatten()
    embed = embed / (np.linalg.norm(embed) + 1e-8)

    return embed


def process_youtube_video(youtube_id, model, device='cpu', temp_dir=None):
    """
    Process a single YouTube video: lookup -> iTunes -> preview -> CQT -> embedding.

    Args:
        youtube_id: YouTube video ID
        model: Loaded CoverHunter model
        device: Device to compute on
        temp_dir: Temporary directory for audio files

    Returns:
        dict with keys: success, embedding, track_id, error
    """
    import librosa

    result = {
        'success': False,
        'embedding': None,
        'track_id': None,
        'error': None,
    }

    # Get YouTube title
    title = get_youtube_title(youtube_id)
    if not title:
        result['error'] = 'Failed to get YouTube title'
        return result

    # Search iTunes
    track_info, track_id = smart_search_itunes(youtube_id, title)
    if not track_info or not track_info.get('previewUrl'):
        result['error'] = 'No iTunes preview found'
        return result

    result['track_id'] = track_id
    preview_url = track_info['previewUrl']

    # Download and process
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    m4a_path = os.path.join(temp_dir, f'{youtube_id}.m4a')
    wav_path = os.path.join(temp_dir, f'{youtube_id}.wav')

    try:
        # Download preview
        if not download_preview(preview_url, m4a_path):
            result['error'] = 'Failed to download preview'
            return result

        # Convert to WAV
        if not convert_to_wav(m4a_path, wav_path, sample_rate=16000):
            result['error'] = 'Failed to convert to WAV'
            return result

        # Load audio
        signal = load_audio(wav_path, sample_rate=16000)

        # Compute CQT features
        cqt = compute_cqt_features(signal, sample_rate=16000, hop_size=0.04, mean_size=3)

        # Compute embedding
        embedding = compute_embedding(model, cqt, device)

        result['success'] = True
        result['embedding'] = embedding

    except Exception as e:
        result['error'] = str(e)

    finally:
        # Clean up temp files
        for path in [m4a_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

    return result


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(input_file, output_file, checkpoint_dir, progress_file=None, device='cpu'):
    """
    Run the full pipeline on a list of YouTube IDs.

    Args:
        input_file: Path to file with YouTube IDs (one per line)
        output_file: Path to output CSV file
        checkpoint_dir: Path to CoverHunter checkpoint directory
        progress_file: Path to progress JSON file (for resumability)
        device: Device to use ('cpu', 'cuda', 'mps')
    """
    import torch
    from model.utils import load_model

    # Load model
    print(f"Loading CoverHunter model from {checkpoint_dir}...")
    model = load_model(checkpoint_dir, device)
    print(f"Model loaded on {device}")

    # Load progress if exists
    progress = {'processed': [], 'failed': {}, 'track_ids': {}}
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"Resuming from {len(progress['processed'])} processed videos")

    processed_set = set(progress['processed'])

    # Load YouTube IDs
    with open(input_file, 'r') as f:
        youtube_ids = [line.strip() for line in f if line.strip()]
    print(f"Total videos: {len(youtube_ids)}")

    # Open output file in append mode
    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    out_f = open(output_file, 'a')
    if write_header:
        out_f.write('youtube_id,embeddings\n')

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        for i, vid in enumerate(youtube_ids):
            if vid in processed_set:
                continue

            print(f"[{i+1}/{len(youtube_ids)}] Processing {vid}...", end=' ')

            result = process_youtube_video(vid, model, device, temp_dir)

            if result['success']:
                # Write to CSV
                emb_str = ' '.join(f'{x:.6f}' for x in result['embedding'])
                out_f.write(f'{vid},"[ {emb_str} ]"\n')
                out_f.flush()

                progress['processed'].append(vid)
                if result['track_id']:
                    progress['track_ids'][vid] = result['track_id']
                print(f"OK (trackId={result['track_id']})")
            else:
                progress['failed'][vid] = result['error']
                print(f"FAILED: {result['error']}")

            # Save progress periodically
            if progress_file and (len(progress['processed']) % 10 == 0):
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)

            # Rate limit
            time.sleep(0.2)

    finally:
        out_f.close()

        # Save final progress
        if progress_file:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

    # Print summary
    total = len(youtube_ids)
    success = len(progress['processed'])
    failed = len(progress['failed'])
    print(f"\nComplete: {success}/{total} successful, {failed} failed")


def main():
    # Default checkpoint: check persistent volume first, then local
    persistent_checkpoint = '/app/data/coverhunter_checkpoint'
    local_checkpoint = os.path.join(SCRIPT_DIR, 'checkpoint')
    default_checkpoint = persistent_checkpoint if os.path.exists(os.path.join(persistent_checkpoint, 'hparams.yaml')) else local_checkpoint

    parser = argparse.ArgumentParser(description='Compute CoverHunter embeddings from iTunes previews')
    parser.add_argument('--input', default='youtube_ids.txt', help='Input file with YouTube IDs')
    parser.add_argument('--output', default='vectors_coverhunter.csv', help='Output CSV file')
    parser.add_argument('--checkpoint', default=default_checkpoint, help='Checkpoint directory')
    parser.add_argument('--progress', default='progress.json', help='Progress file for resumability')
    parser.add_argument('--device', default='cpu', help='Device (cpu, cuda, mps)')
    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        checkpoint_dir=args.checkpoint,
        progress_file=args.progress,
        device=args.device,
    )


if __name__ == '__main__':
    main()
